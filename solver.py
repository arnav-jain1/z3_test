import z3

class MatMulShapeInference:
    """
    Infers valid symbolic LHS and output shapes for np.matmul given a known RHS.

    This class translates NumPy's matmul broadcasting and core multiplication
    rules into a set of logical constraints that can be solved by Z3.
    """

    def __init__(self, rhs_shape: tuple[int, ...]):
        """
        Initializes the inference engine with a known RHS shape.

        Args:
            rhs_shape: A tuple of positive integers representing the shape
                       of the right-hand side tensor.
        """
        if not isinstance(rhs_shape, tuple) or not all(isinstance(d, int) and d > 0 for d in rhs_shape):
            raise ValueError("rhs_shape must be a tuple of positive integers.")
        
        self.rhs_shape = rhs_shape
        self.rhs_rank = len(rhs_shape)
        self.results = []
        self._analyzed_ranks = set()

    def solve(self, max_lhs_rank: int = 7) -> list[dict]:
        """
        Computes all valid symbolic LHS shapes up to a maximum rank.

        Args:
            max_lhs_rank: The maximum number of dimensions to check for the LHS.

        Returns:
            A list of dictionaries, where each dictionary describes a valid
            class of LHS shapes and the corresponding output shape.
        """
        print(f"\n--- Analyzing RHS Shape: {self.rhs_shape} ---\n")

        # 1. Handle cases where the RHS is a 1-D vector.
        if self.rhs_rank == 1:
            self._solve_for_1d_rhs()

        # 2. Handle cases where the LHS is a 1-D vector.
        self._solve_for_1d_lhs()
        
        # 3. Handle the general case where both LHS and RHS are >= 2D.
        #    We iterate through possible ranks for the LHS.
        for lhs_rank in range(2, max_lhs_rank + 1):
            self._solve_for_nd_lhs(lhs_rank)

        # 4. Generalize and consolidate the results
        self._summarize_results()
        
        return self.results

    def _solve_for_1d_rhs(self):
        """Rule: (..., M, K) @ (K,) -> (..., M)"""
        if self._is_rank_analyzed('1d_rhs'): return

        solver = z3.Solver()
        K_rhs = self.rhs_shape[0]
        
        # We model the simplest valid LHS: a 2D matrix (M, K)
        M, K = z3.Ints("M K")
        solver.add(M > 0, K > 0)
        
        # Core matmul constraint
        solver.add(K == K_rhs)

        if solver.check() == z3.sat:
            self.results.append({
                "type": "vector_rhs",
                "description": "LHS is a stack of matrices contracting with a 1-D RHS vector.",
                "lhs_symbolic": f"(..., M, {K_rhs})",
                "output_symbolic": "(..., M)",
                "constraints": f"The last dimension of the LHS must be {K_rhs}."
            })
        self._mark_rank_analyzed('1d_rhs')

    def _solve_for_1d_lhs(self):
        """Rule: (K,) @ (..., K, N) -> (..., N)"""
        if self._is_rank_analyzed('1d_lhs'): return

        # Case 1: 1D @ 1D -> scalar output
        if self.rhs_rank == 1:
            K_rhs = self.rhs_shape[0]
            self.results.append({
                "type": "scalar_dot",
                "description": "1D LHS vector dot product with 1D RHS vector.",
                "lhs_symbolic": f"({K_rhs},)",
                "output_symbolic": "() (scalar)",
                "constraints": f"The LHS must be a vector of shape ({K_rhs},)."
            })
            self._mark_rank_analyzed('1d_lhs')
            return

        # Case 2: 1D @ >=2D
        if self.rhs_rank >= 2:
            K_rhs = self.rhs_shape[-2]
            output_batch = self.rhs_shape[:-2]
            N = self.rhs_shape[-1]

            output_shape_str = f"{output_batch + (N,)}".replace(",)", ")")

            self.results.append({
                "type": "vector_lhs",
                "description": "1D LHS vector contracting with a stack of RHS matrices.",
                "lhs_symbolic": f"({K_rhs},)",
                "output_symbolic": output_shape_str,
                "constraints": f"The LHS must be a vector of shape ({K_rhs},)."
            })
        self._mark_rank_analyzed('1d_lhs')


    def _solve_for_nd_lhs(self, lhs_rank: int):
        """Rule: (B_lhs, M, K) @ (B_rhs, K, N) -> (B_out, M, N)"""
        if self.rhs_rank < 2 or self._is_rank_analyzed(lhs_rank):
            return

        solver = z3.Solver()
        
        # Create symbolic variables for the LHS shape
        lhs_shape_vars = [z3.Int(f"lhs{i}") for i in range(lhs_rank)]
        solver.add([d > 0 for d in lhs_shape_vars])

        # Deconstruct shapes into batch and matrix parts
        lhs_batch_vars = lhs_shape_vars[:-2]
        M, K = lhs_shape_vars[-2], lhs_shape_vars[-1]

        rhs_batch_dims = self.rhs_shape[:-2]
        K_rhs, N = self.rhs_shape[-2], self.rhs_shape[-1]
        
        # Add core matmul constraint
        solver.add(K == K_rhs)

        # Add broadcasting constraints for batch dimensions
        broadcast_constraints, _ = self._get_broadcast_constraints(lhs_batch_vars, list(rhs_batch_dims))
        solver.add(broadcast_constraints)

        # Check if a valid shape of this rank exists
        if solver.check() == z3.sat:
            self._add_nd_result(lhs_rank, lhs_batch_vars, rhs_batch_dims, K_rhs, N)
        
        self._mark_rank_analyzed(lhs_rank)

    def _get_broadcast_constraints(self, batch1_vars, batch2_dims):
        """Generates Z3 constraints for broadcasting two batch shapes."""
        constraints = []
        output_batch_sym = []
        
        len1, len2 = len(batch1_vars), len(batch2_dims)
        max_len = max(len1, len2)

        for i in range(max_len):
            # Align from the right
            idx1 = len1 - 1 - i
            idx2 = len2 - 1 - i
            
            dim1_name = f"d{idx1}"

            if idx1 >= 0 and idx2 >= 0:
                # Both shapes have a dimension here
                d1 = batch1_vars[idx1]
                d2 = batch2_dims[idx2]
                constraints.append(z3.Or(d1 == 1, d1 == d2))
                output_batch_sym.append(f"broadcast({dim1_name}, {d2})")
            elif idx1 >= 0:
                # Only batch1 has a dimension (the '...' part)
                output_batch_sym.append(dim1_name)
            elif idx2 >= 0:
                # Only batch2 has a dimension
                output_batch_sym.append(str(batch2_dims[idx2]))
        
        return constraints, list(reversed(output_batch_sym))
    
    def _add_nd_result(self, lhs_rank, lhs_batch_vars, rhs_batch_dims, K_rhs, N):
        """Formats and stores a valid symbolic shape configuration."""
        
        # Build human-readable constraints
        constraints = []
        len_lhs_b, len_rhs_b = len(lhs_batch_vars), len(rhs_batch_dims)
        for i in range(min(len_lhs_b, len_rhs_b)):
            # Align from right
            lhs_idx = len_lhs_b - 1 - i
            rhs_idx = len_rhs_b - 1 - i
            dim_name = f"d{lhs_idx}"
            rhs_val = rhs_batch_dims[rhs_idx]
            constraints.append(f"{dim_name} must be 1 or {rhs_val}")
        
        # Build symbolic shape strings
        lhs_batch_sym_parts = []
        if len_lhs_b > len_rhs_b:
            lhs_batch_sym_parts.append("...")
        
        start_idx = len_lhs_b - len_rhs_b if len_lhs_b > len_rhs_b else 0
        for i in range(start_idx, len_lhs_b):
            lhs_batch_sym_parts.append(f"d{i}")

        lhs_batch_str = ", ".join(lhs_batch_sym_parts)
        lhs_sym = f"({lhs_batch_str}, M, {K_rhs})".replace("(, ", "(")

        _, out_batch_sym = self._get_broadcast_constraints(lhs_batch_vars, rhs_batch_dims)
        out_batch_str = ", ".join(out_batch_sym)
        out_sym = f"({out_batch_str}, M, {N})".replace("(, ", "(")

        self.results.append({
            "type": "n_dim",
            "lhs_rank": lhs_rank,
            "description": f"A {lhs_rank}-D LHS broadcasting with the {self.rhs_rank}-D RHS.",
            "lhs_symbolic": lhs_sym,
            "output_symbolic": out_sym,
            "constraints": ", ".join(reversed(constraints)) or "Leading batch dimensions are unconstrained."
        })

    def _summarize_results(self):
        """Consolidates N-D results into a single generalized rule if possible."""
        nd_results = [r for r in self.results if r.get("type") == "n_dim"]
        if not nd_results:
            return

        # Find the result with the highest rank to generalize from
        representative = max(nd_results, key=lambda r: r['lhs_rank'])

        # Create a single, generalized entry
        generalized_result = {
            "type": "n_dim_generalized",
            "description": "LHS is a stack of matrices that must broadcast with the RHS stack.",
            "lhs_symbolic": representative["lhs_symbolic"],
            "output_symbolic": representative["output_symbolic"],
            "constraints": representative["constraints"]
        }
        
        # Remove the individual n_dim results and add the summary
        self.results = [r for r in self.results if r.get("type") != "n_dim"]
        self.results.append(generalized_result)

    def _is_rank_analyzed(self, rank_key):
        return rank_key in self._analyzed_ranks

    def _mark_rank_analyzed(self, rank_key):
        self._analyzed_ranks.add(rank_key)

def print_results(results):
    """Helper function to pretty-print the analysis results."""
    if not results:
        print("No valid LHS shapes could be determined.")
        return
        
    for i, res in enumerate(results):
        print(f"[{i+1}] Found Valid Shape Class: {res['description']}")
        print(f"    {'LHS Shape:':<18} {res['lhs_symbolic']}")
        print(f"    {'Output Shape:':<18} {res['output_symbolic']}")
        print(f"    {'Constraints:':<18} {res['constraints']}")
        print("-" * 20)

if __name__ == '__main__':
    # --- Example 1: User's requested shape ---
    # RHS has batch dimensions (2,) and matrix dimensions (3, 4)
    inference1 = MatMulShapeInference(rhs_shape=(2, 3, 4))
    results1 = inference1.solve()
    print(results1)
    print_results(results1)

    # --- Example 2: Simple 2D RHS ---
    # RHS has no batch dimensions and matrix dimensions (5, 6)
    inference2 = MatMulShapeInference(rhs_shape=(5, 6))
    results2 = inference2.solve()
    print_results(results2)

    # --- Example 3: 1D Vector RHS ---
    inference3 = MatMulShapeInference(rhs_shape=(7,))
    results3 = inference3.solve()
    print_results(results3)
    
    # --- Example 4: RHS with more batch dimensions ---
    inference4 = MatMulShapeInference(rhs_shape=(10, 8, 5, 6))
    results4 = inference4.solve()
    print_results(results4)