from z3 import *

class NumpySolver:
    def __init__(self, lhs, rhs, lhs_rank=None, rhs_rank=None):
        self.lhs = lhs if lhs else []
        self.rhs = rhs if rhs else []

        if (not lhs) and (not lhs_rank) and (not rhs) and (not rhs_rank):
            print("Need at least one of lhs, lhs_rank, rhs, rhs_rank")
            raise RuntimeError         

        self.lhs_rank = lhs_rank if lhs_rank else len(lhs)
        self.rhs_rank = rhs_rank if rhs_rank else len(rhs)
        
        self.solver = Solver()
    
    def solve_matmul(self):
        # This is for vectors, essentially we add a dim that works and remove it later
        self.rhs_vec = False
        self.lhs_vec = False
        if self.rhs_rank == 1:
            self.rhs_vec = True
            self.rhs.append(int)
            self.rhs_rank += 1
        if self.lhs_rank == 1:
            self.lhs_vec = True
            self.lhs.insert(0, int)
            self.lhs_rank += 1
        self.rank = max(self.lhs_rank, self.rhs_rank)

        if len(self.lhs) < self.rank:
            for _ in range(self.rank - len(self.lhs)):
                self.lhs.insert(0, int)

        if len(self.rhs) < self.rank:
            for _ in range(self.rank - len(self.rhs)):
                self.rhs.insert(0, int)
        
        # if self.lhs_rank < len(self.lhs):
        #     print("LHS rank must be >= length of lhs")
        #     print(self.lhs_rank, self.lhs)
        #     raise RuntimeError
        # elif self.rhs_rank < len(self.rhs):
        #     print(self.rhs_rank, self.rhs)
        #     print("RHS rank must be >= length of rhs")
        #     raise RuntimeError


        self.output = [int for _ in range(self.rank)]
        self.lhs_vars = [Int(f"lhs_{i}") for i in range(self.rank)]
        self.rhs_vars = [Int(f"rhs_{i}") for i in range(self.rank)]


        self._solve_matmul()
        lhs, rhs, output = None, None, None
        if self.solver.check() == sat:
            lhs, rhs = self.summarize_nd_sides()
            output = self.output
            while self.lhs_rank < len(lhs):
                lhs.pop(0)
            while self.rhs_rank < len(rhs):
                rhs.pop(0)

            if self.lhs_vec and self.rhs_vec:
                output = int
                lhs.pop(0)
                rhs.pop(-1)
                
            elif self.lhs_vec:
                output.pop(-2)
                lhs.pop(0)
            elif self.rhs_vec:
                output.pop(-1)
                rhs.pop(-1)

        return lhs, rhs, output

    def solve_broadcast(self):
        self.rank = max(self.lhs_rank, self.rhs_rank)

        if len(self.lhs) < self.rank:
            for _ in range(self.rank - len(self.lhs)):
                self.lhs.insert(0, int)

        if len(self.rhs) < self.rank:
            for _ in range(self.rank - len(self.rhs)):
                self.rhs.insert(0, int)
        
        # if self.lhs_rank < len(self.lhs):
        #     print("LHS rank must be >= length of lhs")
        #     print(self.lhs_rank, self.lhs)
        #     raise RuntimeError
        # elif self.rhs_rank < len(self.rhs):
        #     print(self.rhs_rank, self.rhs)
        #     print("RHS rank must be >= length of rhs")
        #     raise RuntimeError


        self.output = [int for _ in range(self.rank)]
        self.lhs_vars = [Int(f"lhs_{i}") for i in range(self.rank)]
        self.rhs_vars = [Int(f"rhs_{i}") for i in range(self.rank)]


        self._solve_broadcast()
        lhs, rhs, output = None, None, None
        if self.solver.check() == sat:
            lhs, rhs = self.summarize_nd_sides()
            output = self.output
            while self.lhs_rank < len(lhs):
                lhs.pop(0)
            while self.rhs_rank < len(rhs):
                rhs.pop(0)

        return lhs, rhs, output
    
    def _solve_matmul(self):

        if isinstance(self.lhs[-2], int):
            self.solver.add(self.lhs_vars[-2] == self.lhs[-2])
        elif (isinstance(self.lhs[-2], tuple)):
            or_clauses = [self.lhs_vars[-2] == val for val in self.lhs[-2]]
            self.solver.add(Or(or_clauses))
        if isinstance(self.lhs[-1], int):
            self.solver.add(self.lhs_vars[-1] == self.lhs[-1])
        elif (isinstance(self.lhs[-1], tuple)):
            or_clauses = [self.lhs_vars[-1] == val for val in self.lhs[-1]]
            self.solver.add(Or(or_clauses))

        if isinstance(self.rhs[-2], int):
            self.solver.add(self.rhs_vars[-2] == self.rhs[-2])
        elif (isinstance(self.rhs[-2], tuple)):
            or_clauses = [self.rhs_vars[-2] == val for val in self.rhs[-2]]
            self.solver.add(Or(or_clauses))
        if isinstance(self.rhs[-1], int):
            self.solver.add(self.rhs_vars[-1] == self.rhs[-1])
        elif (isinstance(self.rhs[-1], tuple)):
            or_clauses = [self.rhs_vars[-1] == val for val in self.rhs[-1]]
            self.solver.add(Or(or_clauses))
        

        if not ((self.lhs[-1] is int) and (self.rhs[-2] is int)):
            self.solver.add(self.lhs_vars[-1] == self.rhs_vars[-2])
        

        self.output[-2] = self.lhs[-2]
        self.output[-1] = self.rhs[-1]

        lhs_broadcasting = self.lhs[:-2]
        rhs_broadcasting = self.rhs[:-2]
        lhs_broadcasting_vars = self.lhs_vars[:-2]
        rhs_broadcasting_vars = self.rhs_vars[:-2]


        lhs_dim = len(lhs_broadcasting)
        rhs_dim = len(rhs_broadcasting)

        i = 3

        while i <= self.rank:
            idx = -i

            lhs_d = self.lhs[idx]
            rhs_d = self.rhs[idx]
            lhs_var = self.lhs_vars[idx]
            rhs_var = self.rhs_vars[idx]

            # Edge case both 1
            if isinstance(lhs_d, int) and lhs_d == 1 and isinstance(rhs_d, int) and rhs_d == 1:
                self.solver.add(lhs_var == 1)
                self.solver.add(rhs_var == 1)
                self.solver.add(lhs_var == rhs_var)
                self.output[idx] = 1

            # Edge case both tuple
            elif isinstance(lhs_d, tuple) and isinstance(rhs_d, tuple):
                lhs_or_clauses = [lhs_var == val for val in lhs_d]
                rhs_or_clauses = [rhs_var == val for val in rhs_d]

                possible_outputs = set()
                possible_outputs.update(lhs_d)
                possible_outputs.update(rhs_d)
                possible_outputs = tuple(sorted(list(possible_outputs)))
                self.output[idx] = possible_outputs

                self.solver.add(Or(lhs_or_clauses))
                self.solver.add(Or(rhs_or_clauses))
                self.solver.add(Or([lhs_var == 1, rhs_var == 1, lhs_var == rhs_var]))

            # Case 1 is lhs being an int, add that lhs has to be that int to the solver
            elif isinstance(lhs_d, int):
                self.solver.add(lhs_var == lhs_d)

                # If rhs is an int, add that to the solver
                # If it is a tup, add the possible vals
                # If it is any int, then just skip
                if isinstance(rhs_d, int):
                    self.solver.add(rhs_var == rhs_d)
                elif isinstance(rhs_d, tuple):
                    or_clauses = [rhs_var == val for val in rhs_d]
                    self.solver.add(Or(or_clauses))
                
                # If lhs is not 1, then add that rhs has to be 1 or lhs, output has to have it be lhs
                if lhs_d != 1:
                    self.solver.add(Or(rhs_var == 1, rhs_var == lhs_d))
                    self.output[idx] = lhs_d
                # If lhs is 1, then rhs can be anything and output is the rhs
                else:
                    self.output[idx] = rhs_d

            # Case 2: RHS is a tup
            # Issue here is if lhs is 1 or 5, then if lhs is 1 rhs can be anything if lhs is 5 then rhs has to be 1 or 5
            # False positives better than false negs, so let rhs be anything, maybe add a warning here? Look at later
            # Also the output is anything
            elif isinstance(lhs_d, tuple):
                or_clauses = [lhs_var == val for val in lhs_d]
                self.solver.add(Or(or_clauses))

                # If rhs is an int, add that to the solver 
                # If it is any int, then just skip
                if isinstance(rhs_d, int):
                    self.solver.add(rhs_var == rhs_d)
                    # If the int is not 1, then the output is rhs and if the int is 1, then the output is lhs
                    if rhs_d != 1:
                        self.solver.add(Or(lhs_var == rhs_var, lhs_var == 1))
                        self.output[idx] = rhs_d
                    else:
                        self.output[idx] = lhs_d
                # This is where it becomes tricky. Lets say lhs is (1,3) and rhs is anything. Then if lhs is 1, rhs is anything
                # but if lhs is 3, then rhs has to be 3. For now we just keep it anything and continue along
                else:
                    pass
            # Last case, lhs is any int
            elif lhs_d == int:
                # If rhs is an int, add that to the solver and the fact that rhs == lhs
                # If it is a tup, then similar for before, just cont
                # If it is any int, then just skip
                if isinstance(rhs_d, int):
                    self.solver.add(rhs_var == rhs_d)
                    # If rhs is not 1, then lhs has to be 1 or rhs, if it is 1 then cont
                    if rhs_d != 1:
                        self.solver.add(Or(lhs_var == 1, lhs_var == rhs_d))
                        self.output[idx] = rhs_d

                elif isinstance(rhs_d, tuple):
                    or_clauses = [rhs_var == val for val in rhs_d]
                    self.solver.add(Or(or_clauses))
                    
            i += 1
    
    def _solve_broadcast(self):
        i = 1

        while i <= self.rank:
            idx = -i

            lhs_d = self.lhs[idx]
            rhs_d = self.rhs[idx]
            lhs_var = self.lhs_vars[idx]
            rhs_var = self.rhs_vars[idx]

            # Edge case both 1
            if isinstance(lhs_d, int) and lhs_d == 1 and isinstance(rhs_d, int) and rhs_d == 1:
                self.solver.add(lhs_var == 1)
                self.solver.add(rhs_var == 1)
                self.solver.add(lhs_var == rhs_var)
                self.output[idx] = 1

            # Edge case both tuple
            elif isinstance(lhs_d, tuple) and isinstance(rhs_d, tuple):
                lhs_or_clauses = [lhs_var == val for val in lhs_d]
                rhs_or_clauses = [rhs_var == val for val in rhs_d]

                possible_outputs = set()
                possible_outputs.update(lhs_d)
                possible_outputs.update(rhs_d)
                possible_outputs = tuple(sorted(list(possible_outputs)))
                self.output[idx] = possible_outputs

                self.solver.add(Or(lhs_or_clauses))
                self.solver.add(Or(rhs_or_clauses))
                self.solver.add(Or([lhs_var == 1, rhs_var == 1, lhs_var == rhs_var]))

            # Case 1 is lhs being an int, add that lhs has to be that int to the solver
            elif isinstance(lhs_d, int):
                self.solver.add(lhs_var == lhs_d)

                # If rhs is an int, add that to the solver
                # If it is a tup, add the possible vals
                # If it is any int, then just skip
                if isinstance(rhs_d, int):
                    self.solver.add(rhs_var == rhs_d)
                elif isinstance(rhs_d, tuple):
                    or_clauses = [rhs_var == val for val in rhs_d]
                    self.solver.add(Or(or_clauses))
                
                # If lhs is not 1, then add that rhs has to be 1 or lhs, output has to have it be lhs
                if lhs_d != 1:
                    self.solver.add(Or(rhs_var == 1, rhs_var == lhs_d))
                    self.output[idx] = lhs_d
                # If lhs is 1, then rhs can be anything and output is the rhs
                else:
                    self.output[idx] = rhs_d

            # Case 2: RHS is a tup
            # Issue here is if lhs is 1 or 5, then if lhs is 1 rhs can be anything if lhs is 5 then rhs has to be 1 or 5
            # False positives better than false negs, so let rhs be anything, maybe add a warning here? Look at later
            # Also the output is anything
            elif isinstance(lhs_d, tuple):
                or_clauses = [lhs_var == val for val in lhs_d]
                self.solver.add(Or(or_clauses))

                # If rhs is an int, add that to the solver 
                # If it is any int, then just skip
                if isinstance(rhs_d, int):
                    self.solver.add(rhs_var == rhs_d)
                    # If the int is not 1, then the output is rhs and if the int is 1, then the output is lhs
                    if rhs_d != 1:
                        self.solver.add(Or(lhs_var == rhs_var, lhs_var == 1))
                        self.output[idx] = rhs_d
                    else:
                        self.output[idx] = lhs_d
                # This is where it becomes tricky. Lets say lhs is (1,3) and rhs is anything. Then if lhs is 1, rhs is anything
                # but if lhs is 3, then rhs has to be 3. For now we just keep it anything and continue along
                else:
                    pass
            # Last case, lhs is any int
            elif lhs_d == int:
                # If rhs is an int, add that to the solver and the fact that rhs == lhs
                # If it is a tup, then similar for before, just cont
                # If it is any int, then just skip
                if isinstance(rhs_d, int):
                    self.solver.add(rhs_var == rhs_d)
                    # If rhs is not 1, then lhs has to be 1 or rhs, if it is 1 then cont
                    if rhs_d != 1:
                        self.solver.add(Or(lhs_var == 1, lhs_var == rhs_d))
                        self.output[idx] = rhs_d

                elif isinstance(rhs_d, tuple):
                    or_clauses = [rhs_var == val for val in rhs_d]
                    self.solver.add(Or(or_clauses))
                    
            i += 1

    def _parse_var_node(self, var_node):
        """Helper to parse a Z3 variable node into its side ('lhs'/'rhs') and index."""
        var_str = str(var_node)
        side, index_str = var_str.split('_')
        return side, int(index_str)

    def summarize_nd_sides(self):
        """
        Summarizes possible dimension values by repeatedly finding valid models.

        This function iteratively asks the Z3 solver for a satisfying model.
        For each model found, it records the concrete values for each dimension.
        It then adds a constraint to the solver to exclude that specific solution,
        forcing the next check to find a *different* model. This process repeats
        until no more solutions can befound.

        A heuristic is included: if a dimension is found to have more than 5
        possible values, it's considered unconstrained (effectively 'any integer')
        to prevent excessive looping in under-constrained scenarios.
        """
        # --- PASS 1: Iteratively find all possible solutions ---

        # Store sets of all discovered values for each dimension
        lhs_solutions = [set() for _ in range(self.rank)]
        rhs_solutions = [set() for _ in range(self.rank)]

        # Track variables that are deemed unconstrained to speed up solving
        unconstrained_vars = set()
        
        # Define the threshold for considering a variable unconstrained
        SOLUTION_LIMIT = 5

        while self.solver.check() == sat:
            model = self.solver.model()
            
            # This list will hold clauses to block the current solution
            blocking_clauses = []

            # Process LHS variables
            for i, var in enumerate(self.lhs_vars):
                if var in unconstrained_vars:
                    continue # Skip variables we've already marked as unconstrained
                
                # Get the concrete value from the model
                val = model.eval(var, model_completion=True).as_long()
                lhs_solutions[i].add(val)
                
                # Add a clause to block this specific value in the next iteration
                blocking_clauses.append(var != val)

                # Heuristic: If we find too many possibilities, treat it as 'any int'
                if len(lhs_solutions[i]) > SOLUTION_LIMIT:
                    unconstrained_vars.add(var)

            # Process RHS variables
            for i, var in enumerate(self.rhs_vars):
                if var in unconstrained_vars:
                    continue
                
                val = model.eval(var, model_completion=True).as_long()
                rhs_solutions[i].add(val)
                blocking_clauses.append(var != val)
                
                if len(rhs_solutions[i]) > SOLUTION_LIMIT:
                    unconstrained_vars.add(var)

            # Add the combined blocking constraint to the solver. This is crucial
            # to ensure we find a *new* solution in the next loop iteration.
            if not blocking_clauses:
                # This can happen if all variables become unconstrained
                break
                
            self.solver.add(Or(blocking_clauses))

        # --- PASS 2: Format the collected solutions into the final output ---

        lhs_output = [None] * self.rank
        rhs_output = [None] * self.rank

        for i in range(self.rank):
            # Format LHS output
            lhs_var = self.lhs_vars[i]
            solutions = lhs_solutions[i]
            if lhs_var in unconstrained_vars or not solutions:
                lhs_output[i] = int # No specific values or too many -> any int
            elif len(solutions) == 1:
                lhs_output[i] = solutions.pop() # A single, concrete integer
            else:
                lhs_output[i] = tuple(sorted(list(solutions))) # A tuple of possibilities

            # Format RHS output
            rhs_var = self.rhs_vars[i]
            solutions = rhs_solutions[i]
            if rhs_var in unconstrained_vars or not solutions:
                rhs_output[i] = int
            elif len(solutions) == 1:
                rhs_output[i] = solutions.pop()
            else:
                rhs_output[i] = tuple(sorted(list(solutions)))
                
        return lhs_output, rhs_output
    
import traceback

# --- TEST SUITE ---
def run_tests():
    """
    Runs a suite of tests for the MatMulUnknown class.
    """
    test_cases = [
        # === BASIC SUCCESS (LHS/RHS MOSTLY KNOWN) ===
        {"name": "T01: Simple Matmul (Both Known)", "lhs_shape": [2, 3], "rhs_shape": [3, 4], "expected_lhs": [2, 3], "expected_rhs": [3, 4], "expected_output": [2, 4]},
        {"name": "T02: Broadcasting (Both Known)", "lhs_shape": [5, 1, 3, 4], "rhs_shape": [7, 4, 8], "expected_lhs": [5, 1, 3, 4], "expected_rhs": [7, 4, 8], "expected_output": [5, 7, 3, 8]},
        {"name": "T03: Resolve LHS Tuple (Matmul Dim)", "lhs_shape": [5, (1, 7)], "rhs_shape": [7, 8], "expected_lhs": [5, 7], "expected_rhs": [7, 8], "expected_output": [5, 8]},
        {"name": "T04: Resolve RHS Tuple (Broadcast Dim)", "lhs_shape": [8, 4, 5], "rhs_shape": [(1, 8), 5, 6], "expected_lhs": [8, 4, 5], "expected_rhs": [(1, 8), 5, 6], "expected_output": [8, 4, 6]},
        
        {"name": "T05: Matrix-Vector Multiplication", "lhs_shape": [15, 20, 10], "rhs_shape": [10], "rhs_rank": 1, "expected_lhs": [15, 20, 10], "expected_rhs": [10], "expected_output": [15, 20]},
        {"name": "T06: Vector-Matrix Multiplication", "lhs_shape": [10], "rhs_shape": [12, 10, 20], "lhs_rank": 1, "expected_lhs": [10], "expected_rhs": [12, 10, 20], "expected_output": [12, 20]},
        {"name": "T26: Matrix-Vector Multiplication", "lhs_shape": [15, 20, 10], "rhs_shape": [(1,10)], "rhs_rank": 1, "expected_lhs": [15, 20, 10], "expected_rhs": [10], "expected_output": [15, 20]},
        {"name": "T27: Vector-Matrix Multiplication", "lhs_shape": [int], "rhs_shape": [12, 10, 20], "lhs_rank": 1, "expected_lhs": [10], "expected_rhs": [12, 10, 20], "expected_output": [12, 20]},
        {"name": "T28: Vector-Matrix Multiplication", "lhs_shape": [int], "rhs_shape": [10], "lhs_rank": 1, "expected_lhs": [10], "expected_rhs": [10], "expected_output": int},
        {"name": "T29: Int output", "lhs_shape": [int], "rhs_shape": [(1,10)], "lhs_rank": 1, "expected_lhs": [(1,10)], "expected_rhs": [(1,10)], "expected_output": int},
        {"name": "T30: Int output", "lhs_shape": [int], "rhs_shape": [int], "lhs_rank": 1, "expected_lhs": [int], "expected_rhs": [int], "expected_output": int},
        {"name": "T07: LHS Fully Unknown (Rank Given)", "lhs_shape": [], "lhs_rank": 2, "rhs_shape": [5, 6], "expected_lhs": [int, 5], "expected_rhs": [5, 6], "expected_output": [int, 6]},
        {"name": "T08: RHS Fully Unknown (Rank Given)", "lhs_shape": [7, 8], "rhs_shape": [], "rhs_rank": 2, "expected_lhs": [7, 8], "expected_rhs": [8, int], "expected_output": [7, int]},
        {"name": "T11: Rank Promotion on Unknown LHS", "lhs_shape": [], "lhs_rank": 4, "rhs_shape": [10, 20, 30], "expected_lhs": [int, (1, 10), int, 20], "expected_rhs": [10, 20, 30], "expected_output": [int, 10, int, 30]},
        {"name": "T17: Only Ranks Provided", "lhs_shape": [], "lhs_rank": 3, "rhs_shape": [], "rhs_rank": 3, "expected_lhs": [int, int, int], "expected_rhs": [int, int, int], "expected_output": [int, int, int]},
        {"name": "T23: Scalar-like Output", "lhs_shape": [1, 10], "rhs_shape": [10, 1], "expected_lhs": [1, 10], "expected_rhs": [10, 1], "expected_output": [1, 1]},

        # === ONE SIDE PARTIALLY/FULLY UNKNOWN ===
        {"name": "T09: Infer LHS from RHS Broadcast", "lhs_shape": [int, 1, 3, 4], "rhs_shape": [7, 4, 8], "expected_lhs": [int, 1, 3, 4], "expected_rhs": [7, 4, 8], "expected_output": [int, 7, 3, 8]},
        {"name": "T10: Infer RHS from LHS Broadcast", "lhs_shape": [5, 1, 3, 4], "rhs_shape": [int, 4, 8], "expected_lhs": [5, 1, 3, 4], "expected_rhs": [int, 4, 8], "expected_output": [5, int, 3, 8]},
        
        # === BOTH SIDES UNKNOWN ===
        {"name": "T12: Both Sides Partially Unknown", "lhs_shape": [10, int], "rhs_shape": [int, 12], "expected_lhs": [10, int], "expected_rhs": [int, 12], "expected_output": [10, 12]},
        {"name": "T13: Broadcasting With Both Unknown", "lhs_shape": [8, int, 6, 7], "rhs_shape": [int, 5, 7, 9], "expected_lhs": [8, (1, 5), 6, 7], "expected_rhs": [(1, 8), 5, 7, 9], "expected_output": [8, 5, 6, 9]},
        {"name": "T14: High Rank Broadcast (Both Unknown)", "lhs_shape": [2, 3, int, 5, 6], "rhs_shape": [int, 3, 8, 6, 7], "expected_lhs": [2, 3, (1, 8), 5, 6], "expected_rhs": [(1, 2), 3, 8, 6, 7], "expected_output": [2, 3, 8, 5, 7]},
        {"name": "T15: Unresolvable Tuple Resolves to 1", "lhs_shape": [(1, 6), 7, 8], "rhs_shape": [9, 8, 9], "expected_lhs": [1, 7, 8], "expected_rhs": [9, 8, 9], "expected_output": [9, 7, 9]},
        {"name": "T16: Both Ranks Promoted & Unknown", "lhs_shape": [5, 6], "lhs_rank": 4, "rhs_shape": [(1,5), 6, 7], "rhs_rank": 4, "expected_lhs": [int, int, 5, 6], "expected_rhs": [int, (1, 5), 6, 7], "expected_output": [int, int, 5, 7]},

        # === FAILURE CASES (UNSAT) ===
        {"name": "T18: Simple Matmul Mismatch", "lhs_shape": [2, 3], "rhs_shape": [4, 5], "expected_lhs": None, "expected_rhs": None, "expected_output": None},
        {"name": "T19: Broadcasting Mismatch", "lhs_shape": [5, 3, 2], "rhs_shape": [6, 2, 4], "expected_lhs": None, "expected_rhs": None, "expected_output": None},
        {"name": "T20: Tuple vs Int Broadcast Mismatch", "lhs_shape": [(2, 5), 3, 4], "rhs_shape": [6, 4, 8], "expected_lhs": None, "expected_rhs": None, "expected_output": None},
        {"name": "T21: Tuple vs Tuple Matmul Mismatch", "lhs_shape": [10, (1, 5)], "rhs_shape": [(2, 6), 20], "expected_lhs": None, "expected_rhs": None, "expected_output": None},

        # === EDGE CASES ===
        {"name": "T24: Broadcasting with Same Dim and 1", "lhs_shape": [5, int, 3, 4], "rhs_shape": [5, 1, 4, 8], "expected_lhs": [5, int, 3, 4], "expected_rhs": [5, 1, 4, 8], "expected_output": [5, int, 3, 8]},
        {"name": "T25: Full Pass-through Unknowns", "lhs_shape": [int, int], "rhs_shape": [int, int], "expected_lhs": [int, int], "expected_rhs": [int, int], "expected_output": [int, int]},

        {"name": "Broadcast: Int vs Tuple", "lhs_shape": [int, 5, 2], "rhs_shape": [(1, 4), 2, 3], "expected_lhs": [int, 5, 2], "expected_rhs": [(1, 4), 2, 3], "expected_output": [int, 5, 3]},
        {"name": "Broadcast: Tuple vs Int", "lhs_shape": [(1, 6), 5, 2], "rhs_shape": [int, 2, 3], "expected_lhs": [(1, 6), 5, 2], "expected_rhs": [int, 2, 3], "expected_output": [int, 5, 3]},
        {"name": "Broadcast: Tuple vs Tuple", "lhs_shape": [(1, 7), 5, 2], "rhs_shape": [(1, 8), 2, 3], "expected_lhs": [(1, 7), 5, 2], "expected_rhs": [(1, 8), 2, 3], "expected_output": [(1, 7, 8), 5, 3]},
        {"name": "Broadcast: Unknown `int` vs Tuple", "lhs_shape": [int, 5, 2], "rhs_shape": [(8, 9), 2, 3], "expected_lhs": [int, 5, 2], "expected_rhs": [(8, 9), 2, 3], "expected_output": [int, 5, 3]},
        {"name": "Broadcast: 3-element Tuple vs 3-element Tuple", "lhs_shape": [(1, 2, 3), 5, 2], "rhs_shape": [(1, 3, 4), 2, 3], "expected_lhs": [(1, 2, 3), 5, 2], "expected_rhs": [(1, 3, 4), 2, 3], "expected_output": [(1, 2, 3, 4), 5, 3]},

        # === CATEGORY 2: MATRIX CORE DIMENSIONS ===
        {"name": "Matmul Core: Int vs Tuple", "lhs_shape": [5, 4], "rhs_shape": [(4, 6), 3], "expected_lhs": [5, 4], "expected_rhs": [4, 3], "expected_output": [5, 3]},
        {"name": "Matmul Core: Tuple vs Int", "lhs_shape": [5, (8, 9)], "rhs_shape": [8, 3], "expected_lhs": [5, 8], "expected_rhs": [8, 3], "expected_output": [5, 3]},
        {"name": "Matmul Core: Tuple vs Tuple", "lhs_shape": [5, (10, 11)], "rhs_shape": [(11, 12), 3], "expected_lhs": [5, 11], "expected_rhs": [11, 3], "expected_output": [5, 3]},
        {"name": "Matmul Core: Mixed Unknowns", "lhs_shape": [int, (13, 14)], "rhs_shape": [13, int], "expected_lhs": [int, 13], "expected_rhs": [13, int], "expected_output": [int, int]},
    
        # === NEW UNSATISFIABLE CASES ===
        {"name": "FAIL_01: Matmul Core Tuple Mismatch (No Overlap)", "lhs_shape": [10, (2, 3)], "rhs_shape": [(4, 5), 8], "expected_lhs": None, "expected_rhs": None, "expected_output": None},
        {"name": "FAIL_02: Broadcast Tuple vs Int Mismatch", "lhs_shape": [(2, 3), 5, 6], "rhs_shape": [4, 6, 7], "expected_lhs": None, "expected_rhs": None, "expected_output": None},
        {"name": "FAIL_05: Matrix-Vector Contraction Dim Mismatch", "lhs_shape": [5, 6, 7], "rhs_shape": [8], "rhs_rank": 1, "expected_lhs": None, "expected_rhs": None, "expected_output": None},
        {"name": "FAIL_06: Multi-Dimension Broadcast Tuple Mismatch (No Overlap)", "lhs_shape": [(2, 3), (4, 5), 10], "rhs_shape": [(6, 7), (8, 9), 10, 12], "expected_lhs": None, "expected_rhs": None, "expected_output": None},
        {"name": "FAIL_07: High Rank Promotion & Broadcast Mismatch", "lhs_shape": [int, 5, 6], "rhs_shape": [8, 7], "rhs_rank": 4, "expected_lhs": None, "expected_rhs": None, "expected_output": None},
    ]

    passed_count = 0
    for i, test in enumerate(test_cases):
        # print("-" * 60)
        # print(f"Running Test {i+1:02d}: {test['name']}")
        
        # Construct inputs for clarity
        inputs_str = f"Inputs: lhs={test['lhs_shape']}, rhs={test['rhs_shape']}"
        if "lhs_rank" in test: inputs_str += f", lhs_rank={test.get('lhs_rank')}"
        if "rhs_rank" in test: inputs_str += f", rhs_rank={test.get('rhs_rank')}"
        # print(inputs_str)

        try:
            inference = NumpySolver(
                lhs=test["lhs_shape"], 
                rhs=test["rhs_shape"],
                lhs_rank=test.get("lhs_rank"),
                rhs_rank=test.get("rhs_rank")
            )
            lhs, rhs, output = inference.solve_matmul()
            
            expected_lhs = test["expected_lhs"]
            expected_rhs = test["expected_rhs"]
            expected_output = test["expected_output"]

            # print(f"  -> Expected: lhs={expected_lhs}, rhs={expected_rhs}, output={expected_output}")
            # print(f"  -> Got:      lhs={lhs}, rhs={rhs}, output={output}")

            if lhs == expected_lhs and rhs == expected_rhs and output == expected_output:
                # print("Result: ✅ PASS")
                passed_count += 1
            else:
                print(inputs_str)
                print(f"  -> Expected: lhs={expected_lhs}, rhs={expected_rhs}, output={expected_output}")
                print(f"  -> Got:      lhs={lhs}, rhs={rhs}, output={output}")
                print(f"Running Test {i+1:02d}: {test['name']}")
                print(f"Result: 💥 ERROR - An exception occurred: {e}")
                print("Result: ❌ FAIL")

        except Exception as e:
            print(inputs_str)
            print(f"  -> Expected: lhs={expected_lhs}, rhs={expected_rhs}, output={expected_output}")
            print(f"  -> Got:      lhs={lhs}, rhs={rhs}, output={output}")
            print(f"Running Test {i+1:02d}: {test['name']}")
            print(f"Result: 💥 ERROR - An exception occurred: {e}")
            print(f"Result: 💥 ERROR - An exception occurred: {e}")
            traceback.print_exc()

    print("=" * 60)
    print(f"Test Suite Finished. Passed: {passed_count}/{len(test_cases)}")
    print("=" * 60)

def run_tests_broadcast():
    """
    Runs a suite of tests for the MatMulUnknown class.
    """
    test_cases = [
        # === BASIC SUCCESS (LHS/RHS MOSTLY KNOWN) ===
        {"name": "T02: Broadcasting (Both Known)", "lhs_shape": [5, 1], "rhs_shape": [7], "expected_lhs": [5, 1], "expected_rhs": [7], "expected_output": [5, 7]},
        {"name": "T04: Resolve RHS Tuple (Broadcast Dim)", "lhs_shape": [8], "rhs_shape": [(1, 8)], "expected_lhs": [8], "expected_rhs": [(1, 8)], "expected_output": [8]},
        
        # === ONE SIDE PARTIALLY/FULLY UNKNOWN ===
        {"name": "T09: Infer LHS from RHS Broadcast", "lhs_shape": [int, 1], "rhs_shape": [7], "expected_lhs": [int, 1], "expected_rhs": [7], "expected_output": [int, 7]},
        {"name": "T10: Infer RHS from LHS Broadcast", "lhs_shape": [5, 1], "rhs_shape": [int], "expected_lhs": [5, 1], "expected_rhs": [int], "expected_output": [5, int]},
        
        # === BOTH SIDES UNKNOWN ===
        {"name": "T13: Broadcasting With Both Unknown", "lhs_shape": [8, int], "rhs_shape": [int, 5], "expected_lhs": [8, (1, 5)], "expected_rhs": [(1, 8), 5], "expected_output": [8, 5]},
        {"name": "T14: High Rank Broadcast (Both Unknown)", "lhs_shape": [2, 3, int], "rhs_shape": [int, 3, 8], "expected_lhs": [2, 3, (1, 8)], "expected_rhs": [(1, 2), 3, 8], "expected_output": [2, 3, 8]},
        {"name": "T15: Unresolvable Tuple Resolves to 1", "lhs_shape": [(1, 6)], "rhs_shape": [9], "expected_lhs": [1], "expected_rhs": [9], "expected_output": [9]},

        # === FAILURE CASES (UNSAT) ===
        {"name": "T19: Broadcasting Mismatch", "lhs_shape": [5], "rhs_shape": [6], "expected_lhs": None, "expected_rhs": None, "expected_output": None},
        {"name": "T20: Tuple vs Int Broadcast Mismatch", "lhs_shape": [(2, 5)], "rhs_shape": [6], "expected_lhs": None, "expected_rhs": None, "expected_output": None},

        # === EDGE CASES ===
        {"name": "T24: Broadcasting with Same Dim and 1", "lhs_shape": [5, int], "rhs_shape": [5, 1], "expected_lhs": [5, int], "expected_rhs": [5, 1], "expected_output": [5, int]},

        {"name": "Broadcast: Int vs Tuple", "lhs_shape": [int], "rhs_shape": [(1, 4)], "expected_lhs": [int], "expected_rhs": [(1, 4)], "expected_output": [int]},
        {"name": "Broadcast: Tuple vs Int", "lhs_shape": [(1, 6)], "rhs_shape": [int], "expected_lhs": [(1, 6)], "expected_rhs": [int], "expected_output": [int]},
        {"name": "Broadcast: Tuple vs Tuple", "lhs_shape": [(1, 7)], "rhs_shape": [(1, 8)], "expected_lhs": [(1, 7)], "expected_rhs": [(1, 8)], "expected_output": [(1, 7, 8)]},
        {"name": "Broadcast: Unknown `int` vs Tuple", "lhs_shape": [int], "rhs_shape": [(8, 9)], "expected_lhs": [int], "expected_rhs": [(8, 9)], "expected_output": [int]},
        {"name": "Broadcast: 3-element Tuple vs 3-element Tuple", "lhs_shape": [(1, 2, 3)], "rhs_shape": [(1, 3, 4)], "expected_lhs": [(1, 2, 3)], "expected_rhs": [(1, 3, 4)], "expected_output": [(1, 2, 3, 4)]},

        # === NEW UNSATISFIABLE CASES ===
        {"name": "FAIL_02: Broadcast Tuple vs Int Mismatch", "lhs_shape": [(2, 3)], "rhs_shape": [4], "expected_lhs": None, "expected_rhs": None, "expected_output": None},
        {"name": "FAIL_06: Multi-Dimension Broadcast Tuple Mismatch (No Overlap)", "lhs_shape": [(2, 3)], "rhs_shape": [(6, 7), (8, 9)], "expected_lhs": None, "expected_rhs": None, "expected_output": None},
    ]

    passed_count = 0
    for i, test in enumerate(test_cases):
        # print("-" * 60)
        # print(f"Running Test {i+1:02d}: {test['name']}")
        
        # Construct inputs for clarity
        inputs_str = f"Inputs: lhs={test['lhs_shape']}, rhs={test['rhs_shape']}"
        if "lhs_rank" in test: inputs_str += f", lhs_rank={test.get('lhs_rank')}"
        if "rhs_rank" in test: inputs_str += f", rhs_rank={test.get('rhs_rank')}"
        # print(inputs_str)

        try:
            inference = NumpySolver(
                lhs=test["lhs_shape"], 
                rhs=test["rhs_shape"],
                lhs_rank=test.get("lhs_rank"),
                rhs_rank=test.get("rhs_rank")
            )
            lhs, rhs, output = inference.solve_broadcast()
            
            expected_lhs = test["expected_lhs"]
            expected_rhs = test["expected_rhs"]
            expected_output = test["expected_output"]

            # print(f"  -> Expected: lhs={expected_lhs}, rhs={expected_rhs}, output={expected_output}")
            # print(f"  -> Got:      lhs={lhs}, rhs={rhs}, output={output}")

            if lhs == expected_lhs and rhs == expected_rhs and output == expected_output:
                # print("Result: ✅ PASS")
                passed_count += 1
            else:
                print(inputs_str)
                print(f"  -> Expected: lhs={expected_lhs}, rhs={expected_rhs}, output={expected_output}")
                print(f"  -> Got:      lhs={lhs}, rhs={rhs}, output={output}")
                print(f"Running Test {i+1:02d}: {test['name']}")
                print("Result: ❌ FAIL")

        except Exception as e:
            print(inputs_str)
            print(f"  -> Expected: lhs={expected_lhs}, rhs={expected_rhs}, output={expected_output}")
            print(f"  -> Got:      lhs={lhs}, rhs={rhs}, output={output}")
            print(f"Running Test {i+1:02d}: {test['name']}")
            print(f"Result: 💥 ERROR - An exception occurred: {e}")
            traceback.print_exc()

    print("=" * 60)
    print(f"Test Suite Finished. Passed: {passed_count}/{len(test_cases)}")
    print("=" * 60)

if __name__ == '__main__':
    run_tests()
    run_tests_broadcast()