from z3 import *

class MatMulUnknown:
    def __init__(self, lhs, rhs, lhs_rank=None, rhs_rank=None):
        self.lhs = lhs if lhs else []
        self.rhs = rhs if rhs else []

        if (not lhs) and (not lhs_rank) and (not rhs) and (not rhs_rank):
            print("Need at least one of lhs, lhs_rank, rhs, rhs_rank")
            raise RuntimeError         

        self.lhs_rank = lhs_rank if lhs_rank else len(lhs)
        self.rhs_rank = rhs_rank if rhs_rank else len(rhs)
        self.rank = max(self.lhs_rank, self.rhs_rank)

        if self.lhs and len(self.lhs) < self.rank:
            for _ in range(self.rank - len(self.lhs)):
                self.lhs.insert(0, int)

        if self.rhs and len(self.rhs) < self.rank:
            for _ in range(self.rank - len(self.rhs)):
                self.rhs.insert(0, int)

        # if self.lhs_rank < len(self.lhs):
        #     print("LHS rank must be >= length of lhs")
        #     raise RuntimeError
        # elif self.rhs_rank < len(self.rhs):
        #     print("RHS rank must be >= length of rhs")
        #     raise RuntimeError

        self.output = [int for _ in range(self.rank)]

        self.solver = Solver()

        # while lhs_rank > len(self.lhs):
        #     self.lhs.insert(int, 0)
        # while rhs_rank > len(self.rhs):
        #     self.lhs.insert(int, 0)
    
    def solve(self):
        self.solve_matmul()
        lhs, rhs, output = None, None, None
        if self.solver.check() == sat:
            lhs, rhs = self.summarize_nd_sides()
            output = self.output
        return lhs, rhs, output

    def solve_matmul(self):
        lhs_vars = [Int(f"lhs_{i}") for i in range(self.rank)]
        rhs_vars = [Int(f"rhs_{i}") for i in range(self.rank)]

        if isinstance(self.lhs[-2], int):
            self.solver.add(lhs_vars[-2] == self.lhs[-2])
        elif (isinstance(self.lhs[-2], tuple)):
            or_clauses = [lhs_vars[-2] == val for val in self.lhs[-2]]
            self.solver.add(Or(or_clauses))
        if isinstance(self.lhs[-1], int):
            self.solver.add(lhs_vars[-1] == self.lhs[-1])
        elif (isinstance(self.lhs[-1], tuple)):
            or_clauses = [lhs_vars[-1] == val for val in self.lhs[-1]]
            self.solver.add(Or(or_clauses))

        if isinstance(self.rhs[-2], int):
            self.solver.add(rhs_vars[-2] == self.rhs[-2])
        elif (isinstance(self.rhs[-2], tuple)):
            or_clauses = [rhs_vars[-2] == val for val in self.rhs[-2]]
            self.solver.add(Or(or_clauses))
        if isinstance(self.rhs[-1], int):
            self.solver.add(rhs_vars[-1] == self.rhs[-1])
        elif (isinstance(self.rhs[-1], tuple)):
            or_clauses = [rhs_vars[-1] == val for val in self.rhs[-1]]
            self.solver.add(Or(or_clauses))
        

        if not ((self.lhs[-1] is int) and (self.rhs[-2] is int)):
            self.solver.add(lhs_vars[-1] == rhs_vars[-2])
        

        self.output[-2] = self.lhs[-2]
        self.output[-1] = self.rhs[-1]

        lhs_broadcasting = self.lhs[:-2]
        rhs_broadcasting = self.rhs[:-2]
        lhs_broadcasting_vars = lhs_vars[:-2]
        rhs_broadcasting_vars = rhs_vars[:-2]


        lhs_dim = len(lhs_broadcasting)
        rhs_dim = len(rhs_broadcasting)

        i = 3

        while i <= self.rank:
            idx = -i

            lhs_d = self.lhs[idx]
            rhs_d = self.rhs[idx]
            lhs_var = lhs_vars[idx]
            rhs_var = rhs_vars[idx]

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
                possible_outputs.add(val for val in lhs_d)
                possible_outputs.add(val for val in rhs_d)
                possible_outputs = tuple(list(possible_outputs).sort())
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
        Summarizes Z3 constraints for lhs and rhs variables.

        This function handles three types of constraints:
        1. Direct equality to a constant (e.g., lhs_0 == 5).
        2. 'Or' clauses for multiple possible values (e.g., Or(lhs_1 == 1, lhs_1 == 4)).
        3. Equality between two variables (e.g., lhs_1 == rhs_0).
        """
        constraints = self.solver.assertions()

        # Initialize output lists for both sides
        lhs_output = [None] * self.rank
        rhs_output = [None] * self.rank
        
        # A list to store relationships between variables (e.g., lhs_1 == rhs_0)
        # We process these after finding all direct values.
        aliases = []

        # --- PASS 1: Handle direct value assignments and 'Or' clauses ---
        for c in constraints:
            # Handle 'Or' clauses, e.g., Or(lhs_1 == 1, lhs_1 == 4)
            if is_or(c):
                # The first child gives us the variable name and index
                variable_node = c.children()[0].arg(0)
                side, index = self._parse_var_node(variable_node)
                
                # Collect all possible integer values from the 'Or'
                options = [opt.arg(1).as_long() for opt in c.children()]
                
                # Assign to the correct output list
                if side == 'lhs':
                    # Only update if a more specific rule (like '==') hasn't been set
                    if not isinstance(lhs_output[index], int):
                        lhs_output[index] = tuple(options)
                else: # side == 'rhs'
                    if not isinstance(rhs_output[index], int):
                        rhs_output[index] = tuple(options)
            
            # Handle equality constraints, e.g., lhs_0 == 5 or lhs_1 == rhs_0
            elif is_eq(c):
                arg0 = c.arg(0)
                arg1 = c.arg(1)

                # Case 1: Equality to a constant value (e.g., lhs_0 == 5)
                if is_int_value(arg1):
                    side, index = self._parse_var_node(arg0)
                    value = arg1.as_long()
                    
                    if side == 'lhs':
                        lhs_output[index] = value
                    else: # side == 'rhs'
                        rhs_output[index] = value
                
                # Case 2: Equality between two variables (e.g., lhs_1 == rhs_0)
                else:
                    # Parse both sides and store the relationship to solve later
                    side1, index1 = self._parse_var_node(arg0)
                    side2, index2 = self._parse_var_node(arg1)
                    aliases.append(((side1, index1), (side2, index2)))

        # --- PASS 2: Propagate values using the recorded aliases ---
        # This loop continues until no more values can be propagated, handling chains
        # like a = b, b = c, c = 5.
        value_propagated = True
        while value_propagated:
            value_propagated = False
            for (side1, index1), (side2, index2) in aliases:
                # Get the current values from the output lists
                val1 = lhs_output[index1] if side1 == 'lhs' else rhs_output[index1]
                val2 = lhs_output[index2] if side2 == 'lhs' else rhs_output[index2]

                # Propagate from var2 to var1 if var1 is unresolved
                if val2 is not None and val1 is None:
                    if side1 == 'lhs':
                        lhs_output[index1] = val2
                    else:
                        rhs_output[index1] = val2
                    value_propagated = True

                # Propagate from var1 to var2 if var2 is unresolved
                if val1 is not None and val2 is None:
                    if side2 == 'lhs':
                        lhs_output[index2] = val1
                    else:
                        rhs_output[index2] = val1
                    value_propagated = True
                
                # Overwrite tuple from 'Or' with a more specific integer value
                if isinstance(val1, tuple) and isinstance(val2, int):
                    if side1 == 'lhs': lhs_output[index1] = val2
                    else: rhs_output[index1] = val2
                    value_propagated = True
                
                if isinstance(val2, tuple) and isinstance(val1, int):
                    if side2 == 'lhs': lhs_output[index2] = val1
                    else: rhs_output[index2] = val1
                    value_propagated = True


        return lhs_output, rhs_output

    def change_output(self, value, location):
        """
        Updates self.output based on a priority system.

        Priority Rules:
        1. int (highest): Cannot be overwritten.
        2. tuple: Can be replaced by an int, but not another tuple.
        3. other (lowest): Can be replaced by anything.
        """
        current_val = self.output[location]

        # If current value is an int, do not update.
        if isinstance(current_val, int):
            return

        # If current is a tuple, only update if the new value is NOT a tuple.
        if isinstance(current_val, tuple) and isinstance(value, tuple):
            return
            
        # If the checks above pass, perform the update.
        self.output[location] = value
            

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
        {"name": "T05: Matrix-Vector Multiplication", "lhs_shape": [20, 10], "rhs_shape": [10], "rhs_rank": 2, "expected_lhs": [20, 10], "expected_rhs": [10, int], "expected_output": [20, int]},
        {"name": "T06: Vector-Matrix Multiplication", "lhs_shape": [10], "rhs_shape": [10, 20], "lhs_rank": 2, "expected_lhs": [int, 10], "expected_rhs": [10, 20], "expected_output": [int, 20]},

        # === ONE SIDE PARTIALLY/FULLY UNKNOWN ===
        {"name": "T07: LHS Fully Unknown (Rank Given)", "lhs_shape": [], "lhs_rank": 2, "rhs_shape": [5, 6], "expected_lhs": [int, 5], "expected_rhs": [5, 6], "expected_output": [int, 6]},
        {"name": "T08: RHS Fully Unknown (Rank Given)", "lhs_shape": [7, 8], "rhs_shape": [], "rhs_rank": 2, "expected_lhs": [7, 8], "expected_rhs": [8, int], "expected_output": [7, int]},
        {"name": "T09: Infer LHS from RHS Broadcast", "lhs_shape": [int, 1, 3, 4], "rhs_shape": [7, 4, 8], "expected_lhs": [(1, 7), 1, 3, 4], "expected_rhs": [int, 7, 4, 8], "expected_output": [7, 7, 3, 8]},
        {"name": "T10: Infer RHS from LHS Broadcast", "lhs_shape": [5, 1, 3, 4], "rhs_shape": [int, 4, 8], "expected_lhs": [5, 1, 3, 4], "expected_rhs": [(1, 5), int, 4, 8], "expected_output": [5, int, 3, 8]},
        {"name": "T11: Rank Promotion on Unknown LHS", "lhs_shape": [], "lhs_rank": 4, "rhs_shape": [10, 20, 30], "expected_lhs": [int, (1, 10), int, 20], "expected_rhs": [int, 10, 20, 30], "expected_output": [int, 10, int, 30]},
        
        # === BOTH SIDES UNKNOWN ===
        {"name": "T12: Both Sides Partially Unknown", "lhs_shape": [10, int], "rhs_shape": [int, 12], "expected_lhs": [10, int], "expected_rhs": [int, 12], "expected_output": [10, 12]},
        {"name": "T13: Broadcasting With Both Unknown", "lhs_shape": [8, int, 6, 7], "rhs_shape": [int, 5, 1, 9], "expected_lhs": [8, (1, 5), 6, 7], "expected_rhs": [(1, 8), 5, 1, 9], "expected_output": [8, 5, 6, 9]},
        {"name": "T14: High Rank Broadcast (Both Unknown)", "lhs_shape": [2, 3, int, 5, 6], "rhs_shape": [int, 3, 8, 6, 7], "expected_lhs": [2, 3, (1, 8), 5, 6], "expected_rhs": [(1, 2), 3, 8, 6, 7], "expected_output": [2, 3, 8, 5, 7]},
        {"name": "T15: Unresolvable Tuple Resolves to 1", "lhs_shape": [(1, 6), 7, 8], "rhs_shape": [9, 8, 9], "expected_lhs": [1, 7, 8], "expected_rhs": [9, 8, 9], "expected_output": [9, 7, 9]},
        {"name": "T16: Both Ranks Promoted & Unknown", "lhs_shape": [5, 6], "lhs_rank": 4, "rhs_shape": [int, 6, 7], "rhs_rank": 4, "expected_lhs": [int, int, 5, 6], "expected_rhs": [int, (1, 5), 6, 7], "expected_output": [int, 5, 5, 7]},
        {"name": "T17: Only Ranks Provided", "lhs_shape": [], "lhs_rank": 3, "rhs_shape": [], "rhs_rank": 3, "expected_lhs": [int, int, int], "expected_rhs": [int, int, int], "expected_output": [int, int, int]},

        # === FAILURE CASES (UNSAT) ===
        {"name": "T18: Simple Matmul Mismatch", "lhs_shape": [2, 3], "rhs_shape": [4, 5], "expected_lhs": None, "expected_rhs": None, "expected_output": None},
        {"name": "T19: Broadcasting Mismatch", "lhs_shape": [5, 3, 2], "rhs_shape": [6, 2, 4], "expected_lhs": None, "expected_rhs": None, "expected_output": None},
        {"name": "T20: Tuple vs Int Broadcast Mismatch", "lhs_shape": [(2, 5), 3, 4], "rhs_shape": [6, 4, 8], "expected_lhs": None, "expected_rhs": None, "expected_output": None},
        {"name": "T21: Tuple vs Tuple Matmul Mismatch", "lhs_shape": [10, (1, 5)], "rhs_shape": [(2, 6), 20], "expected_lhs": None, "expected_rhs": None, "expected_output": None},
        {"name": "T22: Mismatch after Rank Promotion", "lhs_shape": [4, 5], "lhs_rank": 3, "rhs_shape": [3, 5, 6], "expected_lhs": None, "expected_rhs": None, "expected_output": None},

        # === EDGE CASES ===
        {"name": "T23: Scalar-like Output", "lhs_shape": [1, 10], "rhs_shape": [10, 1], "expected_lhs": [1, 10], "expected_rhs": [10, 1], "expected_output": [1, 1]},
        {"name": "T24: Broadcasting with Same Dim and 1", "lhs_shape": [5, int, 3, 4], "rhs_shape": [5, 1, 4, 8], "expected_lhs": [5, 1, 3, 4], "expected_rhs": [5, 1, 4, 8], "expected_output": [5, 1, 3, 8]},
        {"name": "T25: Full Pass-through Unknowns", "lhs_shape": [int, int], "rhs_shape": [int, int], "expected_lhs": [int, int], "expected_rhs": [int, int], "expected_output": [int, int]},
    ]

    passed_count = 0
    for i, test in enumerate(test_cases):
        print("-" * 60)
        print(f"Running Test {i+1:02d}: {test['name']}")
        
        # Construct inputs for clarity
        inputs_str = f"Inputs: lhs={test['lhs_shape']}, rhs={test['rhs_shape']}"
        if "lhs_rank" in test: inputs_str += f", lhs_rank={test.get('lhs_rank')}"
        if "rhs_rank" in test: inputs_str += f", rhs_rank={test.get('rhs_rank')}"
        print(inputs_str)

        try:
            inference = MatMulUnknown(
                lhs=test["lhs_shape"], 
                rhs=test["rhs_shape"],
                lhs_rank=test.get("lhs_rank"),
                rhs_rank=test.get("rhs_rank")
            )
            lhs, rhs, output = inference.solve()
            
            # Helper to convert None to `int` for consistent comparison
            def normalize(shape):
                if shape is None: return None
                return [d if d is not None else int for d in shape]

            lhs = normalize(lhs)
            rhs = normalize(rhs)
            output = normalize(output)
            
            expected_lhs = test["expected_lhs"]
            expected_rhs = test["expected_rhs"]
            expected_output = test["expected_output"]

            print(f"  -> Expected: lhs={expected_lhs}, rhs={expected_rhs}, output={expected_output}")
            print(f"  -> Got:      lhs={lhs}, rhs={rhs}, output={output}")

            if lhs == expected_lhs and rhs == expected_rhs and output == expected_output:
                print("Result: ✅ PASS")
                passed_count += 1
            else:
                print("Result: ❌ FAIL")

        except Exception as e:
            print(f"Result: 💥 ERROR - An exception occurred: {e}")
            traceback.print_exc()

    print("=" * 60)
    print(f"Test Suite Finished. Passed: {passed_count}/{len(test_cases)}")
    print("=" * 60)

if __name__ == '__main__':
    run_tests()