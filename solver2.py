from z3 import *



class MatMulRHSKnown:
    def __init__(self, lhs, rhs):
        self.rhs = rhs
        self.lhs = lhs if lhs else []
        self.rhs_rank = len(rhs)


    # This takes in an input lhs_rank. This is so that if the LHS is longer than what we think it is.
    def solve(self, lhs_rank = None):
        if self.lhs:
            rank = lhs_rank if lhs_rank else len(self.lhs)
        else:
            rank = lhs_rank if lhs_rank else self.rhs_rank
        
        if self.rhs_rank == 1 and rank == 1:
            lhs, output = self.solve1x1()
        else:
            lhs, output = self.solve_ndim(rank)

    
        return lhs, output
    
    def solve1x1(self):
        s = Solver()

        # Init the solver
        lhs_vars = [Int(f"lhs_0")]
        s.add([d > 0 for d in lhs_vars])

        if self.lhs:
            for i, elem in enumerate(self.lhs):
                if isinstance(elem, int):
                    s.add(lhs_vars[i] == elem)
                elif (isinstance(elem, tuple)):
                    or_clauses = [lhs_vars[i] == val for val in elem]
                    s.add(Or(or_clauses))



        # Split the shape
        M, N = lhs_vars[-1], self.rhs[-1]

        s.add(M == N)

        if s.check() == sat:
            output = [int]
            lhs = self.summarize_nd_lhs(s, 1)
            return lhs, output

        return None, None

    def solve_ndim(self, lhs_rank):

        s = Solver()

        # Init the solver
        lhs_vars = [Int(f"lhs_{i}") for i in range(lhs_rank)]
        s.add([d > 0 for d in lhs_vars])

        if self.lhs and len(self.lhs) < lhs_rank:
            for _ in range(lhs_rank - len(self.lhs)):
                self.lhs.insert(0, int)

        if self.lhs:
            for i, elem in enumerate(self.lhs):
                if isinstance(elem, int):
                    s.add(lhs_vars[i] == elem)
                elif (isinstance(elem, tuple)):
                    or_clauses = [lhs_vars[i] == val for val in elem]
                    s.add(Or(or_clauses))



        # Split the shape
        lhs_broadcasting_vars = lhs_vars[:-2]
        M, K = lhs_vars[-2], lhs_vars[-1]

        rhs_broadcasting_vars = self.rhs[:-2]
        Kr, N = self.rhs[-2], self.rhs[-1]


        s.add(K == Kr)

        broadcasting_constraints, output = self._broadcasting(lhs_broadcasting_vars, rhs_broadcasting_vars)
        if self.lhs and self.lhs[-2]:
            output.append(self.lhs[-2])
        else: 
            output.append(int)
        output.append(N)
        s.add(broadcasting_constraints)

        


        if s.check() == sat:
            # print(s)
            # print(broadcasting_constraints)
            lhs = self.summarize_nd_lhs(s, lhs_rank)
            # print(f"output: {output}")
        else:
            lhs = None
            output = None
        
        return lhs, output
    

    def _broadcasting(self, lhs_broadcasting, rhs_broadcasting):
        constraints = []
        output = []

        lhs_dim = len(lhs_broadcasting)
        rhs_dim = len(rhs_broadcasting)

        for i in range(max(rhs_dim, lhs_dim)):

            lhs_idx = lhs_dim -1 -i
            rhs_idx = rhs_dim -1 -i

            if lhs_idx >= 0 and rhs_idx >= 0:
                lhs_d = lhs_broadcasting[lhs_idx]
                rhs_d = rhs_broadcasting[rhs_idx]
                if rhs_d == 1:
                    if self.lhs:
                        output.append(self.lhs[lhs_idx])
                    else:
                        output.append(int)
                else:
                    constraints.append(Or(lhs_d == 1, lhs_d == rhs_d))
                    output.append(rhs_d)
            elif lhs_idx >= 0:
                # This is the scenario where the LHS is bigger than the rhs, can be anything so we append anything
                if self.lhs:
                    output.append(self.lhs[lhs_idx])
                else:
                    output.append(int)
            elif rhs_idx >= 0:
                output.append(rhs_broadcasting[rhs_idx])
        

        # reverse the output 
        return constraints, output[::-1]

    def summarize_nd_lhs(self, s, lhs_rank):
        constraints = s.assertions()
        # print(constraints)

        # Lowkey from Chatgpt, just turns the constraints into usable output 
        lhs_output = [None for _ in range(lhs_rank)]
        for c in constraints:
            # Handle the special 'Or' case first
            if is_or(c):
                # The children of 'Or' are the inner equality expressions
                # e.g., [lhs_0 == 1, lhs_0 == 2]
                options = []
                # We get the index from the first child expression (e.g., lhs_0 == 1)
                # arg(0) is the variable (lhs_0)
                variable_node = c.children()[0].arg(0)
                index = int(str(variable_node).split('_')[1])
                
                # Loop through the inner expressions to get their values
                for option_expr in c.children():
                    # arg(1) is the value (e.g., 1, then 2)
                    # .as_long() converts the Z3 number to a Python int
                    value = option_expr.arg(1).as_long()
                    options.append(value)
                
                # Store all options as a tuple in the results 
                # make sure that the thing is not more specific
                if not isinstance(lhs_output[index], int):
                    lhs_output[index] = tuple(options)

            # Handle simple binary expressions like '>' or '=='
            else:
                # Get the variable (e.g., lhs_0) and its index
                variable_node = c.arg(0)
                index = int(str(variable_node).split('_')[1])
                
                # Get the operation name (e.g., '>', '==')
                operation = c.decl().name()
                
                # Get the value node (e.g., 0, 3)
                value_node = c.arg(1)

                # Apply your rules
                if operation == '=':
                    lhs_output[index] = value_node.as_long()
                elif operation == '>' and value_node.as_long() == 0:
                    # We only update if a more specific rule (like '==' or 'Or')
                    # hasn't already filled the spot.
                    if lhs_output[index] is None:
                        lhs_output[index] = int
        # print(f"LHS: {lhs_output}")
        return lhs_output

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

        if self.lhs and len(self.lhs) < self.lhs_rank:
            for _ in range(lhs_rank - len(self.lhs)):
                self.lhs.insert(0, int)

        if self.rhs and len(self.rhs) < self.rhs_rank:
            for _ in range(rhs_rank - len(self.rhs)):
                self.rhs.insert(0, int)

        if self.lhs_rank < len(self.lhs):
            print("LHS rank must be >= length of lhs")
            raise RuntimeError
        elif self.rhs_rank < len(self.rhs):
            print("RHS rank must be >= length of rhs")
            raise RuntimeError

        self.lhs_output = []
        self.rhs_output = []
        self.output = [int for _ in range(self.rank)]

        self.solver = Solver()
        self.output_solver = Solver()


        # while lhs_rank > len(self.lhs):
        #     self.lhs.insert(int, 0)
        # while rhs_rank > len(self.rhs):
        #     self.lhs.insert(int, 0)
    
    def solve(self):
        self.solve_matmul()
        lhs, rhs = None, None
        if self.solver.check() == sat:
            lhs, rhs = self.summarize_nd_sides()
        return lhs, rhs, self.output

    def solve_matmul(self):
        lhs_vars = [Int(f"lhs_{i}") for i in range(self.lhs_rank)]
        rhs_vars = [Int(f"rhs_{i}") for i in range(self.rhs_rank)]

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


            if i <= self.lhs_rank and i <= self.rhs_rank:
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

                    self.output_solver.add(Or(lhs_or_clauses))
                    self.output_solver.add(Or(rhs_or_clauses))

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
        lhs_output = [None] * self.lhs_rank
        rhs_output = [None] * self.rhs_rank
        
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
            



if __name__ == '__main__':
    x_shape = [3, (1,4)]
    y_shape = [4, 5]
    inference = MatMulRHSKnown(x_shape, y_shape)
    lhs, output = inference.solve()
    print(f"{x_shape} @ {y_shape} = {output} and lhs (should be y_shape): {lhs}")


# print("--- Test Case 1: Standard Case (Same Rank) ---")
# print("RHS: (2, 3, 4)")
# inference1 = MatMulRHSKnown(None, (2, 3, 4))
# inference1.solve()
# # Expected LHS: [2, <class 'int'>, 3]
# # Expected Output: [2, <class 'int'>, 4]


# print("--- Test Case 2: Standard Case (RHS Rank > LHS Rank) ---")
# print("RHS: (10, 8, 5, 6), solving for a compatible LHS of Rank 2")
# inference2 = MatMulRHSKnown(None, (10, 8, 5, 6))
# # We explicitly ask to find a compatible LHS of rank 2
# lhs1, out1 = inference2.solve(5)
# Expected LHS: [<class 'int'>, 5]
# Expected Output: [10, 8, <class 'int'>, 6]

# inference3 = MatMulRHSKnown(out1, (10, 8, 6, 10))
# inference3.solve()


# print("--- Test Case 3: Standard Case (LHS Rank > RHS Rank) ---")
# print("RHS: (5, 6), solving for a compatible LHS of Rank 4")
# inference3 = MatMulRHSKnown(None, (5, 6))
# inference3.solve(lhs_rank=4)
# # Expected LHS: [<class 'int'>, <class 'int'>, <class 'int'>, 5]
# # Expected Output: [<class 'int'>, <class 'int'>, <class 'int'>, 6]


# print("--- Test Case 4: Broadcasting Case ---")
# print("RHS: (1, 8, 5, 6)")
# inference4 = MatMulRHSKnown(None, (1, 8, 5, 6))
# inference4.solve()
# # The first broadcasting dim of LHS must be 1. The second can be 1 or 8.
# # The solver will find a valid instance.
# # Expected LHS: [1, 8, <class 'int'>, 5] (or a variation where dim 1 is 1)
# # Expected Output: [1, 8, <class 'int'>, 6]


# print("--- Test Case 5: Edge Case (2D Matrix) ---")
# print("RHS: (5, 6)")
# inference5 = MatMulRHSKnown(None, (5, 6))
# inference5.solve()
# # Expected LHS: [<class 'int'>, 5]
# # Expected Output: [<class 'int'>, 6]


# # 1
# x1 = (5, 3)
# y1 = (2, 3, 5, 4)

# # 2
# x2 = (4, 3, 2)
# y2 = (5, 2, 6)

# # 3
# x3 = (2, 3, 4, 5)
# y3 = (4, 1, 5, 6)

# # 4
# x4 = (2, 2, 2, 3, 4)
# y4 = (2, 3, 5)

# # 5
# x5 = (4, 6, 3)
# y5 = (2, 5)

# # 6
# x6 = (2, 1, 4, 2, 3)
# y6 = (3, 2, 1, 3, 6)

# # 7
# x7 = (3, 4)
# y7 = (4, 3, 2)

# # 8
# x8 = (3, 4)
# y8 = (3, 3)

# # 11
# x11 = (3, 4)
# y11 = (4, 2)

# # 12
# x12 = (5, 3, 4)
# y12 = (5, 4, 2)

# # 13
# x13 = (1, 2, 3, 6)
# y13 = (5, 2, 6, 4)

# # 14
# x14 = (4, 6)
# y14 = (1, 6, 5)

# # 15
# x15 = (3, 6, 7)
# y15 = (7, 2)

# # 16
# x16 = (2, 1, 4, 2, 3)
# y16 = (2, 3, 1, 3, 6)

# # 17
# x17 = (3, 1, 2, 8)
# y17 = (1, 8, 4)

# # 18
# x18 = (2, 5, 3)
# y18 = (1, 2, 3, 6)

# # 19
# x19 = (1, 1, 3, 4, 7)
# y19 = (7, 5)

# # 20
# x20 = (1, 9, 6)
# y20 = (2, 3, 1, 6, 10)

# test_cases = [
    # # failing
    # (x1, y1), (x2, y2), (x3, y3), (x4, y4),
    # (x5, y5), (x6, y6), (x7, y7), (x8, y8),

    # # passing
    # (x11, y11), (x12, y12), (x13, y13), (x14, y14),
    # (x15, y15), (x16, y16), (x17, y17), (x18, y18),
    # (x19, y19), (x20, y20),
# ]

# for x_shape, y_shape in test_cases:
    # inference = MatMulRHSKnown(x_shape, y_shape)
    # lhs, output = inference.solve()
    # print(f"{x_shape} @ {y_shape} = {output} and lhs (should be y_shape): {lhs}")

# inference = MatMulRHSKnown((int, 4,6), (1,6,5))
# lhs, output = inference.solve()
# print(f"{(int, 4, 6)} @ {(1, 6, 5)} = {output} and lhs (should be y_shape): {lhs}")