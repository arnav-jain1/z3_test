import traceback
from solver2 import MatMulRHSKnown, MatMulUnknown

# The 'int' type is used in test case definitions
# --- TEST SUITE ---
def run_tests():
    test_cases = [
        # --- Basic Success ---
        {"name": "T01: Simple Matmul", "x_shape": [2, 3], "y_shape": [3, 4], "expected_lhs": [2, 3], "expected_output": [2, 4]},
        {"name": "T02: Resolve from Tuple", "x_shape": [5, (6, 7)], "y_shape": [7, 8], "expected_lhs": [5, 7], "expected_output": [5, 8]},
        {"name": "T03: Resolve from int", "x_shape": [5, int], "y_shape": [6, 7], "expected_lhs": [5, 6], "expected_output": [5, 7]},
        
        # --- Broadcasting ---
        {"name": "T04: Broadcasting Intersection", "x_shape": [(2, 8), 4, 5], "y_shape": [2, 5, 6], "expected_lhs": [2, 4, 5], "expected_output": [2, 4, 6]},
        {"name": "T05: Broadcasting with 1", "x_shape": [5, 1, 3, 4], "y_shape": [7, 4, 8], "expected_lhs": [5, 1, 3, 4], "expected_output": [5, 7, 3, 8]},
        {"name": "T06: Broadcasting Unconstrained Tuple", "x_shape": [(1, 5), 1, 3, 4], "y_shape": [5, 7, 4, 8], "expected_lhs": [(1, 5), 1, 3, 4], "expected_output": [5, 7, 3, 8]},

        # --- Rank Promotion ---
        {"name": "T07: Rank Promotion Basic", "x_shape": [4, 5], "y_shape": [5, 6], "lhs_rank": 3, "expected_lhs": [int, 4, 5], "expected_output": [int, 4, 6]},
        {"name": "T08: Rank Promotion with Tuple", "x_shape": [(4, 8), 5], "y_shape": [5, 6], "lhs_rank": 3, "expected_lhs": [int, (4, 8), 5], "expected_output": [int, (4, 8), 6]},
        {"name": "T09: Full Unknown LHS", "x_shape": [], "y_shape": [4, 5], "lhs_rank": 2, "expected_lhs": [int, 4], "expected_output": [int, 5]},
        
        # --- Complex Success ---
        {"name": "T10: High Rank Broadcast", "x_shape": [2, 3, 1, 5, 6], "y_shape": [3, 8, 6, 7], "expected_lhs": [2, 3, 1, 5, 6], "expected_output": [2, 3, 8, 5, 7]},
        
        # --- Failure Cases (unsat) ---
        {"name": "T11: Simple Mismatch", "x_shape": [2, 3], "y_shape": [4, 5], "expected_lhs": None, "expected_output": None},
        {"name": "T12: Broadcasting Mismatch", "x_shape": [5, 3, 2], "y_shape": [6, 2, 4], "expected_lhs": None, "expected_output": None},
        {"name": "T13: Complex Broadcasting Mismatch", "x_shape": [(2, 5), 1, 3, 4], "y_shape": [6, 7, 4, 8], "expected_lhs": None, "expected_output": None},
        {"name": "T14: High Rank Broadcasting Mismatch", "x_shape": [2, 3, 4, 5, 6], "y_shape": [2, 3, 8, 6, 7], "expected_lhs": None, "expected_output": None},

        # --- Edge Cases ---
        {"name": "T15: Vector Dot Product", "x_shape": [10], "y_shape": [10], "expected_lhs": [10], "expected_output": [int]},
    ]

    passed_count = 0
    for i, test in enumerate(test_cases):
        print("-" * 50)
        print(f"Running Test {i+1:02d}: {test['name']}")
        print(f"Inputs: lhs={test['x_shape']}, rhs={test['y_shape']}", end="")
        
        lhs_rank = test.get("lhs_rank")
        if lhs_rank:
            print(f", lhs_rank={lhs_rank}")
        else:
            print()

        try:
            inference = MatMulRHSKnown(test["x_shape"], test["y_shape"])
            lhs, output = inference.solve(lhs_rank=lhs_rank)
            
            if lhs is not None and not isinstance(lhs, list): lhs = list(lhs)
            if output is not None and not isinstance(output, list): output = list(output)
            
            expected_lhs = test["expected_lhs"]
            expected_output = test["expected_output"]

            print(f"  -> Expected: lhs={expected_lhs}, output={expected_output}")
            print(f"  -> Got:      lhs={lhs}, output={output}")

            if lhs == expected_lhs and output == expected_output:
                print("Result: ✅ PASS")
                passed_count += 1
            else:
                print("Result: ❌ FAIL")

        except Exception as e:
            print(f"Result: 💥 ERROR - An exception occurred: {e}")
            traceback.print_exc()

    print("=" * 50)
    print(f"Test Suite Finished. Passed: {passed_count}/{len(test_cases)}")
    print("=" * 50)

def test_both_unknown():
    # --- Create and run the test case ---
    print("--- Initializing Test Case ---")
    # LHS: [(2, 3), 1, 5, 4]
    # RHS: [4, 6] with rank 3, so it becomes [<class 'int'>, 4, 6]
    lhs_shape = [(1,3), 5, (1,4)]
    rhs_shape = [1, 4, 6]
    test_case = MatMulUnknown(lhs=lhs_shape, rhs=rhs_shape)

    print(f"LHS (padded): {test_case.lhs}")
    print(f"RHS (padded): {test_case.rhs}")
    print(f"Max Rank: {test_case.rank}\n")

    # Run the solver logic
    lhs, rhs = test_case.solve()

    # --- Print the useful information ---
    print("--- Solver State After Execution ---")

    print("\n## SOLVER CONSTRAINTS ##")
    for constraint in test_case.solver.assertions():
        print(constraint)
    # print("\n## LHS SOLVER CONSTRAINTS ##")
    # for constraint in test_case.lhs_solver.assertions():
    #     print(constraint)

    # print("\n## RHS SOLVER CONSTRAINTS ##")
    # for constraint in test_case.rhs_solver.assertions():
    #     print(constraint)

    # print("\n## OUTPUT SOLVER CONSTRAINTS ##")
    # for constraint in test_case.output_solver.assertions():
    #     print(constraint)

    print(f"\nPartially solved output shape: {test_case.output}")
    print(f"Estimated lhs: {lhs}")
    print(f"Estimated rhs: {rhs}")

if __name__ == "__main__":
    test_both_unknown()