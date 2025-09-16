from z3 import *

def left_shapes_for_matmul(right_shape, max_left_rank=None):
    """
    Return a Z3 model describing one compatible left-hand shape L
    for the fixed right-hand shape R.

    `right_shape`  – tuple of positive ints, e.g. (4, 3, 2)
                     …R[-2] is the inner dimension k
    `max_left_rank` – if None, allow any rank ≥ 2;
                      otherwise fix the rank of L.
    """
    # ------------------------------------------------------------------
    # 1.  Decide the rank of the unknown left tensor
    # ------------------------------------------------------------------
    if max_left_rank is None:                       # let Z3 pick the rank
        rank_L = Int("rank_L")
        rank_constraints = [rank_L >= 2]            # at least 2 dims
    else:
        rank_L = max_left_rank
        rank_constraints = []

    # ------------------------------------------------------------------
    # 2.  Build symbolic dimensions for L = (l0, …, l_{p-2}, m, k)
    # ------------------------------------------------------------------
    def fresh(i): return Int(f"l{i}")
    dims_L   = [fresh(i) for i in range(0 if max_left_rank else 20)]  # room for growth
    dims_L   = dims_L if max_left_rank else dims_L[:rank_L]           # slice when rank decided

    # convenience names
    k_R = right_shape[-2]                       # inner dimension on RHS
    k_L = dims_L[-1]                            # inner dimension on LHS
    m_L = dims_L[-2]                            # the “output” dimension

    # ------------------------------------------------------------------
    # 3.  Assemble constraints
    # ------------------------------------------------------------------
    s = Solver()

    # positive integers
    s.add([d > 0 for d in dims_L] + [k_R > 0])

    # matmul inner dimension equality …, m, k  @  …, k, n
    s.add(k_L == k_R)

    # broadcast every batch dim (right-aligned, skipping the last 2)
    batch_L = dims_L[:-2]
    batch_R = right_shape[:-2]
    len_L, len_R = len(batch_L), len(batch_R)
    for i in range(1, min(len_L, len_R) + 1):
        dL, dR = batch_L[-i], batch_R[-i]
        s.add(Or(dL == dR, dL == 1, dR == 1))

    # any *extra* leading dims that exist only on L may be arbitrary ≥1
    # (already ensured by the positivity constraint)

    s.add(rank_constraints)

    # ------------------------------------------------------------------
    # 4.  Ask Z3 for *one* model
    # ------------------------------------------------------------------
    if s.check() != sat:
        raise ValueError("No compatible left-hand shapes exist.")

    model = s.model()

    # ------------------------------------------------------------------
    # 5.  Pretty-print:   (..., m, k)  – “…” marks unconstrained batch dims
    # ------------------------------------------------------------------
    solved = []
    for d in dims_L:
        if model[d].decl() in model:            # Z3 assigned a concrete number
            solved.append(model[d].as_long())
        else:                                   # freely-variable → keep symbolic
            solved.append("…")

    return tuple(solved)


# ------------------------------------------------------------
# DEMO
# ------------------------------------------------------------
if __name__ == "__main__":
    R = (4, 3, 2)                       # e.g. shape of right tensor
    L = left_shapes_for_matmul(R, max_left_rank=3)
    print(f"One compatible left shape for {R} is {L}")
    # ➜ One compatible left shape for (4, 3, 2) is ('…', 7, 3)
