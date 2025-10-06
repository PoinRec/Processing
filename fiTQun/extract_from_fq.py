import sys, math, argparse, os
import numpy as np
import h5py
import ROOT

sys.path.append('/home/zhihao/Processing/Predictions')
import infer_exit_pos

def get_leaf(tree, *candidates):
    """Return the first found TLeaf among candidates, or None."""
    for name in candidates:
        leaf = tree.GetLeaf(name)
        if leaf:
            return leaf
    return None

def main():
    ap = argparse.ArgumentParser(
        description="Pick muon (hypo=2) results only across seeds (choose smallest muon NLL); also store (nll_mu - nll_e). Save to HDF5 (PyROOT)."
    )
    ap.add_argument("input", help="Input ROOT file")
    ap.add_argument("output", nargs="?", help="Output HDF5 (.h5/.hdf5). If omitted, auto-generate next to input.")
    ap.add_argument("treename", nargs="?", default="fiTQun", help="Tree name (default: fiTQun)")
    ap.add_argument("--no-require-rpc", action="store_true", help="Do not require fq1rpcflg > 0")
    args = ap.parse_args()

    # Auto output path
    if args.output is None:
        in_dir, in_name = os.path.split(args.input)
        base, _ = os.path.splitext(in_name)
        args.output = os.path.join(in_dir, f"{base}_muonly_minNLL.h5")

    # Open ROOT
    f = ROOT.TFile.Open(args.input)
    if not f or f.IsZombie():
        print("Cannot open file:", args.input); sys.exit(1)
    t = f.Get(args.treename)
    if not t:
        print("Tree not found:", args.treename); sys.exit(1)

    # Leaves
    lf_fqnse = get_leaf(t, "fqnse")
    lf_nll   = get_leaf(t, "fq1rnll", "fq1nll")
    lf_mom   = get_leaf(t, "fq1rmom")
    lf_pos   = get_leaf(t, "fq1rpos")
    lf_dir   = get_leaf(t, "fq1rdir")
    lf_rpc   = get_leaf(t, "fq1rpcflg")
    lf_nevt  = get_leaf(t, "nevt")  # optional

    if not (lf_fqnse and lf_nll and lf_mom and lf_pos and lf_dir):
        print("Missing required leaves: need fqnse, fq1rnll/nll, fq1rmom, fq1rpos, fq1rdir"); sys.exit(1)

    require_rpc = not args.no_require_rpc

    # Output containers
    nevt_list, valid_list, seed_list, hypo_list = [], [], [], []
    nll_list, mom_list, pos_list, dir_list = [], [], [], []
    dnll_em_list = []  # nll_mu - nll_e at the chosen seed

    N = t.GetEntries()
    for i in range(N):
        t.GetEntry(i)

        nevt = int(lf_nevt.GetValue()) if lf_nevt else (i+1)
        nevt_list.append(nevt)

        nse = int(lf_fqnse.GetValue())
        if nse <= 0:
            valid_list.append(False)
            seed_list.append(-1); hypo_list.append(-1)
            nll_list.append(np.nan); mom_list.append(np.nan)
            pos_list.append(np.full(3, np.nan)); dir_list.append(np.full(3, np.nan))
            dnll_em_list.append(np.nan)
            continue

        H_E, H_MU = 1, 2

        # Among MUON only, pick the smallest NLL across seeds
        best_mu_val = float("inf")
        best_seed = -1

        for s in range(nse):
            flat_mu = s * 7 + H_MU
            nll_mu = lf_nll.GetValue(flat_mu)

            if (not math.isfinite(nll_mu)) or (nll_mu <= 0):
                continue
            if require_rpc and lf_rpc:
                rpc_mu = lf_rpc.GetValue(flat_mu)
                if rpc_mu <= 0:
                    continue

            if nll_mu < best_mu_val:
                best_mu_val = nll_mu
                best_seed = s

        if best_seed < 0:
            # no valid muon across seeds
            valid_list.append(False)
            seed_list.append(-1); hypo_list.append(-1)
            nll_list.append(np.nan); mom_list.append(np.nan)
            pos_list.append(np.full(3, np.nan)); dir_list.append(np.full(3, np.nan))
            dnll_em_list.append(np.nan)
            continue

        # Extract MUON values at chosen seed
        flat_mu = best_seed * 7 + H_MU
        mom  = lf_mom.GetValue(flat_mu)
        posx, posy, posz = [lf_pos.GetValue(flat_mu*3 + j) for j in range(3)]
        dirx, diry, dirz = [lf_dir.GetValue(flat_mu*3 + j) for j in range(3)]
        nll_mu = lf_nll.GetValue(flat_mu)

        # Compute nll_mu - nll_e at the SAME seed (electron might be invalid)
        flat_e = best_seed * 7 + H_E
        nll_e = lf_nll.GetValue(flat_e)
        # Valid only if finite & >0; otherwise NaN
        dnll = (nll_mu - nll_e) if (math.isfinite(nll_e) and nll_e > 0) else np.nan

        valid_list.append(True)
        seed_list.append(best_seed)
        hypo_list.append(H_MU)    # always 2 (muon)
        nll_list.append(nll_mu)
        mom_list.append(mom)
        pos_list.append(np.array([posx, posy, posz], dtype=float))
        dir_list.append(np.array([dirx, diry, dirz], dtype=float))
        dnll_em_list.append(dnll)

        if (i+1) % 10000 == 0:
            print(f"[{i+1}/{N}] processed")

    # To numpy
    nevt_arr  = np.asarray(nevt_list,  dtype=np.int64)
    valid_arr = np.asarray(valid_list, dtype=np.bool_)
    seed_arr  = np.asarray(seed_list,  dtype=np.int32)
    hypo_arr  = np.asarray(hypo_list,  dtype=np.int32)
    nll_arr   = np.asarray(nll_list,   dtype=np.float64)
    mom_arr   = np.asarray(mom_list,   dtype=np.float64)
    pos_arr   = np.vstack(pos_list).astype(np.float64) if pos_list else np.empty((0,3))
    dir_arr   = np.vstack(dir_list).astype(np.float64) if dir_list else np.empty((0,3))
    dnll_arr  = np.asarray(dnll_em_list, dtype=np.float64)

    # Save HDF5
    with h5py.File(args.output, "w") as h:
        h.attrs["source"] = args.input
        h.attrs["tree"]   = args.treename
        h.attrs["note"]   = "Muon-only selection: among seeds, pick smallest muon NLL; store (nll_mu - nll_e) at the chosen seed."
        h.attrs["require_rpcflg_gt0"] = str(require_rpc)

        h.create_dataset("nevt", data=nevt_arr, dtype="i8")
        h.create_dataset("valid", data=valid_arr, dtype="bool")
        h.create_dataset("best_seed", data=seed_arr, dtype="i4")

        ds_hypo = h.create_dataset("best_hypo", data=hypo_arr, dtype="i4")
        ds_hypo.attrs["hypo_convention"] = "Always 2 (MUON) in this file."

        h.create_dataset("best_nll", data=nll_arr, dtype="f8")

        ds_mom = h.create_dataset("best_mom", data=mom_arr, dtype="f8")
        ds_mom.attrs["unit"] = "MeV/c"

        ds_pos = h.create_dataset("best_pos", data=pos_arr, dtype="f8")
        ds_pos.attrs["shape"] = "(N,3)"
        ds_pos.attrs["order"] = "x,y,z"
        ds_pos.attrs["unit"] = "cm"

        ds_dir = h.create_dataset("best_dir", data=dir_arr, dtype="f8")
        ds_dir.attrs["shape"] = "(N,3)"
        ds_dir.attrs["order"] = "dx,dy,dz"
        ds_dir.attrs["note"] = "unit vector"

        # extra diagnostic: nll_mu - nll_e at chosen seed
        h.create_dataset("dnll_mu_minus_e", data=dnll_arr, dtype="f8")
        
        
    POS_KEY = "best_pos"
    DIR_KEY = "best_dir"
   
    with h5py.File(args.output, "r+") as f_out:
        if POS_KEY not in f_out or DIR_KEY not in f_out:
            raise KeyError(f"{POS_KEY} / {DIR_KEY} not found in {args.output}")

        positions  = f_out[POS_KEY][...]   # (N, 3)
        directions = f_out[DIR_KEY][...]   # (N, 3)

        result = infer_exit_pos.ray_cylinder_intersection_yaxis_batch(positions, directions)
        
        if isinstance(result, dict):
            t_exit, X_exit, kind_exit = result["t"], result["X"], result["kind"]
        else:
            t_exit, X_exit, kind_exit = result  # 假设是 (t, X, kind)

        def upsert(name, data, *, dtype=None):
            if name in f_out:
                del f_out[name]
            if dtype is not None:
                f_out.create_dataset(name, data=data, dtype=dtype)
            else:
                f_out.create_dataset(name, data=data)

        upsert("exit_points", X_exit)               # (N, 3)
        upsert("exit_t", t_exit)                    # (N,)

        str_dtype = h5py.string_dtype(encoding="utf-8")
        upsert("exit_kind", np.asarray(kind_exit, dtype=object), dtype=str_dtype)

        hit_mask = np.isfinite(t_exit)
        counts = hit_mask.sum()
        print(f"Computed exit points for {counts} / {len(t_exit)} rays ({counts / len(t_exit) * 100:.2f}%)")
        upsert("exit_hit_mask", hit_mask.astype(np.uint8))

    print(f"Saved HDF5: {args.output}")
    print(f"Total entries: {len(nevt_arr)}, valid picks: {int(valid_arr.sum())}, invalid: {int((~valid_arr).sum())}")

if __name__ == "__main__":
    main()