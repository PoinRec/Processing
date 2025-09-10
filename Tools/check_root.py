import os, sys
import ROOT

if len(sys.argv) < 1+1:
    print("Usage: python check_particles_and_pairs.py /path/to/file_or_glob.root")
    sys.exit(1)

rootfile = sys.argv[1]

# load WCSim dictionary
if "WCSIM_BUILD_DIR" in os.environ:
    lib = os.path.join(os.environ["WCSIM_BUILD_DIR"], "lib", "libWCSimRoot.so")
else:
    lib = "/opt/WCSim/build/install/lib/libWCSimRoot.so"

if ROOT.gSystem.Load(lib) != 0:
    print(f"ERROR: failed to load {lib}. Make sure env is sourced.")
    sys.exit(1)

# build chain
t = ROOT.TChain("wcsimT")
if t.Add(rootfile) == 0:
    print(f"ERROR: no files matched: {rootfile}")
    sys.exit(1)

print("Total entries:", t.GetEntries())


wcsim_event = ROOT.WCSimRootEvent()

try:
    t.SetBranchAddress("wcsimrootevent", wcsim_event)
except Exception:
    t.SetBranchAddress("wcsimrootevent", ROOT.AddressOf(wcsim_event))

has_pair_count = 0

n = t.GetEntries()
for iev in range(n):
    t.GetEntry(iev)
    trg = wcsim_event.GetTrigger(0)
    if not trg:
        continue
    tracks = trg.GetTracks()
    if not tracks:
        continue

    prim_pdg = []
    for j in range(trg.GetNtrack()):
        trk = tracks.At(j)
        if not trk:
            continue
        if trk.GetFlag() == 0 and trk.GetParenttype() == 0:
            prim_pdg.append(trk.GetIpnu())


    e_minus = 0
    e_plus  = 0
    kConv = getattr(ROOT, "kConv")
    for j in range(trg.GetNtrack()):
        trk = tracks.At(j)
        if not trk:
            continue
        pid = trk.GetIpnu()
        if abs(pid) == 11 and trk.GetParentId() == 1 and trk.GetCreatorProcess() == kConv:
            if pid == 11:
                e_minus += 1
            else:
                e_plus  += 1

    has_pair = (e_minus > 0 and e_plus > 0)
    
    if has_pair: has_pair_count += 1

    print(
        f"Event {iev}: len(particles)={len(prim_pdg)}, PDG={prim_pdg}; "
        f"conv e-={e_minus}, e+={e_plus}, pair={'YES' if has_pair else 'NO'}"
    )
    
print(f"Total events with pair: {has_pair_count} / {n} ({100.0*has_pair_count/n:.1f}%)")