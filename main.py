# main.py — Entry point triforcato: WS / LOSO / LOSO+FT
import numpy as np
from config import CFG, HARD_SUBJECTS, _mode_tag
from preprocessing import build_loso_cache
from pipelines import (
    run_subject, run_subject_multiseed,
    run_loso_fold, run_loso_fold_multiseed,
    run_loso_ft_fold, run_loso_ft_fold_multiseed,
)

SUBJECT_DEBUG = 2
RUN_ALL = True

if not RUN_ALL:
    seed = CFG['seeds'][0] if CFG['multi_seed'] else 42
    if CFG.get('loso_ft_approach', False):
        cache = build_loso_cache(CFG)
        run_loso_ft_fold(SUBJECT_DEBUG, cache, seed=seed)
    elif CFG['loso_approach']:
        cache = build_loso_cache(CFG)
        run_loso_fold(SUBJECT_DEBUG, cache, seed=seed)
    else:
        run_subject(SUBJECT_DEBUG, seed=seed)

else:
    results   = {}
    proto_tag = ("LOSO+FT" if CFG.get('loso_ft_approach', False)
                 else ("LOSO" if CFG['loso_approach'] else "WS"))

    if CFG.get('loso_ft_approach', False):
        print("\n[v16] Modalità LOSO+FT — caricamento cache globale...")
        cache = build_loso_cache(CFG)
        for s in range(1, CFG['n_subjects'] + 1):
            if CFG['multi_seed']:
                mean_acc, std_acc, mean_k, std_k, seed_accs, mean_loso = \
                    run_loso_ft_fold_multiseed(s, cache)
            else:
                mean_acc, _, mean_k, _, seed_accs, mean_loso = \
                    (*run_loso_ft_fold(s, cache, seed=42)[:2], 0.0, 0.0,
                     [run_loso_ft_fold(s, cache, seed=42)[0]], run_loso_ft_fold(s, cache, seed=42)[2])
                acc_ft, kappa, acc_loso = run_loso_ft_fold(s, cache, seed=42)
                mean_acc, mean_k, mean_loso = acc_ft, kappa, acc_loso
                std_acc = 0.0; seed_accs = [acc_ft]
            results[s] = {'acc': mean_acc, 'std': std_acc, 'kappa': mean_k,
                          'kappa_std': 0.0, 'seeds': seed_accs, 'loso_acc': mean_loso}

    elif CFG['loso_approach']:
        print("\n[v16] Modalità LOSO — caricamento cache globale...")
        cache = build_loso_cache(CFG)
        for s in range(1, CFG['n_subjects'] + 1):
            if CFG['multi_seed']:
                mean_acc, std_acc, mean_k, std_k, seed_accs = run_loso_fold_multiseed(s, cache)
            else:
                mean_acc, mean_k = run_loso_fold(s, cache, seed=42)
                std_acc = 0.0; seed_accs = [mean_acc]
            results[s] = {'acc': mean_acc, 'std': std_acc, 'kappa': mean_k,
                          'kappa_std': 0.0, 'seeds': seed_accs, 'loso_acc': mean_acc}
    else:
        for s in range(1, CFG['n_subjects'] + 1):
            if CFG['multi_seed']:
                mean_acc, std_acc, mean_k, std_k, seed_accs = run_subject_multiseed(s)
            else:
                mean_acc, mean_k = run_subject(s, seed=42)
                std_acc = 0.0; seed_accs = [mean_acc]
            results[s] = {'acc': mean_acc, 'std': std_acc, 'kappa': mean_k,
                          'kappa_std': 0.0, 'seeds': seed_accs, 'loso_acc': None}

    # ── Tabella finale ────────────────────────────────────────
    accs    = [results[s]['acc']   for s in range(1, 10)]
    kappas  = [results[s]['kappa'] for s in range(1, 10)]
    n_s     = len(CFG['seeds']) if CFG['multi_seed'] else 1
    ms_tag  = f"Multi-seed ({n_s}×)" if CFG['multi_seed'] else "Single-seed"
    show_loso_col = CFG.get('loso_ft_approach', False)

    print(f"\n{'='*65}")
    print(f" {_mode_tag(CFG)} | {ms_tag}")
    print(f"{'='*65}")
    if CFG['multi_seed']:
        hdr = f"{'Sub':>4} | {'Mean (%)':>9} | {'Std':>6} | {'Kappa':>7}"
        if show_loso_col: hdr += f" | {'LOSO(%)':>8} | {'Δ FT':>7}"
        print(hdr); print('-' * len(hdr))
        for s in range(1, 10):
            tag = ' *' if s in HARD_SUBJECTS else '  '
            row = (f" S{s:02d}{tag}| {results[s]['acc']:9.2f} | "
                   f"{results[s]['std']:6.2f} | {results[s]['kappa']:7.4f}")
            if show_loso_col:
                la = results[s]['loso_acc']
                row += f" | {la:8.2f} | {results[s]['acc']-la:+7.2f}"
            print(row)
        print('-' * len(hdr))
        avg_row = (f"  Avg | {np.mean(accs):9.2f} | "
                   f"{np.mean([results[s]['std'] for s in range(1,10)]):6.2f} | "
                   f"{np.mean(kappas):7.4f}")
        if show_loso_col:
            loso_accs = [results[s]['loso_acc'] for s in range(1,10)]
            avg_row += f" | {np.mean(loso_accs):8.2f} | {np.mean(accs)-np.mean(loso_accs):+7.2f}"
        print(avg_row)
    else:
        hdr = f"{'Sub':>4} | {'Acc (%)':>8} | {'Kappa':>7}"
        if show_loso_col: hdr += f" | {'LOSO(%)':>8} | {'Δ FT':>7}"
        print(hdr); print('-' * len(hdr))
        for s in range(1, 10):
            tag = ' *' if s in HARD_SUBJECTS else '  '
            row = f" S{s:02d}{tag}| {results[s]['acc']:8.2f} | {results[s]['kappa']:7.4f}"
            if show_loso_col:
                la = results[s]['loso_acc']
                row += f" | {la:8.2f} | {results[s]['acc']-la:+7.2f}"
            print(row)
        print('-' * len(hdr))
        print(f"  Avg | {np.mean(accs):8.2f} | {np.mean(kappas):7.4f}")
        print(f"  Std | {np.std(accs):8.2f} | {np.std(kappas):7.4f}")

    print(f" * = soggetto difficile (CFG override attivo)")
    print(f"\n✅ [{proto_tag}] Accuracy: {np.mean(accs):.2f} ± {np.std(accs):.2f}%")