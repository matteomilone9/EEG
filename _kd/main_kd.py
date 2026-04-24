# main_kd.py — Entry point separato per esperimenti Teacher-Student KD
# Import aggiornati: config_kd e pipelines_kd (già self-contained)
# ============================================================

import argparse
import numpy as np

from config_kd import KD_CFG
from pipelines_kd import (
    run_subject_kd,
    run_subject_kd_multiseed,
    run_subject_kd_align,
    run_subject_kd_align_multiseed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline Teacher-Student KD separata (EEG student, EEG+GAF teacher)"
    )
    parser.add_argument("--subject",    type=int, default=None,
                        help="ID singolo soggetto da eseguire (override del config)")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Seed per run single-seed (override del config)")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Esegue la pipeline KD in modalità multi-seed")
    parser.add_argument("--quiet",      action="store_true",
                        help="Riduce l'output verboso")
    parser.add_argument("--kd-align",   action="store_true",
                        help="Forza l'uso della pipeline KD-Align")
    parser.add_argument("--no-kd-align",action="store_true",
                        help="Forza l'uso della pipeline KD standard")
    return parser.parse_args()


def main():
    args = parse_args()

    default_seed       = KD_CFG.get("default_seed", 42)
    default_multi_seed = KD_CFG.get("default_multi_seed", False)
    cfg_use_kd_align   = KD_CFG.get("use_kd_align", False)

    seed       = args.seed if args.seed is not None else default_seed
    multi_seed = args.multi_seed or default_multi_seed

    if args.kd_align and args.no_kd_align:
        raise ValueError("Non puoi usare contemporaneamente --kd-align e --no-kd-align")

    if args.kd_align:
        use_kd_align = True
    elif args.no_kd_align:
        use_kd_align = False
    else:
        use_kd_align = cfg_use_kd_align

    if args.subject is not None:
        subjects_to_run = [args.subject]
    else:
        if KD_CFG.get("run_all_subjects", True):
            subjects_to_run = KD_CFG["subject_ids"]
        else:
            subjects_to_run = [KD_CFG["single_subject"]]

    exp_name = "KD-ALIGN" if use_kd_align else "KD STANDARD"

    print("\n" + "=" * 80)
    print("KD EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Esperimento  : {exp_name}")
    print(f"Soggetti     : {subjects_to_run}")
    print(f"Multi-seed   : {multi_seed}")
    print(f"Seed default : {seed}")
    print(f"use_gaf      : {KD_CFG.get('use_gaf', False)}")
    print(f"use_kd_align : {use_kd_align}")
    print("=" * 80)

    all_results = []

    for sub_id in subjects_to_run:
        print(f"\n\n{'#' * 80}")
        print(f"Esecuzione soggetto {sub_id}")
        print(f"{'#' * 80}")

        if multi_seed:
            if use_kd_align:
                out = run_subject_kd_align_multiseed(sub_id)
            else:
                out = run_subject_kd_multiseed(sub_id)
        else:
            if use_kd_align:
                out = run_subject_kd_align(sub_id, seed=seed, verbose=not args.quiet)
            else:
                out = run_subject_kd(sub_id, seed=seed, verbose=not args.quiet)

        all_results.append(out)

    print("\n" + "=" * 80)
    print("RIEPILOGO FINALE")
    print("=" * 80)

    # Riferimento EEG-only best (multi-seed 5x) per calcolo delta
    eeg_only_best = {
        1: 87.78, 2: 58.61, 3: 96.53, 4: 79.51, 5: 67.85,
        6: 61.53, 7: 93.33, 8: 85.21, 9: 87.50,
    }

    if multi_seed:
        student_means = [r["student_acc_mean"] for r in all_results]
        teacher_means = [r["teacher_acc_mean"] for r in all_results]

        print(f"{'Sub':<5} {'Teacher':>18} {'Student':>22} {'vs EEG-Only':>13} {'Std':>8}")
        print("-" * 80)
        for r in all_results:
            s = r["subject"]
            delta = r["student_acc_mean"] - eeg_only_best.get(s, 0)
            arrow = "▲" if delta >= 0 else "▼"
            flag  = " ⚠" if r["student_acc_std"] > 5.0 else ""
            print(
                f"S{s:02d} | "
                f"Teacher {r['teacher_acc_mean']:5.2f}±{r['teacher_acc_std']:4.2f}% | "
                f"Student {r['student_acc_mean']:5.2f}±{r['student_acc_std']:4.2f}% | "
                f"{arrow}{abs(delta):5.2f}pp{flag}"
            )

        print("-" * 80)
        avg_student = np.mean(student_means)
        avg_teacher = np.mean(teacher_means)
        avg_eeg     = np.mean(list(eeg_only_best.values()))
        print(f"Media Teacher         : {avg_teacher:.2f}%")
        print(f"Media Student KD/Align: {avg_student:.2f}%")
        print(f"Media EEG-Only Best   : {avg_eeg:.2f}%  (riferimento)")
        print(f"Gap KD/Align vs EEG   : {avg_student - avg_eeg:+.2f}pp")

    else:
        student_accs = [r["student_acc"] for r in all_results]
        teacher_accs = [r["teacher_acc"] for r in all_results]

        print(f"{'Sub':<5} {'Teacher':>14} {'Student':>20} {'vs EEG-Only':>13}")
        print("-" * 80)
        for r in all_results:
            s = r["subject"]
            delta = r["student_acc"] - eeg_only_best.get(s, 0)
            arrow = "▲" if delta >= 0 else "▼"
            print(
                f"S{s:02d} | "
                f"Teacher {r['teacher_acc']:5.2f}% | "
                f"Student {r['student_acc']:5.2f}% | "
                f"{arrow}{abs(delta):5.2f}pp"
            )

        print("-" * 80)
        avg_student = np.mean(student_accs)
        avg_teacher = np.mean(teacher_accs)
        avg_eeg     = np.mean(list(eeg_only_best.values()))
        print(f"Media Teacher         : {avg_teacher:.2f}%")
        print(f"Media Student KD/Align: {avg_student:.2f}%")
        print(f"Media EEG-Only Best   : {avg_eeg:.2f}%  (riferimento)")
        print(f"Gap KD/Align vs EEG   : {avg_student - avg_eeg:+.2f}pp")

    print("=" * 80)


if __name__ == "__main__":
    main()
