import os, time, yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from utils.plotting import plot_confusion_matrix, plot_curve
from utils.metrics  import MetricsCallback, write_summary
from utils.latency  import measure_latency
from utils.misc     import visualize_model_graph, show_gpu_info

from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_model_cls import get_model_cls
from utils.seed import seed_everything

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

def train_and_test(config):
    model_name = config["model"]
    dataset_name = config["dataset_name"]
    seed = config["seed"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_dir = (
        Path(__file__).resolve().parent /
        f"results/{model_name}_{dataset_name}_seed-{seed}_aug-{config['preprocessing']['interaug']}"
        f"_GPU{config['gpu_id']}_{timestamp}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["checkpoints", "confmats", "curves"]:
        (result_dir / sub).mkdir(parents=True, exist_ok=True)

    with open(result_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    model_cls = get_model_cls(model_name)
    datamodule_cls = get_datamodule_cls(dataset_name)

    config["model_kwargs"]["n_channels"] = datamodule_cls.channels
    config["model_kwargs"]["n_classes"] = datamodule_cls.classes

    subj_cfg = config["subject_ids"]
    subject_ids = datamodule_cls.all_subject_ids if subj_cfg == "all" else \
                  [subj_cfg] if isinstance(subj_cfg, int) else \
                  subj_cfg

    test_accs, test_losses, test_kappas = [], [], []
    train_times, test_times, response_times = [], [], []
    all_confmats = []

    for subject_id in subject_ids:
        print(f"\n>>> Training on subject: {subject_id}")

        seed_everything(config["seed"])
        metrics_callback = MetricsCallback()

        trainer = Trainer(
            max_epochs=config["max_epochs"],
            devices='auto',
            num_sanity_val_steps=0,
            accelerator="auto",
            strategy="auto",
            logger=False,
            enable_checkpointing=False,
            callbacks=[metrics_callback]
        )

        datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)
        model = model_cls(**config["model_kwargs"], max_epochs=config["max_epochs"])

        param_count = sum(p.numel() for p in model.parameters())

        st_train = time.time()
        trainer.fit(model, datamodule=datamodule)
        train_times.append((time.time() - st_train) / 60)

        st_test = time.time()
        test_results = trainer.test(model, datamodule)
        test_duration = time.time() - st_test
        test_times.append(test_duration)

        sample_x, _ = datamodule.test_dataset[0]
        input_shape = (1, *sample_x.shape)
        device_str = "cpu"
        lat_ms = measure_latency(model, input_shape, device=device_str)
        response_times.append(lat_ms)

        test_accs.append(test_results[0]["test_acc"])
        test_losses.append(test_results[0]["test_loss"])
        test_kappas.append(test_results[0]["test_kappa"])

        cm = model.test_confmat.numpy()
        all_confmats.append(cm)

        if config.get("plot_cm_per_subject", False):
            plot_confusion_matrix(
                cm, save_path=result_dir / f"confmats/confmat_subject_{subject_id}.png",
                class_names=datamodule_cls.class_names,
                title=f"Confusion Matrix - Subject {subject_id}",
            )

        if metrics_callback.train_loss and metrics_callback.val_loss:
            plot_curve(metrics_callback.train_loss, metrics_callback.val_loss,
                       "Loss", subject_id, result_dir / f"curves/subject_{subject_id}_loss.png")
        if metrics_callback.train_acc and metrics_callback.val_acc:
            plot_curve(metrics_callback.train_acc, metrics_callback.val_acc,
                       "Accuracy", subject_id, result_dir / f"curves/subject_{subject_id}_acc.png")

        if config.get("save_checkpoint", False):
            ckpt_path = result_dir / f"checkpoints/subject_{subject_id}_model.ckpt"
            trainer.save_checkpoint(ckpt_path)

    write_summary(result_dir, model_name, dataset_name, subject_ids, param_count,
                  test_accs, test_losses, test_kappas, train_times, test_times, response_times)

    if config.get("plot_cm_average", True) and all_confmats:
        avg_cm = np.mean(np.stack(all_confmats), axis=0)
        plot_confusion_matrix(
            avg_cm, save_path=result_dir / "confmats/avg_confusion_matrix.png",
            class_names=datamodule_cls.class_names,
            title="Average Confusion Matrix",
        )


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="tcformer_gaf",
        help="Name of the model to use. Options:\n"
             "tcformer, tcformer_gaf, atcnet, d-atcnet, atcnet_2_0, eegnet, shallownet, basenet\n"
             "eegtcnet, eegconformer, tsseffnet, eegdeformer, sst_dpn, ctnet, mscformer"
    )
    parser.add_argument("--dataset", type=str, default="bcic2a",
        help="Name of the dataset to use. Options: bcic2a, bcic2b, hgd, reh_mi, bcic3"
    )
    parser.add_argument("--loso", action="store_true", default=True,
        help="Enable subject-independent (LOSO) mode"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed value (overrides config if specified)")
    parser.add_argument("--interaug", action="store_true",
                        help="Enable inter-trial augmentation (overrides config if specified)")
    parser.add_argument("--no_interaug", action="store_true",
                        help="Disable inter-trial augmentation (overrides config if specified)")
    return parser.parse_args()


def run():
    args = parse_arguments()

    config_path = os.path.join(CONFIG_DIR, f"{args.model}.yaml")
    with open(config_path, encoding="utf-8") as f:      # ← fix encoding
        config = yaml.safe_load(f)

    if args.loso:
        config["dataset_name"] = args.dataset + "_loso"
        config["max_epochs"] = config["max_epochs_loso_hgd"] if args.dataset == "hgd" \
                                else config["max_epochs_loso"]
        config["model_kwargs"]["warmup_epochs"] = config["model_kwargs"]["warmup_epochs_loso"]
    else:
        config["dataset_name"] = args.dataset
        config["max_epochs"] = config["max_epochs_2b"] if args.dataset == "bcic2b" \
                                else config["max_epochs"]

    config["preprocessing"] = config["preprocessing"][args.dataset]
    config["preprocessing"]["z_scale"] = config["z_scale"]

    if args.interaug:
        config["preprocessing"]["interaug"] = True
    elif args.no_interaug:
        config["preprocessing"]["interaug"] = False
    else:
        config["preprocessing"]["interaug"] = config["interaug"]
    config.pop("interaug", None)

    config["gpu_id"] = args.gpu_id
    if args.seed is not None:
        config["seed"] = args.seed

    config["plot_cm_per_subject"] = True
    config["plot_cm_average"]     = True

    train_and_test(config)


if __name__ == "__main__":
    run()
