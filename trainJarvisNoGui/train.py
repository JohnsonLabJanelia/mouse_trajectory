import argparse
from jarvis.train_interface import train_efficienttrack, train_hybridnet

def main():
    p = argparse.ArgumentParser("Train CenterDetect, KeypointDetect, then HybridNet (3D-only + finetune)")
    p.add_argument("--project", default="fin1")
    p.add_argument("--epochs-center", type=int, default=1)
    p.add_argument("--epochs-keypoint", type=int, default=1)
    p.add_argument("--epochs-3d", type=int, default=1)
    p.add_argument("--epochs-finetune", type=int, default=1)
    p.add_argument("--pretrain-center", default="None")   # 'None' | 'EcoSet' | path/to/.pth
    p.add_argument("--pretrain-keypoint", default="None") # 'None' | 'EcoSet' | path/to/.pth
    p.add_argument("--weights-keypoint", default="latest")
    p.add_argument("--weights-hybridnet", default="None")
    p.add_argument("--mode", default="3D_only", choices=["3D_only", "all"], help="Training mode: '3D_only' or 'all'")
    p.add_argument("--skip-center", action="store_true")
    p.add_argument("--skip-keypoint", action="store_true")
    p.add_argument("--skip-3d", action="store_true")
    p.add_argument("--skip-finetune", action="store_true")
    args = p.parse_args()

    if not args.skip_center:
        print(f"[Train] CenterDetect | project={args.project} epochs={args.epochs_center} ")
        train_efficienttrack('CenterDetect', args.project, args.epochs_center, args.pretrain_center)

    if not args.skip_keypoint:
        print(f"[Train] KeypointDetect | project={args.project} epochs={args.epochs_keypoint} ")
        train_efficienttrack('KeypointDetect', args.project, args.epochs_keypoint, args.pretrain_keypoint)
    # ...existing code...
    if not args.skip_3d:
        mode_display = "all" if args.mode == "all" else "3D-only"
        print(f"[Train] HybridNet {mode_display} | project={args.project} epochs={args.epochs_3d} ")
        # Use pretrain-keypoint if provided, otherwise use weights-keypoint
        weights_keypoint = args.pretrain_keypoint if args.pretrain_keypoint != "None" else args.weights_keypoint
        # Expected signature: train_hybridnet(project_name, num_epochs, weights, weights_hybridnet, mode, ...)
        train_hybridnet(
            args.project,
            args.epochs_3d,
            weights_keypoint,            # weights (e.g., KeypointDetect weights)
            args.weights_hybridnet,     # weights_hybridnet or None
            args.mode
        )


if __name__ == "__main__":
    main()

