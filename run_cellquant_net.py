import os
import argparse
import subprocess
import time


def run_cmd(cmd, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80 + "\n")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":

    t_0 = time.time()
    parser = argparse.ArgumentParser(description="Run WSI-QA then CP-Net")

    parser.add_argument("--cpu_workers", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--cell_connectivity", action="store_true")
    parser.add_argument("--model_type", type=str, default="tnmi_20x.pth")

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))

    test_qa_path = os.path.join(project_root, "WSI_QA", "test_qa.py")
    test_cpnet_path = os.path.join(project_root, "CP-Net", "test_cp_net.py")

    qa_cmd = [
        "python", test_qa_path,
        "--cpu_workers", str(args.cpu_workers),
        "--batch_size", str(args.batch_size),
    ]

    cpnet_cmd = [
        "python", test_cpnet_path,
        "--model_type", args.model_type,
    ]

    if args.cell_connectivity:
        cpnet_cmd.append("--cell_connectivity")

    run_cmd(
        qa_cmd,
        "RUNNING WSI-QA PIPELINE"
    )

    run_cmd(
        cpnet_cmd,
        "RUNNING CP-NET PIPELINE"
    )

    print("\n" + "=" * 80)
    print("CELLQUANT-NET PIPELINE COMPLETED SUCCESSFULLY")
    print("Total inference time:", time.time() - t_0)
    print("=" * 80)