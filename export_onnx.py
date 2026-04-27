import torch
import torch.nn as nn
import torchvision.models as models

def export(checkpoint_path, onnx_path):
    print(f"Exporting: {onnx_path}")
    ckpt    = torch.load(checkpoint_path, map_location="cpu")
    classes = ckpt["classes"]
    model   = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print("Done!")


if __name__ == "__main__":
    from pathlib import Path

    onnx_dir = Path("onnx")
    onnx_dir.mkdir(exist_ok=True)

    for pt_file in sorted(Path("checkpoints").glob("*.pt")):
        onnx_path = onnx_dir / pt_file.with_suffix(".onnx").name
        export(str(pt_file), str(onnx_path))
