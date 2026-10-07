#!/usr/bin/env python3
"""GPT Fusion User Interface (UI).

PURPOSE:
# Proof-of-concept implementation of GPT-based reasoning to integrate complementary
# evidence from specialized vision models--ResNet-50 and SIIM-90 in this case--to
# enhance melanoma diagnosis.

REQUIREMENTS:
# Python package gradio
# MILK10K ResNet-50 model and SIIM-90 ensemble
# OpenAI API key to reason over ResNet-50 and SIIM-90 outputs with GPT-5.5

INPUT:
# Dermoscopic image: used by both MILK10K ResNet-50 and SIIM-90.
# Clinical close-up image: used as the second input to ResNet-50 model.

OUTPUT:
# Primary diagnosis harmonized to the Derm7pt diagnostic categories
# Top-3 differential diagnosis harmonized to the Derm7pt diagnostic categories
# Melanoma prediction
# Malignancy prediction

COMMAND TO START BACKEND: 
# Assuming ResNet-50 and SIIM-90 are in the default directories. 
# Below is a command to start the gradio backend:
    python gpt-fusion-ui.py 

# Explictly specifying ResNet-50 and SIIM-90 directories in command:
    python gpt-fusion-ui.py \
        --resnet_run_dir milk10k_train_base/runs/20260605_083158 \
        --siim_model_dir /Users/qwang/models/melanoma-winning-models \
        --device auto --n_test 1

FRONTEND URL:
    http://127.0.0.1:7860
"""

import argparse
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from openai import OpenAI

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESNET_REPO = (
    BASE_DIR /
    "milk10k_train_base"
)
if not RESNET_REPO.exists():
    raise FileNotFoundError(
        f"RESNET-50 repository not found:\n{RESNET_REPO}"
    )
sys.path.insert(0, str(RESNET_REPO))

SIIM_AVAILABLE = False
SIIM_UNAVAILABLE_REASON = None
siim = None

SIIM_REPO = (
    BASE_DIR /
    "SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master"
)
if SIIM_REPO.exists():
    try:
        sys.path.insert(0, str(SIIM_REPO))
        import siim90_predict_one as siim
        SIIM_AVAILABLE = True
    except Exception as exc:
        SIIM_UNAVAILABLE_REASON = f"SIIM-90 code could not be imported: {exc}"
else:
    SIIM_UNAVAILABLE_REASON = f"SIIM-90 repository not found: {SIIM_REPO}"

from utils.data import image_transforms
from utils.model import MultimodalNetSimple


# The simplified MILK10K families.
FAMILY_NAMES = {
    "MEL": "Melanoma",
    "BCC": "Basal cell carcinoma",
    "SCCKA": "Squamous cell carcinoma / keratoacanthoma",
    "AKIEC": "Actinic keratosis / intraepithelial carcinoma",
    "NV": "Melanocytic nevus",
    "BKL": "Benign keratinocytic lesion",
    "DF": "Dermatofibroma",
    "INF": "Inflammatory / infectious",
    "VASC": "Vascular lesion",
    "MAL_OTH": "Other malignant lesion",
    "BEN_OTH": "Other benign lesion",
    "NA": "Other / ungrouped",
}

MALIGNANT_FAMILIES = {"MEL", "BCC", "SCCKA", "MAL_OTH"}
PREMALIGNANT_FAMILIES = {"AKIEC"}
BENIGN_FAMILIES = {"NV", "BKL", "DF", "INF", "VASC", "BEN_OTH"}


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_resnet_model(checkpoint_path: Path, device: torch.device,
                      num_classes: int = 48):
    model = MultimodalNetSimple(num_classes=num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if (
        isinstance(ckpt, dict) and "model_state_dict" in ckpt
    ) else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_resnet_bundle(run_dir: Path, device: torch.device):
    with open(run_dir / "dataset-idx_to_class.json") as f:
        idx_to_class = json.load(f)
    with open(run_dir / "dataset-label_to_simplified.json") as f:
        label_to_simplified = json.load(f)

    models = []
    for fold in range(5):
        ckpt = run_dir / f"model_fold_{fold}_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing ResNet checkpoint: {ckpt}")
        print(f"Loading ResNet fold {fold}: {ckpt.name}")
        models.append(load_resnet_model(ckpt, device))

    return models, idx_to_class, label_to_simplified


def load_image_for_resnet(path: str, transform, device: torch.device):
    from PIL import Image
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def family_probabilities(probs, idx_to_class, label_to_simplified):
    fam_probs = {}
    for i, p in enumerate(probs.tolist()):
        dx = idx_to_class[str(i)]
        fam = label_to_simplified.get(dx, "NA")
        fam_probs[fam] = fam_probs.get(fam, 0.0) + float(p)
    return fam_probs


@torch.inference_mode()
def unique_top_families(mean_probs, idx_to_class, label_to_simplified, topk=3):
    """Match resnet_assess_derm7pt.py for differential selection."""
    sorted_idx = torch.argsort(mean_probs, descending=True).tolist()
    families = []
    diagnoses = []
    for idx in sorted_idx:
        dx = idx_to_class[str(idx)]
        fam = label_to_simplified.get(dx, "NA")
        prob = float(mean_probs[idx].item())
        if fam not in families:
            families.append(fam)
            diagnoses.append((dx, fam, prob))
        if len(families) >= topk:
            break
    return diagnoses


def predict_resnet(models, closeup_path, derm_path, transform, device,
                   idx_to_class, label_to_simplified):
    """Run ResNet-50 as in resnet_assess_derm7pt.py.
    Each paired clinical/dermoscopic lesion is run through all five MILK10K
    fold models. The 48-class probability vectors are averaged first.
    Differential diagnoses are then selected with unique_top_families(), while
    family probabilities are separately summed for melanoma/malignancy outputs.
    """
    clinic = load_image_for_resnet(closeup_path, transform, device)
    derm = load_image_for_resnet(derm_path, transform, device)

    fold_probs = []
    for model in models:
        logits = model(clinic, derm)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        fold_probs.append(probs)

    fold_probs = torch.stack(fold_probs)
    mean_probs = fold_probs.mean(dim=0)
    sd_probs = fold_probs.std(dim=0)

    fam_probs = family_probabilities(
        mean_probs, idx_to_class, label_to_simplified
    )

    melanoma = fam_probs.get("MEL", 0.0)
    invasive_malignancy = sum(
        fam_probs.get(f, 0.0) for f in MALIGNANT_FAMILIES
    )
    premalignant = sum(
        fam_probs.get(f, 0.0) for f in PREMALIGNANT_FAMILIES
    )
    concerning = invasive_malignancy + premalignant
    benign = sum(fam_probs.get(f, 0.0) for f in BENIGN_FAMILIES)

    # IMPORTANT: this is the Derm7pt assessment method. It is NOT ranking
    # summed family probabilities. It ranks the original 48 diagnoses and
    # retains the highest-ranked diagnosis from each unique family.
    top3_diagnoses = unique_top_families(
        mean_probs, idx_to_class, label_to_simplified, topk=3
    )

    return {
        "top3_diagnoses": top3_diagnoses,
        # Keep top3 as family/probability pairs for compact evidence display.
        "top3": [(fam, prob) for _, fam, prob in top3_diagnoses],
        # GPT fusion needs diagnosis + family + probability, matching the
        # columns produced by resnet_assess_derm7pt.py.
        "top3_raw": [(dx, prob) for dx, _, prob in top3_diagnoses],
        "family_probabilities": fam_probs,
        "fold_sd_probabilities": sd_probs,
        "melanoma_probability": float(melanoma),
        "invasive_malignancy_probability": float(invasive_malignancy),
        "premalignant_probability": float(premalignant),
        "clinically_concerning_probability": float(concerning),
        "benign_probability": float(benign),
    }


@torch.inference_mode()
def predict_siim90_one(image_path: str, model_dir: Path,
                       device: torch.device, n_test: int):
    """Probability ensemble following the SIIM-90 script.
    Models are loaded one at a time to avoid keeping all 90 networks in memory.
    First average checkpoints within architecture/kernel type, then average the
    architecture probabilities, matching siim90_assess_derm7pt.py.
    """
    ckpts = sorted(model_dir.rglob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoints found in {model_dir}")

    by_arch = {}
    for i, ckpt_path in enumerate(ckpts, start=1):
        print(f"SIIM-90 [{i}/{len(ckpts)}] {ckpt_path.name}")
        kernel_type = siim.extract_kernel_type(ckpt_path)
        config = siim.MODEL_CONFIGS[kernel_type]

        model = siim.build_model(config, device)
        state_dict = siim.load_state_dict_safely(ckpt_path, device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        x = siim.load_image(
            image_path=str(image_path),
            image_size=config["image_size"],
            device=device,
        )
        prob = siim.melanoma_probability_with_tta(
            model=model,
            x=x,
            config=config,
            device=device,
            n_test=n_test,
        )
        by_arch.setdefault(kernel_type, []).append(float(prob))

        del model, x
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    architecture_probs = {
        arch: float(np.mean(values)) for arch, values in by_arch.items()
    }
    probability_ensemble = float(np.mean(list(architecture_probs.values())))

    return {
        "melanoma_probability": probability_ensemble,
        "n_checkpoints": len(ckpts),
        "architecture_probabilities": architecture_probs,
    }


# Thresholds and GPT fusion prompt match those in gpt_fusion_assess_derm7pt.py.
# NOTE: SIIM rank ensemble is cohort-relative and cannot be computed for a
# single newly uploaded lesion, so the web app supplies the probability
# ensemble and explicitly marks rank ensemble as unavailable.
SIIM_MEL_THRESHOLD = 0.16081724078112586
RESNET_MEL_THRESHOLD = 0.18557974882423878
RESNET_MALIGNANCY_THRESHOLD = 0.22138070850633085

def parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

def build_fusion_prompt(resnet, siim_result):
    resnet_mel = float(resnet["melanoma_probability"])
    resnet_malign = float(resnet["clinically_concerning_probability"])
    siim_mel = float(siim_result["melanoma_probability"])

    resnet_mel_pos = int(resnet_mel >= RESNET_MEL_THRESHOLD)
    resnet_malign_pos = int(resnet_malign >= RESNET_MALIGNANCY_THRESHOLD)
    siim_mel_pos = int(siim_mel >= SIIM_MEL_THRESHOLD)

    top_dx = list(resnet["top3_diagnoses"])
    while len(top_dx) < 3:
        top_dx.append(("", "NA", 0.0))

    # Preserve the original assessment prompt's evidence and weighting logic.
    # The web deployment has no cohort-level SIIM rank ensemble.
    return f"""
You are a diagnostic fusion agent for a research classification task.
You are NOT allowed to use images. Use ONLY the model-output evidence below.

Your role:
- Do NOT independently invent a diagnosis.
- Integrate evidence from two AI systems.
- Treat the ResNet as the broader differential-diagnosis model.
- Treat the SIIM ensemble as a melanoma-specialist model.
- For melanoma prediction, trust SIIM more than ResNet because SIIM was specifically optimized for melanoma detection.
- For broad malignancy prediction, use SIIM mainly as melanoma evidence; use ResNet to assess non-melanoma malignancy such as BCC or SCCKA.
- A low SIIM score does NOT rule out non-melanoma malignancy.

Family labels must be exactly one of:
MEL, BCC, NV, BKL, DF, VASC, SCCKA, MISC

ResNet outputs:
- Top1 diagnosis: {top_dx[0][0]}
- Top1 family: {top_dx[0][1]}
- Top1 probability: {top_dx[0][2]}
- Top2 diagnosis: {top_dx[1][0]}
- Top2 family: {top_dx[1][1]}
- Top2 probability: {top_dx[1][2]}
- Top3 diagnosis: {top_dx[2][0]}
- Top3 family: {top_dx[2][1]}
- Top3 probability: {top_dx[2][2]}

ResNet family probabilities:
- MEL: {resnet["family_probabilities"].get("MEL", 0)}
- BCC: {resnet["family_probabilities"].get("BCC", 0)}
- NV: {resnet["family_probabilities"].get("NV", 0)}
- BKL: {resnet["family_probabilities"].get("BKL", 0)}
- DF: {resnet["family_probabilities"].get("DF", 0)}
- VASC: {resnet["family_probabilities"].get("VASC", 0)}
- SCCKA: {resnet["family_probabilities"].get("SCCKA", 0)}
- AKIEC: {resnet["family_probabilities"].get("AKIEC", 0)}
- MAL_OTH: {resnet["family_probabilities"].get("MAL_OTH", 0)}
- BEN_OTH: {resnet["family_probabilities"].get("BEN_OTH", 0)}
- INF: {resnet["family_probabilities"].get("INF", 0)}

ResNet threshold-adjusted interpretation:
- ResNet melanoma probability: {resnet_mel}
- ResNet melanoma threshold: {RESNET_MEL_THRESHOLD}
- ResNet melanoma positive: {resnet_mel_pos}
- ResNet malignancy probability: {resnet_malign}
- ResNet malignancy threshold: {RESNET_MALIGNANCY_THRESHOLD}
- ResNet malignancy positive: {resnet_malign_pos}

SIIM melanoma-specialist ensemble:
- SIIM melanoma probability ensemble: {siim_mel}
- SIIM melanoma rank ensemble: unavailable for a single web-uploaded lesion
- SIIM melanoma threshold: {SIIM_MEL_THRESHOLD}
- SIIM melanoma positive: {siim_mel_pos}

Suggested weighting:
- Melanoma prediction: SIIM 70-80%, ResNet 20-30%.
- Broad malignancy prediction: SIIM contributes strong melanoma evidence only; ResNet contributes broader non-melanoma malignancy evidence.

Return ONLY valid JSON:

{{
  "primary_diagnosis": "",
  "primary_family": "",
  "top1_probability": 0.0,
  "top3_differential": [
    {{"diagnosis": "", "family": "", "probability": 0.0}},
    {{"diagnosis": "", "family": "", "probability": 0.0}},
    {{"diagnosis": "", "family": "", "probability": 0.0}}
  ],
  "melanoma_probability": 0.0,
  "malignancy_probability": 0.0,
  "melanoma_prediction": 0,
  "malignancy_prediction": 0,
  "followed_resnet": true,
  "followed_siim": true,
  "resnet_weight": 0.0,
  "siim_weight": 0.0,
  "confidence": 0.0,
  "melanoma_decision_basis": "",
  "malignancy_decision_basis": "",
  "brief_rationale": ""
}}
"""

def ask_fusion_llm(client, model, resnet, siim_result):
    response = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [{"type": "input_text",
                         "text": build_fusion_prompt(resnet, siim_result)}],
        }],
    )
    return parse_json_response(response.output_text)

def pad_top3(top3):
    if not isinstance(top3, list):
        top3 = []
    while len(top3) < 3:
        top3.append({"diagnosis": "", "family": "MISC", "probability": 0.0})
    return top3[:3]

def pct(x):
    return f"{100.0 * x:.1f}%"


def make_app(args):
    device = get_device(args.device)
    print(f"Using device: {device}")

    resnet_models, idx_to_class, label_to_simplified = load_resnet_bundle(
        Path(args.resnet_run_dir), device
    )
    transform = image_transforms["val"]
    siim_model_dir = Path(args.siim_model_dir)
    fusion_client = OpenAI()

    # Check SIIM-90 availability without preventing the app from starting.
    siim_ckpts = list(siim_model_dir.rglob("*.pth")) if siim_model_dir.exists() else []
    siim_ready = SIIM_AVAILABLE and bool(siim_ckpts)
    if siim_ready:
        print(f"SIIM-90 available: {len(siim_ckpts)} checkpoint(s) found in {siim_model_dir}")
    else:
        reason = SIIM_UNAVAILABLE_REASON or f"No .pth checkpoints found in {siim_model_dir}"
        print(f"SIIM-90 unavailable; continuing with ResNet-50 only. {reason}")

    def analyze_image(derm_path, closeup_path):
        if not derm_path:
            raise gr.Error("Please upload a dermoscopic image.")
        if not closeup_path:
            raise gr.Error("Please upload the matching clinical close-up image.")

        resnet = predict_resnet(
            resnet_models, closeup_path, derm_path, transform, device,
            idx_to_class, label_to_simplified,
        )
        if not siim_ready:
            raise gr.Error(
                "SIIM-90 is required for GPT Fusion, but its code/checkpoints "
                "were not found on this computer."
            )

        try:
            siim_result = predict_siim90_one(
                derm_path, siim_model_dir, device, args.n_test
            )
        except Exception as exc:
            raise gr.Error(f"SIIM-90 inference failed: {exc}")

        try:
            fused = ask_fusion_llm(
                fusion_client, args.gpt_model, resnet, siim_result
            )
        except Exception as exc:
            raise gr.Error(f"GPT Fusion reasoning failed: {exc}")

        top3 = pad_top3(fused.get("top3_differential", []))
        differential = pd.DataFrame([
            {
                "Rank": i,
                "Differential diagnosis": x.get("diagnosis", ""),
                "Family": x.get("family", ""),
                "Probability": pct(float(x.get("probability", 0.0))),
            }
            for i, x in enumerate(top3, 1)
        ])

        melanoma = pd.DataFrame([{
            "GPT Fusion prediction":
                "Melanoma" if int(fused.get("melanoma_prediction", 0)) else "Not melanoma",
            "Probability": pct(float(fused.get("melanoma_probability", 0.0))),
            "Confidence": pct(float(fused.get("confidence", 0.0))),
        }])

        malignancy = pd.DataFrame([{
            "GPT Fusion prediction":
                "Malignant / concerning" if int(fused.get("malignancy_prediction", 0))
                else "Not malignant / concerning",
            "Probability": pct(float(fused.get("malignancy_probability", 0.0))),
        }])

        model_evidence = (
            f"### ResNet-50\n"
            f"- Melanoma probability: **{pct(resnet['melanoma_probability'])}**\n"
            f"- Clinically concerning probability: "
            f"**{pct(resnet['clinically_concerning_probability'])}**\n"
            f"- Top-3 differential evidence: " +
            ", ".join(
                f"{dx} [{fam}] ({pct(prob)})"
                for dx, fam, prob in resnet["top3_diagnoses"]
            ) +
            f"\n\n### SIIM-90\n"
            f"- Melanoma probability ensemble: "
            f"**{pct(siim_result['melanoma_probability'])}**\n"
            f"- Checkpoints used: **{siim_result['n_checkpoints']}**"
        )

        fusion_details = (
            f"**Primary diagnosis:** {fused.get('primary_diagnosis','')} "
            f"({fused.get('primary_family','')})\n\n"
            f"**Melanoma decision basis:** "
            f"{fused.get('melanoma_decision_basis','')}\n\n"
            f"**Malignancy decision basis:** "
            f"{fused.get('malignancy_decision_basis','')}\n\n"
            f"**Rationale:** {fused.get('brief_rationale','')}\n\n"
            f"**Reported weights:** ResNet "
            f"{fused.get('resnet_weight','')}; SIIM {fused.get('siim_weight','')}"
        )
        return differential, melanoma, malignancy, model_evidence, fusion_details

    css = """
    .compact-image {max-height: 300px !important;}
    .compact-image img {max-height: 260px !important; object-fit: contain !important;}
    """
    with gr.Blocks(title="GPT Fusion", css=css) as demo:
        gr.Markdown(
            "# GPT Fusion\n"
            "### Multimodel skin-lesion analysis\n"
            "ResNet-50 and SIIM-90 first analyze the lesion. GPT-5.5 then "
            "reasons over **only their model outputs** to produce the fused result."
        )
        with gr.Row():
            derm = gr.Image(
                type="filepath", label="Dermoscopic image",
                height=280, elem_classes=["compact-image"]
            )
            closeup = gr.Image(
                type="filepath", label="Clinical close-up image",
                height=280, elem_classes=["compact-image"]
            )

        analyze = gr.Button("Analyze lesion", variant="primary")

        gr.Markdown("## GPT Fusion — Top-3 differential diagnosis")
        differential = gr.Dataframe(
            headers=["Rank","Differential diagnosis","Family","Probability"],
            interactive=False,
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("## GPT Fusion — Melanoma prediction")
                melanoma = gr.Dataframe(interactive=False)
            with gr.Column():
                gr.Markdown("## GPT Fusion — Malignancy prediction")
                malignancy = gr.Dataframe(interactive=False)

        with gr.Accordion("Model evidence / individual CNN outputs", open=False):
            model_evidence = gr.Markdown()
        with gr.Accordion("Fusion details", open=False):
            fusion_details = gr.Markdown()

        gr.Markdown("*Research-use prototype. Model outputs are not a clinical diagnosis.*")

        analyze.click(
            fn=analyze_image,
            inputs=[derm, closeup],
            outputs=[
                differential, melanoma, malignancy,
                model_evidence, fusion_details
            ],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="GPT Fusion")
    parser.add_argument(
        "--resnet_run_dir",
        default="milk10k_train_base/runs/20260605_083158",
        help="Directory containing the MILK10K model_fold_*_best.pt files and dataset JSON files",
    )
    parser.add_argument(
        "--siim_model_dir",
        default="/Users/qwang/models/melanoma-winning-models",
        help="Directory containing the SIIM-90 .pth checkpoints",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--n_test", type=int, default=1, choices=[1, 8])
    parser.add_argument("--gpt_model", default="gpt-5.5")
    parser.add_argument("--server_name", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = make_app(args)
    demo.launch(server_name=args.server_name, server_port=args.server_port)
