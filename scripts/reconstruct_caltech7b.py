"""Run the pinned author scorer for the 100-row Flowers check or Caltech-101 reconstruction."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np

MODELS = {"Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
          "Qwen/Qwen2.5-7B": "d149729398750b98c0af14eb82c78cfe92750796"}
SCORER = "8d6860e16790999af7369e6060bb15186e7378dd6d5dded5a404db1146213785"
FLOWERS = "fdb051c89eabc1c51ab80947251fb7176f64893bbf0641e229028a9490b6d74d"
CALTECH_Q = "0c6eb894fcb2008e4b405dc61599efa8f003be58f259e5ca5c6c650f684e8538"
CALTECH_CLASSES = "0a12263eee5e53253780d7ba9e17c626bf1b0d9fcbae827400299b8c6157f6f6"
FACE_CLASSES = "7cd2b68f0fe0de975ba1e3088302bfdec515658f997f499b67bd7c811e638cdf"
FLOWER_PROBE = ["pink primrose", "hard-leaved pocket orchid"]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def checked(p, expected):
    if digest(p) != expected:
        raise ValueError(f"Unexpected author file: {p}")
    return p


def read(p):
    return json.loads(p.read_bytes())


def write(p, payload):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def verified_matrix(payload, names):
    if set(payload) != set(names):
        raise ValueError("Class name coverage differs from the requested input")
    matrix = np.asarray([payload[n] for n in names], dtype="<f8")
    if (matrix.shape != (len(names), 50, 5) or not np.isfinite(matrix).all()
            or matrix.min() < 0 or matrix.max() > 1
            or not np.allclose(matrix.sum(-1), 1, atol=.005, rtol=0)):
        raise ValueError("Expected a probability matrix with 50 five-option questions per class")
    return matrix


def audit_flowers(root, prediction, output):
    reference = read(checked(root / "questions/flowers/logits/qwen2.5-7b_chat_50-5.json", FLOWERS))
    pred = verified_matrix(read(prediction), FLOWER_PROBE)
    truth = verified_matrix({name: reference[name] for name in FLOWER_PROBE}, FLOWER_PROBE)
    norm = lambda m: m / m.sum(-1, keepdims=True)
    tv = .5 * np.abs(norm(pred) - norm(truth)).sum(-1)
    same = pred.argmax(-1) == truth.argmax(-1)
    median, p90 = np.quantile(tv, [.5, .9])
    audit = {"source_sha256": FLOWERS, "prediction_sha256": digest(prediction),
             "n": same.size, "matching_top1": int(same.sum()), "median_tv": float(median),
             "p90_tv": float(p90), "max_abs_raw_probability_difference": float(np.abs(pred-truth).max()),
             "passes_frozen_check": bool(same.sum() >= 95 and median <= .02 and p90 <= .075),
             "rows": [{"class": name, "question": q+1, "reference_top1": int(truth[c,q].argmax())+1,
                       "prediction_top1": int(pred[c,q].argmax())+1, "tv": float(tv[c,q])}
                      for c,name in enumerate(FLOWER_PROBE) for q in range(50)]}
    write(output, audit)
    print(json.dumps({k:v for k,v in audit.items() if k!="rows"}))


def score(args):
    # All weights are already staged by the cloud worker. Require an explicit source revision.
    if MODELS.get(args.model_id) != args.revision:
        raise ValueError("Unexpected model/revision pair")
    if not args.model_path.is_dir() or not (args.model_path/"config.json").is_file():
        raise FileNotFoundError(args.model_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    root = args.author_root.resolve()
    module_path = checked(root/"lavid/extract/llm_logits.py", SCORER)
    questions = root/f"questions/{args.dataset}/chat_50-5.txt"
    classes = root/f"lavid/data/classes/{args.dataset}_classes.json"
    all_names = read(classes)
    if args.dataset == "caltech101":
        checked(questions, CALTECH_Q)
        checked(classes, CALTECH_CLASSES)
        checked(root/"lavid/data/classes/caltech101_classes_duplicate_face.json", FACE_CLASSES)
        names = all_names
    else:
        if all_names[:2] != FLOWER_PROBE:
            raise ValueError("Flowers calibration identities changed")
        names = FLOWER_PROBE
    if len(names) != (100 if args.dataset=="caltech101" else 2):
        raise ValueError("Wrong number of class prompts")
    import torch
    import transformers
    spec = importlib.util.spec_from_file_location("locked_author_llm_logits", module_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    output=args.out.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        temp = Path(tmp)
        # The only adapted input is the class list; all chat/token/logit operations are upstream.
        class_root = temp/"classes"
        class_root.mkdir()
        write(class_root/f"{args.dataset}_classes.json",names)
        scorer.files = lambda package: temp
        native = temp/"predictions.json"
        scorer.option_number(str(args.model_path.resolve()), args.dataset, str(native),
                             str(questions), batch_size=25, temperature=1., alphabet=False)
        data = read(native)
        matrix = verified_matrix(data,names)
        write(output,{name:data[name] for name in names})
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.model_path.resolve()), padding_side="left")
    digits=[tokenizer.convert_tokens_to_ids(str(v)) for v in range(1,6)]
    if len(set(digits))!=5 or tokenizer.unk_token_id in digits:
        raise ValueError("Five distinct answer tokens required")
    proof = {"dataset":args.dataset,"source_model":args.model_id,"staged_revision":args.revision,
             "runtime_model_path":str(args.model_path.resolve()),"dtype":"torch.bfloat16","batch_size":25,
             "temperature":1,"tokenizer_class":type(tokenizer).__name__,"option_token_ids":digits,
             "torch":torch.__version__,"transformers":transformers.__version__,"cuda":torch.version.cuda,
             "gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None,
             "python_hashseed":os.getenv("PYTHONHASHSEED"),"author_scorer_sha256":SCORER,
             "author_questions_sha256":digest(questions),"author_class_file_sha256":digest(classes),
             "chat_template_sha256":hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
             "json_sha256":digest(output),"numeric_f64_sha256":hashlib.sha256(matrix.tobytes()).hexdigest(),
             "shape":list(matrix.shape)}
    if args.dataset=="caltech101":
        label_names=read(root/"lavid/data/classes/caltech101_classes_duplicate_face.json")
        expanded=np.asarray([data[n] for n in label_names],dtype="<f8")
        assert expanded.shape==(101,50,5) and np.array_equal(expanded[0],expanded[1])
        proof["visual_label_matrix_f64_sha256"]=hashlib.sha256(expanded.tobytes()).hexdigest()
    write(output.with_suffix(".provenance.json"),proof)
    if args.dataset == "flowers":
        audit_flowers(root,output,output.with_suffix(".audit.json"))
    else:
        print(json.dumps(proof))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="command",required=True)
    p=subs.add_parser("score")
    p.add_argument("--author-root",type=Path,required=True)
    p.add_argument("--dataset",choices=["flowers","caltech101"],required=True)
    p.add_argument("--model-id",choices=MODELS,required=True)
    p.add_argument("--revision",required=True)
    p.add_argument("--model-path",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p=subs.add_parser("audit-flowers")
    p.add_argument("--author-root",type=Path,required=True)
    p.add_argument("--prediction",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    if a.command=="score":
        score(a)
    else:
        audit_flowers(a.author_root,a.prediction,a.out)
