"""Verify exported ONNX matches selected PyTorch model on held-out LFW faces."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import numpy as np
import onnxruntime as ort
import torch
from scripts.train_face_rejection import FaceNegatives,validate_negatives
from model import build_model
from preprocess import write_json

def main():
    torch.set_num_threads(2)
    weights=ROOT/'runs/resnet18_face_oe_v1/best.pt'
    c=torch.load(weights,map_location='cpu',weights_only=True)
    model=build_model(**c['model_config']).eval();model.load_state_dict(c['model_state'])
    root=ROOT/'hard_negatives/lfw';m=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    validate_negatives(root,m)
    ds=FaceNegatives(root,m,'test',c['preprocess'],c['normalization'])
    session=ort.InferenceSession(str(ROOT/'docs/assets/naifamily.onnx'),providers=['CPUExecutionProvider'])
    expected=[];actual=[]
    for start in range(0,len(ds),16):
        x=torch.stack([ds[i] for i in range(start,min(start+16,len(ds)))])
        with torch.inference_mode():expected.append(model(x).numpy())
        actual.append(session.run(['logits'],{'images':x.numpy()})[0])
    a=np.concatenate(actual);e=np.concatenate(expected)
    np.testing.assert_allclose(a,e,rtol=1e-4,atol=1e-4)
    config=json.loads((ROOT/'docs/model-config.json').read_text(encoding='utf8'));ratio=config['minTopScoreRatio']
    def accepted(logits):
        p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        top=np.sort(p,axis=1)[:,-2:];return top[:,1]>ratio*top[:,0]
    np.testing.assert_array_equal(a.argmax(1),e.argmax(1))
    np.testing.assert_array_equal(accepted(a),accepted(e))
    result=dict(total=len(ds),max_logit_error=float(np.abs(a-e).max()),same_top1=True,same_rejections=True,
                onnx_false_accepted=int(accepted(a).sum()),ratio=ratio)
    write_json(ROOT/'artifacts/face_onnx_verification.json',result);print(json.dumps(result))

if __name__=='__main__':main()
