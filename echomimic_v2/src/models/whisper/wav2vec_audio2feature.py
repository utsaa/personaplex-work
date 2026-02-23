import os
import torch
from torch import nn
import numpy as np
try:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    import librosa
except ImportError:
    print("Wav2Vec2 dependencies not found. Please install transformers and librosa.")

class Wav2Vec2Audio2Feature():
    def __init__(self, 
                 model_path="facebook/wav2vec2-base-960h",
                 device="cuda",
                 adapter_path=None):
        self.device = device
        # Load processor and model
        if model_path.endswith(".bin"):
            model_dir = os.path.dirname(model_path)
            if os.path.exists(os.path.join(model_dir, "config.json")):
                self.model = Wav2Vec2Model.from_pretrained(model_dir).to(device)
                self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
            else:
                print(f"Warning: config.json not found in {model_dir}. Loading facebook/wav2vec2-base-960h and overriding weights.")
                self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        else:
            self.model = Wav2Vec2Model.from_pretrained(model_path).to(device)
            self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        
        self.model.eval()
        
        # Projection layer to map 768 to 384
        self.projection = nn.Linear(768, 384).to(device)
        
        if adapter_path and os.path.exists(adapter_path):
            self.load_adapter(adapter_path)
        else:
            self.init_pseudo_identity()

    def init_pseudo_identity(self):
        """Initialize with a pseudo-identity matrix and a massive scale boost."""
        print("  initialising projection with Pseudo-Identity + 50x Scale (Emergency Sync)...")
        self.feature_scale = 50.0 # Heuristic boost to match Whisper's dynamic range
        with torch.no_grad():
            self.projection.weight.zero_()
            for i in range(384):
                self.projection.weight[i, i % 768] = 1.0
            self.projection.bias.zero_()

    def load_adapter(self, adapter_path):
        self.feature_scale = 1.0 # Adapters are usually already scaled
        print(f"  loading trained adapter from {adapter_path} ...")
        try:
            adapter_weights = torch.load(adapter_path, map_location=self.device)
            
            # Detect key names (handling MuseTalk vs our format)
            weight_key = None
            bias_key = None
            
            if "weight" in adapter_weights and "bias" in adapter_weights:
                weight_key = "weight"
                bias_key = "bias"
            else:
                for k in adapter_weights.keys():
                    if "weight" in k: weight_key = k
                    if "bias" in k: bias_key = k
            
            if weight_key is None:
                print(f"  [ERROR] Could not find weight keys in {adapter_path}")
                return

            w = adapter_weights[weight_key]
            b = adapter_weights[bias_key] if bias_key else None
            
            print(f"  [INFO] Adapter weight shape: {w.shape}")
            
            # Dimension alignment (Target is 384 x 768)
            # MuseTalk might be 768x768 or 1024x768
            target_out = self.projection.weight.shape[0] # 384
            target_in = self.projection.weight.shape[1] # 768
            
            with torch.no_grad():
                # Handle input dimension (usually 768)
                if w.shape[1] != target_in:
                    print(f"  [WARN] Input dimension mismatch: {w.shape[1]} vs {target_in}. Interpolating...")
                    w = torch.nn.functional.interpolate(w.unsqueeze(0), size=(w.shape[0], target_in), mode='bilinear').squeeze(0)
                
                # Handle output dimension (Target 384)
                if w.shape[0] != target_out:
                    print(f"  [WARN] Output dimension mismatch: {w.shape[0]} vs {target_out}. Slicing/Padding...")
                    if w.shape[0] > target_out:
                        w = w[:target_out, :]
                        if b is not None: b = b[:target_out]
                    else:
                        # This should rarely happen for Wav2Vec2 adapters
                        new_w = torch.zeros((target_out, target_in), device=self.device)
                        new_w[:w.shape[0], :] = w
                        w = new_w
                        if b is not None:
                            new_b = torch.zeros(target_out, device=self.device)
                            new_b[:b.shape[0]] = b
                            b = new_b
                
                self.projection.weight.copy_(w)
                if b is not None:
                    self.projection.bias.copy_(b)
                print("  [SUCCESS] Trained weights injected into projection layer.")
                
        except Exception as e:
            print(f"  [ERROR] Failed to load adapter: {e}")
            self.init_pseudo_identity()

    def audio2feat(self, audio_data):
        if isinstance(audio_data, str):
            speech, sr = librosa.load(audio_data, sr=16000)
        else:
            speech = audio_data
            sr = 16000
            
        input_values = self.processor(speech, sampling_rate=sr, return_tensors="pt").input_values.to(self.device)
        
        with torch.no_grad():
            hidden_states = self.model(input_values).last_hidden_state
            projected_features = self.projection(hidden_states) * self.feature_scale
            
        return projected_features.squeeze(0).cpu().numpy()

    def get_sliced_feature(self, feature_array, vid_idx, audio_feat_length=[2,2], fps=25):
        length = len(feature_array)
        selected_feature = []
        selected_idx = []
        
        center_idx = int(vid_idx*50/fps) 
        left_idx = center_idx-audio_feat_length[0]*2
        right_idx = center_idx + (audio_feat_length[1]+1)*2
        
        for idx in range(left_idx,right_idx):
            idx = max(0, idx)
            idx = min(length-1, idx)
            x = feature_array[idx]
            selected_feature.append(x)
            selected_idx.append(idx)
        
        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)
        return selected_feature, selected_idx

    def feature2chunks(self, feature_array, fps, audio_feat_length=[2,2]):
        whisper_chunks = []
        whisper_idx_multiplier = 50./fps 
        i = 0
        while 1:
            start_idx = int(i * whisper_idx_multiplier)
            selected_feature, selected_idx = self.get_sliced_feature(feature_array=feature_array, vid_idx=i, audio_feat_length=audio_feat_length, fps=fps)
            whisper_chunks.append(selected_feature)
            i += 1
            if start_idx > len(feature_array):
                break
        return np.array(whisper_chunks)
