# RVC Implementation Guide: Architecture and Training Overview

Retrieval-based Voice Conversion (RVC) achieves high-quality voice conversion with minimal training data (≥10 minutes) by combining self-supervised learning for content extraction with a novel FAISS-based retrieval mechanism that reduces timbre leakage. The system operates entirely on speech-to-speech conversion **without requiring text transcripts**, using transfer learning from pretrained models to enable rapid fine-tuning on target speakers.

The key innovation is the retrieval module, which searches training features during inference to replace residual source speaker characteristics with target speaker features. ContentVec features retain ~5% source speaker information despite speaker disentanglement; the retrieval mechanism explicitly replaces this contamination with actual target speaker examples, reducing timbre leakage.

## System Components Overview

### Pretrained Components (Never Trained)
- **ContentVec/HuBERT**: Self-supervised speech encoder that extracts speaker-agnostic content features (256-dim for v1, 768-dim for v2)
- **RMVPE**: CNN-based pitch extractor that outputs fundamental frequency (F0) values
- **Base Generator/Discriminator Weights**: Pretrained on large datasets (VCTK/LibriSpeech), provide initialization for fine-tuning

### Trained Components (Speaker-Specific)
- **Generator (VITS-based)**: Learns to synthesize target speaker's voice from content+pitch features
  - Prior encoder: Transformer that processes fused features and generates latent distributions
  - Normalizing flows: 4 affine coupling layers that enhance distribution expressiveness
  - NSF-HiFiGAN decoder: Neural source-filter vocoder that synthesizes waveforms
- **Discriminator**: Multi-period discriminator that distinguishes real vs generated audio

### Post-Training Artifacts
- **FAISS Index**: Approximate nearest-neighbor search structure containing all target speaker ContentVec features from training data
- Used during inference for retrieval mechanism

## Training Pipeline (Step-by-Step)

### Phase 1: Data Preprocessing (Can be cached/done offline)

**Step 1.1: Audio Collection and Initial Processing**
- **Input**: Raw audio files of target speaker (10-50 minutes minimum, any format)
- **Process**: 
  - Load audio using FFmpeg, convert to float32 normalized [-1, 1]
  - Apply basic denoising (scipy filtfilt smoothing)
- **Output**: Normalized audio ready for segmentation
- **Why**: Standardize input format regardless of source

**Step 1.2: Audio Segmentation**
- **Input**: Normalized audio from previous step
- **Process**:
  - Silence-based splitting (breaks at >5 second silence gaps)
  - Fixed-window chunking (4-second segments with 0.3s overlap)
  - Volume normalization per segment
  - Downsample all segments to 16 kHz (required for ContentVec)
- **Output**: Collection of 16 kHz audio segments
- **Why**: Separate distinct utterances while maintaining continuity; 16 kHz is ContentVec's required sample rate
- **Note**: This can be cached - no need to reprocess if retraining

**Step 1.3: Feature Extraction (Parallel Paths)**

**Path A - Pitch Extraction:**
- **Input**: 16 kHz audio segments
- **Process**: Run RMVPE (pretrained, frozen) on each segment
- **Output**: 
  - Continuous F0 values (for vocoder synthesis)
  - Discretized F0 values (integers 1-255 for embedding layers)
- **Why**: Capture prosody/intonation separately from content
- **Note**: Can be cached; RMVPE is never trained

**Path B - Content Feature Extraction:**
- **Input**: 16 kHz audio segments
- **Process**: Run ContentVec/HuBERT (pretrained, frozen) on each segment
  - V1: Extract 256-dim features from 9 layers + final projection
  - V2: Extract 768-dim features from all 12 layers
- **Output**: Content feature vectors for each segment (shape: [N_frames, 256/768])
- **Why**: Get speaker-independent representation of speech content
- **Note**: Can be cached; ContentVec is never trained

### Phase 2: Model Training (Iterative)

**Step 2.1: Initialize Models**
- **Process**:
  - Load pretrained generator weights (f0G40k.pth for v1)
  - Load pretrained discriminator weights (f0D40k.pth for v1)
  - Initialize optimizer (AdamW with β1=0.8, β2=0.999, learning rate 5e-4)
- **Why**: Transfer learning - start from universal speech patterns rather than random initialization

**Step 2.2: Training Loop (300-500 epochs typical)**

For each batch of training data:

**Generator Forward Pass:**
- **Input**: ContentVec features + discrete F0 tokens + continuous F0
- **Process**:
  1. Prior encoder (transformer) processes concatenated features
  2. Generates latent distribution z_p ~ p(z|c) conditioned on features
  3. Sample from distribution (during training) or use mean (during eval)
  4. Normalizing flows transform the latent
  5. NSF-HiFiGAN decoder synthesizes waveform from latent + continuous F0
- **Output**: Generated audio waveform

**Discriminator Forward Pass:**
- **Input**: Both real (target speaker) and generated audio
- **Process**: Multi-period discriminators analyze at different temporal scales
- **Output**: Real/fake classifications and intermediate feature maps

**Loss Calculation:**
- **Generator losses**:
  - Adversarial loss: Fool the discriminator
  - Feature matching loss (L1): Match discriminator intermediate activations
  - Mel-spectrogram loss (heavily weighted, α=45): Ensure spectral similarity
  - KL divergence: Regularize latent space distribution
- **Discriminator losses**:
  - Binary classification: Distinguish real vs fake audio

**Backward Pass:**
- Update discriminator parameters
- Update generator parameters
- Apply gradient clipping (norm 1.0)
- Exponential learning rate decay

**Step 2.3: Periodic Evaluation**
- **Process**: At intervals (e.g., every 50 epochs), save model checkpoint and test on validation audio
- **Why**: Monitor training progress and identify best checkpoint (latest isn't always best)

### Phase 3: Post-Training Index Construction

**Step 3.1: Build FAISS Index**
- **Input**: All ContentVec features extracted from training audio (from Step 1.3 Path B)
- **Process**:
  - Concatenate all feature vectors into single array [N_total_frames, 256/768]
  - Create IVF (Inverted File) index structure:
    - Number of clusters: n_ivf = N // 39
    - Number of search probes: n_probe = int(n_ivf^0.3)
  - Build approximate nearest neighbor search structure using L2 distance
- **Output**: FAISS index file enabling fast similarity search
- **Why**: Enable retrieval mechanism during inference - find similar target speaker features to replace source speaker contamination

## Training Outputs

After training completes, you have:

1. **Fine-tuned Generator Model**: Speaker-specific weights that synthesize target voice
   - Can be saved at multiple checkpoints during training
   - Select best based on perceptual quality evaluation

2. **Fine-tuned Discriminator Model**: (Optional to keep, only needed if resuming training)

3. **FAISS Index**: Searchable database of target speaker's content features
   - Required for inference
   - Contains all ContentVec features from training audio

4. **Preprocessed Features** (can be cached for future retraining):
   - Segmented audio files
   - Extracted pitch information
   - Extracted ContentVec features

## Inference Pipeline (Step-by-Step)

**Step 1: Source Audio Preprocessing**
- **Input**: Source audio (any speaker, any format)
- **Process**:
  - Load and resample to 16 kHz
  - Optional: segment into chunks for streaming/real-time processing
- **Output**: 16 kHz source audio ready for feature extraction

**Step 2: Feature Extraction from Source**
- **Input**: 16 kHz source audio
- **Process**:
  - **ContentVec** (pretrained, frozen): Extract content features with ~5% source speaker contamination
  - **RMVPE** (pretrained, frozen): Extract continuous and discrete F0
- **Output**: 
  - ContentVec features [N_frames, 256/768]
  - Continuous F0 values
  - Discrete F0 tokens (1-255)

**Step 3: Retrieval and Feature Fusion**
- **Input**: Source ContentVec features from Step 2
- **Process**:
  - Query FAISS index with source features
  - Retrieve top-k (typically k=8) most similar target speaker features
  - Blend features: `fused = α × retrieved_target + (1-α) × source`
    - Typical α = 0.3 (30% target, 70% source)
    - Higher α = more target speaker similarity, potentially less natural
  - Optional: Apply "protect mode" - adjust fusion based on F0 values to preserve breath sounds
- **Output**: Fused ContentVec features with reduced source speaker contamination
- **Why**: Replace the ~5% residual source speaker information with actual target speaker characteristics

**Step 4: Generate Target Audio**
- **Input**: 
  - Fused ContentVec features from Step 3
  - Discrete F0 tokens for conditioning
  - Continuous F0 for vocoder synthesis
- **Process**:
  1. Prior encoder (transformer) generates latent distribution conditioned on fused features + discrete F0
  2. Sample latent variable z_p ~ p(z|c)
  3. Inverse normalizing flow transforms: z = f⁻¹(z_p)
  4. NSF-HiFiGAN decoder synthesizes waveform:
     - Source module generates excitation from continuous F0 (harmonic + noise)
     - Filter module applies learned spectral characteristics via residual blocks
     - Multi-scale upsampling produces final waveform
- **Output**: Audio waveform in target speaker's voice

**Step 5: Post-Processing**
- **Process**: 
  - Resample to desired output rate (32k/40k/48k Hz)
  - Optional: Apply final normalization
- **Output**: Final converted audio

## Key Architecture Details

### ContentVec/HuBERT (Pretrained Content Encoder)

Architecture processes raw PCM audio through:
- 7 temporal convolutional blocks → 512-dim features
- Projection to 768 dimensions
- 12 transformer layers (768 hidden, 12 attention heads)
- Output: 256-dim (v1, using 9 layers) or 768-dim (v2, full 12 layers)

**Disentanglement mechanisms** reduce speaker information from 73.7% (standard HuBERT) to 37.7%:
- Offline clustering with voice conversion augmentation
- Contrastive loss with speaker-invariant transforms
- Speaker conditioning in predictor network

Despite optimization, ~5% residual speaker information remains, motivating retrieval.

### RMVPE (Pretrained Pitch Extractor)

U-Net-based CNN that extracts fundamental frequency directly from audio:
- Works with polyphonic music without source separation
- More accurate and faster than alternatives (Crepe, Harvest, PM)
- Outputs both continuous F0 (for synthesis) and discrete F0 (for conditioning)

### VITS-Based Generator (Trained)

**Prior Encoder:**
- Transformer with relative positional embeddings
- Processes concatenated: fused ContentVec features + discrete F0 embeddings
- Outputs distribution parameters: mean μ and log-variance log(σ²)

**Normalizing Flows:**
- 4 affine coupling layers for distribution transformation
- Each layer contains 4 WaveNet residual blocks
- Enables expressive prior distributions matching complex speech patterns

**NSF-HiFiGAN Decoder:**
- **Source module**: Generates excitation signals from F0
  - Harmonic component: periodic waveforms from F0
  - Noise component: aperiodic signals for consonants/breathiness
- **Filter module**: Multi-scale residual blocks across 4 upsampling stages
  - Applies learned spectral characteristics to excitation
  - Typical upsampling rates: [8, 8, 2, 2] with kernels [16, 16, 4, 4]
- Direct waveform generation without intermediate mel-spectrograms

### Multi-Period Discriminator (Trained)

Analyzes audio at different temporal scales:
- V1: 5 periods [2, 3, 5, 7, 11]
- V2: 8 periods [2, 3, 5, 7, 11, 17, 23, 37]

Each discriminator provides:
- Real/fake classification
- Intermediate feature maps for feature matching loss

### Retrieval Mechanism (FAISS Index)

**Structure:**
- Inverted File (IVF) index partitions feature space via clustering
- Approximate nearest neighbor search using L2 distance
- Trade-off between speed and accuracy via n_ivf and n_probe parameters

**Retrieval Process:**
1. Query with source ContentVec features
2. Search n_probe clusters (fast coarse search)
3. Return top-k exact matches from relevant clusters
4. Linear fusion with configurable index_rate α

**Why It Works:**
- ContentVec aims for speaker independence but retains ~5% speaker info
- Retrieval explicitly replaces source speaker characteristics
- Uses actual target speaker examples rather than only learned mappings
- Reduces "timbre leakage" where source voice bleeds through

## Training Configuration

### Required Hyperparameters

**Model Architecture:**
- Version: v1 (256-dim, faster) or v2 (768-dim, higher quality)
- Sample rate: 32k/40k/48k Hz for output (always 16k internal for ContentVec)
- Pitch guidance: Enable for singing, disable for speech-only (lighter models)

**Training:**
- Batch size: 7-32 depending on GPU memory (4-6GB for v1, 6-8GB for v2)
- Epochs: 300-500 typical
- Learning rate: 5e-4 with exponential decay
- Optimizer: AdamW (β1=0.8, β2=0.999, ε=1e-9)
- Gradient clipping: Norm 1.0

**Loss Weights:**
- Mel-spectrogram loss: 45.0 (heavily weighted for spectral accuracy)
- Feature matching loss: 1.0
- KL divergence: Standard VITS weight

### Data Requirements

**Quantity:**
- Minimum: 10 minutes of clean audio
- Recommended: 10-50 minutes
- Quality matters more than quantity

**Quality Criteria:**
- Low background noise
- Consistent recording conditions
- Consistent timbre (single speaker only)
- Emotional/prosodic variety helps but not required

**No Requirements:**
- Text transcripts
- Phonetic annotations
- Parallel recordings
- Specific phonetic coverage

## Version Differences

### RVC v1
- 256-dimensional ContentVec features (9 layers + projection)
- 5 discriminator periods
- Model size: ~55MB
- Faster training and inference
- Suitable for real-time applications

### RVC v2
- 768-dimensional ContentVec features (full 12 layers)
- 8 discriminator periods (adds [17, 23, 37])
- Model size: ~110MB
- Higher quality, especially for singing
- More computational cost

### RVC v3 (Announced)
- Larger parameters and expanded training data
- Improved quality with less target audio required
- Maintains v2 inference speeds

## Performance Characteristics

**Training Time:**
- 30-120 minutes on RTX 3090 for 300 epochs
- 10-30 minutes of training audio typical
- Scales with batch size (larger = faster but more VRAM)

**Inference Speed:**
- Real-time capable: <90ms latency with optimization
- GPU: 4-210x faster than real-time (hardware dependent)
- CPU: 10-100x slower than GPU but viable for offline processing

**Quality:**
- Near-indistinguishable from target speaker with clean training data
- Handles multiple languages via ContentVec's multilingual pretraining
- Works for both speech and singing (with pitch guidance enabled)

## Practical Considerations

### What Can Be Cached/Preprocessed

**During Training:**
- Segmented audio files (Step 1.2)
- Extracted ContentVec features (Step 1.3 Path B)
- Extracted pitch information (Step 1.3 Path A)

These never change unless you modify the training data, so compute once and reuse.

**What Must Be Recomputed:**
- Generator/discriminator forward passes (every training step)
- Loss calculations and gradients (every training step)
- FAISS index (after training, if features or indexing parameters change)

### Training Best Practices

**Data Quality:**
- 10 minutes of clean audio > 60 minutes of noisy audio
- Consistent recording environment matters more than phonetic diversity
- Avoid multiple speakers in training data

**Monitoring:**
- Generator loss should decrease from ~1.5 to <0.5
- Discriminator loss should balance, not dominate
- Test inference at multiple checkpoints (latest ≠ best)

**Common Issues:**
- **Tone leakage** (source bleeding through): Increase index_rate α or enable retrieval
- **Robotic output**: Need longer training or more varied training data
- **Training instability**: Discriminator dominating; adjust learning rates or update ratios

### Inference Parameters

**Index Rate (α):**
- Default: 0.3 (30% retrieved target, 70% source)
- Higher: More target speaker similarity, may reduce naturalness
- Lower: Preserves more source characteristics

**Protect Mode:**
- Value: 0.3-0.5 typical
- Adjusts fusion based on F0 values
- Preserves breath sounds and voiceless consonants
- When F0 < 1 (silence), increases original voice reflection

**Pitch Transpose:**
- Semitone adjustment for gender conversion
- Positive: Higher pitch, negative: Lower pitch

## Implementation Deployment

### Model Export Options

**ONNX:**
- Cross-platform compatibility
- Deploy on CPU/GPU/mobile
- Slight performance overhead vs native PyTorch

**TensorRT:**
- GPU-specific optimization
- Significant inference speedup
- Best for production GPU deployments

**Quantization:**
- FP16: 2x speedup, negligible quality loss
- INT8: 4x speedup, slight quality degradation
- Reduces model size proportionally

### Real-Time Considerations

**Streaming Processing:**
- Process 0.2-0.5 second chunks
- Minimal latency accumulation
- Requires careful buffer management

**FAISS Optimization:**
- IVF-PQ variant for faster CPU search
- GPU index for maximum throughput
- Trade speed vs accuracy via index parameters

**Hardware Requirements:**
- Training: 6GB VRAM (v1), 8GB VRAM (v2)
- Inference: More modest, CPU viable for offline
- Real-time: GPU recommended for <100ms latency
