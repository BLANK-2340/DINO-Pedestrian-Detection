# Pedestrian Detection using DINO (Detection Transformer)

This repository implements and analyzes pedestrian detection using the DINO (Detection Transformer) model, including evaluation of the pre-trained model and fine-tuning experiments.

## Project Structure

```
├── DINO/                  # DINO model implementation
├── validation/            # Validation dataset
├── training/             # Training dataset
├── DINO_output/          # Original DINO evaluation results
├── DINO_finetune/        # Fine-tuning outputs
│   ├── models/          # Saved model checkpoints
│   ├── visualizations/  # Training visualizations
│   ├── attention_maps/  # Attention visualizations
│   └── metrics/         # Training metrics
└── DINO_finetune_output/ # Fine-tuned model evaluation results
```

## Model Details

- Base Model: DINO-4scale
- Backbone: ResNet-50
- Original Performance: mAP@0.5 = 49.0 (on COCO dataset)

## Installation & Setup
set up the project structure and then run the DINO_assignment Jupiter notebook step-by-step 
## Experiments & Results

### 1. Pre-trained DINO Evaluation

Results on validation set:
- AP@0.5: 0.766
- AP@0.75: 0.443
- AP (mean): 0.430
- AR@100: 0.492

Key statistics:
- Average Detection Ratio: 0.949
- Perfect Detections (±5%): 37.5%
- Over-detections: 17.5%
- Under-detections: 45.0%

### 2. Fine-tuning Experiments

Training configuration:
- Epochs: 15
- Batch size: 2 (effective: 8 with gradient accumulation)
- Base learning rate: 1e-4
- Optimizer: AdamW
- Loss components: Classification, Bounding Box, GIoU

Training improvements:
- Training Loss: 82.35% improvement
- Validation Loss: 73.44% improvement
- Component-wise improvements:
  - Class Loss: 82.14% (train) / 68.15% (val)
  - Bbox Loss: 83.09% (train) / 75.27% (val)
  - GIoU Loss: 79.14% (train) / 68.70% (val)

### 3. Fine-tuned Model Performance

The fine-tuned model showed significant improvements in:
- Detection accuracy
- Reduced false positives
- Better localization
- Improved handling of dense pedestrian scenes

## Key Techniques Used

1. Data Augmentation:
   - Random brightness and contrast
   - Gamma adjustments
   - Gaussian and ISO noise
   - Normalization
   - Aspect ratio preservation

2. Training Optimizations:
   - Gradient accumulation (accumulation steps: 4)
   - Learning rate warmup
   - EMA model averaging
   - Weight decay
   - Gradient clipping

3. Loss Functions:
   - Classification loss (CE)
   - Bounding box regression (L1)
   - GIoU loss
   - Custom weighted combination

4. Monitoring & Visualization:
   - Multi-scale attention maps
   - Loss component analysis
   - Performance metrics tracking
   - Deformable attention visualization

## Running the Code

### 1. Initial Evaluation
```python
python evaluate_dino.py --config configs/eval_config.py
```

### 2. Fine-tuning
```python
python train_dino.py --config configs/train_config.py
```

### 3. Fine-tuned Model Evaluation
```python
python evaluate_finetune.py --config configs/finetune_eval_config.py
```

## Advanced Features

1. Deformable Attention:
   - Multi-scale feature processing
   - Adaptive sampling points
   - Cross-attention mechanisms

2. Performance Analysis:
   - Complexity-based evaluation
   - Per-image detection analysis
   - Attention map visualization

3. Training Monitoring:
   - Real-time loss tracking
   - Component-wise analysis
   - Learning rate scheduling visualization

## Results Visualization

The code generates various visualizations:
- Loss curves (training/validation)
- Attention maps
- Detection visualizations
- Performance metrics plots
- Complexity analysis charts

## Notes

- The model performs best on standard pedestrian sizes
- Performance degrades in highly crowded scenes
- Fine-tuning significantly improves detection quality
- Best results achieved with proper image preprocessing

## License

[Specify your license]

## Acknowledgments

- DINO implementation: [Citation]
- Dataset: [Source]
