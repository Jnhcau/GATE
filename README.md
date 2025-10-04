# GATE

This repository contains a PyTorch implementation of a GATE Model. It provides functionalities for training the model and making predictions using a command-line interface.

## <img src="https://badgen.net/badge/Python/3.10+/blue?" alt="Python" width="100"/><img src="https://badgen.net/badge/Pytorch/2.10/orange?" alt="Pytorch" width="100"/><img src="https://badgen.net/badge/Cuda/12.1/pink?" alt="Cuda" width="70"/>

## Features

- Multi-channel convolutional network for extracting local co-expression patterns.
- Gated attention mechanism for dynamically weighting key features.
- Flexible normalization options: batch, group, layer, instance, or none.
- Supports L1 regularization on gate weights for feature selection.
- Command-line interface for training and prediction.
- Reproducible cross-validation with configurable repeats.

## Installation

1. **Clone the repository (if not already done):**

   ```bash
   git clone <(https://github.com/Jnhcau/GATE))>
   cd GATE
   ```

2. **Install dependencies:**

   It is highly recommended to use a virtual environment.

   ```bash
   conda env create -f environment.yml
   conda activate gate_env
   ```

## Usage

The `GATE.py` script can be run in two modes: `train` and `predict`.

### Training the Model

To train the model, use the `train` mode. You can specify various model and training parameters.

---

#### Simple Example

```bash
python GATE.py --mode train \
			   --data_path tra_data.csv \
               --pheno_path pheno_data.csv
```

By default, this will train for 500 epochs with batch size 32, hidden size 64, 8 CNN channels, and instance normalization. Results will be saved to `results.txt` and the model to `GATE_model.pth`.

#### Advanced Example

You can customize multiple parameters to optimize training. For example:

```bash
python GATE.py --mode train \
               --data_path tra_data.csv \
               --pheno_path pheno_data.csv \
               --trait_column Plantheight \
               --hidden_size 128 \
               --kernel_size 5 \
               --hidden_dropout_prob 0.2 \
               --gate_dropout_prob 0.4 \
               --norm_type batch \
               --lr 0.0005 \
               --batch_size 64 \
               --lambda_l1 0.001 \
               --repeat 10 \
               --model_path models/GATE_plantheight.pth \
               --result_txt results/GATE_plantheight.txt
```

This example:

- Increases model capacity with larger hidden layers and more CNN channels.
- Uses larger convolutional kernels to capture broader local patterns.
- Applies stronger dropout for regularization.
- Switches to batch normalization.
- Repeats cross-validation 10 times for more robust evaluation.
- Saves the model and results to custom paths.

**Common Training Arguments:**

*   `--mode`: `train` (default)
*   `--hidden_size`: Hidden layer size (default: 64)
*   `--kernel_size`: Kernel size for CNN (default: 3)
*   `--hidden_dropout_prob`: Dropout probability for hidden layers (default: 0.1)
*   `--gate_dropout_prob`: Dropout probability for gate mechanism (default: 0.1)
*   `--num_channels`: Number of channels in CNN (default: 8)
*   `--norm_type`: Normalization type ('batch', 'group', 'layer', 'instance', 'none'; default: 'instance')
*   `--epochs`: Number of training epochs (default: 500)
*   `--lr`: Learning rate (default: 0.001)
*   `--batch_size`: Batch size for training (default: 32)
*   `--model_path`: Path to save the trained model (default: `GATE_model.pth`)
*   `--data_path`: **(Required)** Path to input transcriptome data CSV file
*   `--pheno_path`: **(Required)** Path to phenotype data CSV file
*   `--trait_column`: Column index (0-based) or name of the trait to predict from pheno_path (default: '0', i.e., first column)
*   `--lambda_l1`: L1 regularization strength for gate weights (default: 0.0005)
*   `--repeat`: Number of times to repeat the cross-validation (default: 5)
*   `--result_txt`: Path to save training results (default: `results.txt`)

### Prediction with the Model

Once the model is trained, you can use the `predict` mode to make phenotype predictions on new transcriptomic data.

---

#### Example

```bash
python GATE.py --mode predict \
               --model_path GATE_model.pth \
               --data_path new_data.csv
```

**Common Prediction Arguments:**

*   `--mode`: `predict`
*   `--model_path`: Path to the trained model file (default: `GATE_model.pth`)
*   `--data_path`: Path to the input transcriptome data CSV file for prediction (same format as training data)

## Important Notes

*   **Data Format:** Input data and labels are expected to be CSV files. 
*   **GPU Usage:** The script automatically detects and uses a GPU (CUDA) if available; otherwise, it falls back to CPU.
*   **Feature Importance:** After training, the script will save a CSV file with gene importance scores, derived from the model's gate weights, to the same directory as `result_txt`.

Feel free to contribute or suggest improvements!

