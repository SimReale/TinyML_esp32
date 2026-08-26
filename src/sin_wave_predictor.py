"""
Train and Export Script for TinyML ESP32
Implements sine wave approximation MLP and exports to TFLite and ESPDL formats
"""

import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
from torch.utils.data import DataLoader
import tf2onnx
import onnx
import onnxsim
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.executor import TorchExecutor
import matplotlib.pyplot as plt
import random

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
WORKING_DIR = Path.cwd()
print(f"Working directory: {WORKING_DIR}")
MODELS_DIR = WORKING_DIR.joinpath('models')
MEDIA_DIR = WORKING_DIR.joinpath('media')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
keras.utils.set_random_seed(42)



def generate_sin_wave_data(num_samples=10000, noise_level=0.05, output_type='numpy'):
    """
    Generate training data for sine wave approximation.
    
    Args:
        num_samples: Number of training samples to generate
        noise_level: Amount of noise to add to the data
        output_type: Type of output data ('numpy' or 'torch')

    Returns:
        x_train: Input data (angles)
        y_train: Output data (sine values)
    """
    # Set dimensions to (1, ) for compatibility with MLP input
    x = np.random.uniform(0, 2*np.pi, num_samples).astype(np.float32)
    y = np.sin(x) + np.random.normal(0, noise_level, num_samples).astype(np.float32)
    x = x.reshape(-1, 1)  # Reshape to (num_samples, 1)
    y = y.reshape(-1, 1)  # Reshape to (num_samples, 1)
    
    if output_type == 'torch':
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
    return x, y


def MLP_model(input_shape=(1,)):
    """
    Create Multi-Layer Perceptron for sine wave approximation.
    Topology: input → 32 → 64 → 128 → output
    
    Args:
        input_shape: Shape of input data
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=input_shape, name='hidden_1'),
        layers.Dense(64, activation='relu', name='hidden_2'),
        layers.Dense(128, activation='relu', name='hidden_3'),
        layers.Dense(1, activation='linear', name='output')
    ])
    
    return model


def compile_and_train_model(model, x_train, y_train, epochs=10, batch_size=32, validation_split=0.0):
    """
    Compile and train the MLP model.
    
    Args:
        model: Keras model to train
        x_train: Training input data
        y_train: Training output data
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
    
    Returns:
        Training history
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nTraining model...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    
    return history


def create_representative_dataset(x_train, num_samples=100):
    """
    Create a representative dataset generator for quantization calibration.
    
    Args:
        x_train: Training data to sample from
        num_samples: Number of samples to use for calibration
    
    Returns:
        Generator function for representative dataset
    """
    def representative_dataset_gen():
        indices = np.random.choice(len(x_train), num_samples, replace=False)
        for idx in indices:
            yield [np.array([x_train[idx]], dtype=np.float32)]
    
    return representative_dataset_gen


def export_to_tflite(model, x_train, output_path=MODELS_DIR.joinpath('sin_wave_model.tflite')):
    """
    Export and quantize model to TFLite format for esp-tflite-micro.
    
    Args:
        model: Trained Keras model
        x_train: Training data for calibration
        output_path: Path to save the TFLite model
    """
    print("\n=== Exporting to TFLite format ===")
    
    # Initialize converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable default optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Force int8 quantization for inputs and outputs
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Provide representative dataset for calibration
    converter.representative_dataset = create_representative_dataset(x_train)
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")

    return tflite_model


def export_to_onnx(model, output_path=MODELS_DIR.joinpath('sin_wave_model.onnx'), opset=13):
    """
    Export Keras model to ONNX format.
    
    Args:
        model: Trained Keras model
        output_path: Path to save the ONNX model
        opset: ONNX opset version
    
    Returns:
        Path to saved ONNX model
    """
    print("\n=== Exporting to ONNX format ===")
    
    # Create concrete function from model
    # ESP-DL deployment examples use a fixed batch size of 1.
    input_spec = tf.TensorSpec(shape=(1, 1), dtype=tf.float32, name='input')

    @tf.function(input_signature=[input_spec])
    def model_func(x):
        return model(x)
    
    # Convert from function
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_model, _ = tf2onnx.convert.from_function(
        model_func,
        input_signature=[input_spec],
        opset=opset,
        output_path=output_path
    )
    
    print(f"ONNX model saved to: {output_path}")
    
    return onnx_model


def evaluate_quantized_model(float_model, quant_ppq_graph, x_test, y_test, device='cpu'):
    """
    Evaluate quantized PPQ graph and compare against float Keras model.

    Args:
        float_model: Trained float Keras model
        quant_ppq_graph: Quantized PPQ graph returned by espdl_quantize_onnx
        x_test: Test input data
        y_test: Test output data
        device: Execution device for PPQ TorchExecutor
    """
    print("\n=== Quantized Model Evaluation (Float vs Quantized) ===")

    # Float baseline predictions
    float_pred = float_model.predict(x_test, verbose=0).reshape(-1)

    # Quantized graph predictions through PPQ executor
    executor = TorchExecutor(graph=quant_ppq_graph, device=device)
    quant_pred = []
    with torch.no_grad():
        for x in x_test:
            sample = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=device)
            output = executor.forward(inputs=sample)[0]
            quant_pred.append(float(output.detach().cpu().numpy().reshape(-1)[0]))

    y_true = y_test.reshape(-1)
    quant_pred = np.array(quant_pred, dtype=np.float32)

    float_mse = float(np.mean((float_pred - y_true) ** 2))
    float_mae = float(np.mean(np.abs(float_pred - y_true)))
    quant_mse = float(np.mean((quant_pred - y_true) ** 2))
    quant_mae = float(np.mean(np.abs(quant_pred - y_true)))
    mae_increase_pct = ((quant_mae - float_mae) / max(float_mae, 1e-9)) * 100.0

    print(f"Float   - MSE: {float_mse:.6f}, MAE: {float_mae:.6f}")
    print(f"Quant   - MSE: {quant_mse:.6f}, MAE: {quant_mae:.6f}")
    print(f"MAE increase after quantization: {mae_increase_pct:.2f}%")

    # Sort by x for cleaner visualization
    x_flat = x_test.reshape(-1)
    idx = np.argsort(x_flat)
    x_sorted = x_flat[idx]
    y_true_sorted = y_true[idx]
    float_sorted = float_pred[idx]
    quant_sorted = quant_pred[idx]

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    plt.scatter(x_sorted, y_true_sorted, label='Data', color='orange', marker='.', alpha=0.4)
    plt.plot(
        np.linspace(0, 2*np.pi, 1000),
        np.sin(np.linspace(0, 2*np.pi, 1000)),
        label='True sin(x)',
        color='red',
        linewidth=2.5,
    )
    plt.plot(x_sorted, float_sorted, label='Float model', color='blue', linewidth=1.7)
    plt.plot(x_sorted, quant_sorted, label='Quantized model', color='purple', linewidth=1.7)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Float vs Quantized Sin Wave Approximation')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(MEDIA_DIR.joinpath('sin_wave_quantized_comparison.png'))

    # Spot-check key phase points
    test_values = np.array(
        [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi],
        dtype=np.float32,
    ).reshape(-1, 1)
    float_points = float_model.predict(test_values, verbose=0).reshape(-1)

    quant_points = []
    with torch.no_grad():
        for x in test_values:
            sample = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=device)
            output = executor.forward(inputs=sample)[0]
            quant_points.append(float(output.detach().cpu().numpy().reshape(-1)[0]))

    print("\nSample predictions (float vs quantized):")
    for x, f_pred, q_pred in zip(test_values.reshape(-1), float_points, quant_points):
        actual = np.sin(x)
        print(
            f"  x={x:7.4f}, sin(x)={actual:7.4f}, "
            f"float={f_pred:7.4f}, quant={q_pred:7.4f}, "
            f"|float_err|={abs(actual-f_pred):7.4f}, |quant_err|={abs(actual-q_pred):7.4f}"
        )


def main(target='esp32s3'):
    """
    Main execution flow for training and exporting the model.
    """
    MODELS_TARGET_DIR = MODELS_DIR.joinpath(target)
    MODELS_TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TinyML ESP32 - Sin Wave Approximation MLP")
    print("=" * 60)
    
    # Step 1: Generate training data
    print("\n[Step 1] Generating sine wave training data...")
    X, y = generate_sin_wave_data(num_samples=10000, noise_level=0.05)
    X_test, y_test = generate_sin_wave_data(num_samples=2000, noise_level=0.05)
    print(f"Training samples: {len(X)}")
    print(f"Test samples: {len(X_test)}")

    # Step 2: Create MLP model
    print("\n[Step 2] Creating MLP model...")
    model = MLP_model(input_shape=(1,))
    
    # Step 3: Train the model
    print("\n[Step 3] Training model...")
    history = compile_and_train_model(
        model, X, y,
        epochs=100,
        batch_size=128,
        validation_split=0.2
    )
    
    # Step 5: Export to TFLite format (for esp-tflite-micro)
    print("\n[Step 4] Exporting to TFLite format...")
    tflite_model = export_to_tflite(model, X, output_path=MODELS_TARGET_DIR.joinpath('sin_wave_model.tflite'))

    # Step 5: Export to ONNX format
    print("\n[Step 5] Exporting to ONNX format...")
    ONNX_MODEL_PATH = MODELS_TARGET_DIR.joinpath('sin_wave_model.onnx')
    onnx_model = export_to_onnx(model, output_path=ONNX_MODEL_PATH, opset=13)
    onnx_model = onnx.load_model(ONNX_MODEL_PATH)
    onnx.checker.check_model(onnx_model)
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, "ONNX simplification not validated"
    onnx.save(onnx_model, ONNX_MODEL_PATH)

    # Step 6: Quantize ONNX model for ESPDL
    print("\n[Step 6] Quantizing ONNX model for ESPDL...")
    ESPDL_MODEL_PATH = MODELS_TARGET_DIR.joinpath('sin_wave_model.espdl')
    INPUT_SHAPE = [1, 1]
    DEVICE = 'cpu'

    X_torch, _ = generate_sin_wave_data(num_samples=2000, output_type='torch')
    dataloader = DataLoader(X_torch, batch_size=256, shuffle=False)

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=ONNX_MODEL_PATH,
        espdl_export_file=ESPDL_MODEL_PATH,
        input_shape=INPUT_SHAPE,
        calib_dataloader=dataloader,
        calib_steps=256,
        num_of_bits=8,
        target=target,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1
    )

    # Step 7: Evaluate quantized graph against float model
    print("\n[Step 7] Evaluating quantized model...")
    evaluate_quantized_model(model, quant_ppq_graph, X_test, y_test, device=DEVICE)

    if ESPDL_MODEL_PATH.exists():
        espdl_size_kb = ESPDL_MODEL_PATH.stat().st_size / 1024.0
        print(f"ESPDL model size: {espdl_size_kb:.2f} KB")

    print("\n" + "=" * 60)
    print("Training and export completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - models/sin_wave_model.tflite (for esp-tflite-micro)")
    print("  - models/sin_wave_model.onnx (intermediate format)")
    print("  - models/sin_wave_model.espdl (quantized format for ESPDL)")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and export MLP for sine wave approximation")
    parser.add_argument('--target',
                        type=str,
                        default='esp32s3',
                        choices=['c', 'esp32s3', 'esp32p4'],
                        help='Target ESP32 variant for ESPDL export')
    args = parser.parse_args()
    main(target=args.target)
