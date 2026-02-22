"""
train_and_export.py
====================
Train an MLP for occupancy detection (UCI dataset) on Keras, then export
the model for two ESP32 inference frameworks:
  - esp-tflite-micro  (.tflite, full int8 quantisation)
  - esp-dl             (.espdl via esp-ppq, guarded by try/except)
"""

import os
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1.  Data loading & preparation
# ---------------------------------------------------------------------------

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Read an occupancy-dataset CSV, drop the row-index and date columns.
    """
    df = pd.read_csv(filepath)
    cols_to_drop = [c for c in df.columns if c.startswith("Unnamed")] + ["date"]
    df = df.drop(columns=cols_to_drop)
    return df


def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Split features / label and standardise using training-set statistics.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray  (float32)
    scaler : fitted StandardScaler (needed later for calibration)
    """
    feature_columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    target_column = "Occupancy"

    X_train = df_train[feature_columns].values.astype(np.float32)
    y_train = df_train[target_column].values.astype(np.float32)
    X_test = df_test[feature_columns].values.astype(np.float32)
    y_test = df_test[target_column].values.astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, y_train, X_test, y_test, scaler


# ---------------------------------------------------------------------------
# 2.  MLP definition and training
# ---------------------------------------------------------------------------

def build_model(input_dim: int) -> tf.keras.Model:
    """Build a small MLP: input → 128 → 64 → 32 → 16 → 1 (sigmoid)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32
):
    """
    Compile and fit the model. Returns the Keras History object.
    """
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )
    return history


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Print loss and accuracy on the test set.
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"[Test] loss={loss:.4f}  accuracy={accuracy:.4f}")
    return loss, accuracy


# ---------------------------------------------------------------------------
# 3.  Quantisation & export for esp-tflite-micro
# ---------------------------------------------------------------------------

def export_tflite(
    model: tf.keras.Model,
    X_calibration: np.ndarray,
    output_path: str,
) -> None:
    """
    Full int8 quantisation → .tflite file.
    """

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Representative dataset generator (required for full int8)
    def representative_dataset():
        for i in range(len(X_calibration)):
            sample = X_calibration[i : i + 1]
            yield [sample]

    converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"[TFLite] saved → {output_path}  ({len(tflite_model)} bytes)")


# ---------------------------------------------------------------------------
# 4.  Export & quantisation for esp-dl
# ---------------------------------------------------------------------------

def _patch_numpy_for_tf2onnx() -> None:
    """Restore NumPy aliases removed in NumPy 2.0 that tf2onnx still needs.

    Must be called *before* ``import tf2onnx`` because the missing
    attributes are referenced at module-import time.
    """
    _compat = {
        "object": object,
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "str": str,
    }
    for attr, builtin in _compat.items():
        if not hasattr(np, attr):
            setattr(np, attr, builtin)

    if not hasattr(np, "cast"):
        class _NpCastCompat:
            """Mimics the old np.cast[dtype](array) interface."""
            def __getitem__(self, dtype):
                return lambda x: np.asarray(x, dtype=dtype)
        np.cast = _NpCastCompat()


def export_onnx(model: tf.keras.Model, output_path: str) -> None:
    """Keras → ONNX (opset 13).  Requires tf2onnx and onnx packages.

    Applies compatibility shims for:
      - NumPy 2.0 removed aliases (np.object, np.cast, …)
      - Keras 3 / TF 2.16+ removed ``model.output_names``
    """
    # --- compatibility patches (must precede import tf2onnx) -------------
    _patch_numpy_for_tf2onnx()

    if not hasattr(model, "output_names"):
        model.output_names = ["output"]

    import onnx
    import tf2onnx

    input_dim = model.input_shape[1]
    input_signature = [
        tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32, name="input")
    ]
    onnx_model, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_signature, opset=13
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx.save(onnx_model, output_path)
    print(f"[ONNX]   saved → {output_path}")


def export_espdl(
    onnx_path: str,
    X_calibration: np.ndarray,
    output_path: str,
) -> None:
    """
    ONNX → ESPDL quantised model.  Requires esp-ppq (guarded).
    """
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from esp_ppq.api import espdl_quantize_onnx
    except ImportError as exc:
        print(
            f"[esp-dl] Skipped – could not import required packages ({exc}).\n"
            "         Install esp-ppq (and torch) to enable ESPDL export."
        )
        return

    # Build calibration DataLoader expected by espdl_quantize_onnx
    calib_tensor = torch.tensor(X_calibration, dtype=torch.float32)
    calib_dataset = TensorDataset(calib_tensor)
    calib_dataloader = DataLoader(calib_dataset, batch_size=32, shuffle=False)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=output_path,
        calib_dataloader=calib_dataloader,
        calib_steps=len(calib_dataloader),
        input_shape=[1, X_calibration.shape[1]],
        target="esp32",
        verbose=True,
    )
    print(f"[ESPDL]  saved → {output_path}")


# ---------------------------------------------------------------------------
# 5.  Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an occupancy-detection MLP and export for ESP32."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50, 
        help="Training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory for exported model files",
    )
    args = parser.parse_args()

    # --- data -----------------------------------------------------------
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    df_train = load_csv(os.path.join(dataset_dir, "datatraining.txt"))
    df_test = load_csv(os.path.join(dataset_dir, "datatest.txt"))

    X_train, y_train, X_test, y_test, scaler = prepare_data(df_train, df_test)
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Test samples     : {X_test.shape[0]}")

    # --- build & train --------------------------------------------------
    model = build_model(input_dim=X_train.shape[1])
    model.summary()
    train_model(model, X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    evaluate_model(model, X_test, y_test)

    # --- exports --------------------------------------------------------
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tflite_path = os.path.join(output_dir, "occupancy_model.tflite")
    export_tflite(model, X_train, tflite_path)

    onnx_path = os.path.join(output_dir, "occupancy_model.onnx")
    export_onnx(model, onnx_path)

    espdl_path = os.path.join(output_dir, "occupancy_model.espdl")
    export_espdl(onnx_path, X_train, espdl_path)

    print("\nDone. Exported files are in:", os.path.abspath(output_dir))


if __name__ == "__main__":
    main()
