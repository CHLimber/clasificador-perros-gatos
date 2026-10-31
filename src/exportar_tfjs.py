"""
Exporta un modelo Keras Sequential (Conv2D, MaxPooling2D, Dropout, Flatten, Dense)
al formato de TensorFlow.js (model.json + group1-shardXofY.bin) SIN usar el paquete
tensorflowjs, que no instala bien en Windows / Python 3.12+.

Uso en el notebook:
    from exportar_tfjs import exportar_tfjs
    exportar_tfjs(modeloCNN_AD, 'carpeta_salida')
"""
import json
from pathlib import Path
import numpy as np

SHARD_BYTES = 4 * 1024 * 1024  # 4 MB por archivo, igual que tensorflowjs_converter


def _init(nombre):
    return {"class_name": nombre, "config": {"seed": None} if nombre != "Zeros" else {}}


def _capa_a_config(capa, primera, input_shape):
    tipo = capa.__class__.__name__
    cfg = {"name": capa.name, "trainable": True, "dtype": "float32"}
    if primera:
        cfg["batch_input_shape"] = [None] + list(input_shape)
    if tipo == "Conv2D":
        cfg.update({
            "filters": capa.filters, "kernel_size": list(capa.kernel_size),
            "strides": list(capa.strides), "padding": capa.padding,
            "data_format": "channels_last", "dilation_rate": list(capa.dilation_rate),
            "activation": capa.activation.__name__, "use_bias": capa.use_bias,
            "kernel_initializer": _init("GlorotUniform"), "bias_initializer": _init("Zeros"),
            "kernel_regularizer": None, "bias_regularizer": None,
            "activity_regularizer": None, "kernel_constraint": None, "bias_constraint": None})
    elif tipo == "MaxPooling2D":
        cfg.update({"pool_size": list(capa.pool_size), "strides": list(capa.strides),
                    "padding": capa.padding, "data_format": "channels_last"})
    elif tipo == "Dropout":
        cfg.update({"rate": float(capa.rate), "noise_shape": None, "seed": None})
    elif tipo == "Flatten":
        cfg.update({"data_format": "channels_last"})
    elif tipo == "Dense":
        cfg.update({
            "units": capa.units, "activation": capa.activation.__name__,
            "use_bias": capa.use_bias,
            "kernel_initializer": _init("GlorotUniform"), "bias_initializer": _init("Zeros"),
            "kernel_regularizer": None, "bias_regularizer": None,
            "activity_regularizer": None, "kernel_constraint": None, "bias_constraint": None})
    else:
        raise ValueError(f"Capa no soportada por este exportador: {tipo}")
    return {"class_name": tipo, "config": cfg}


def exportar_tfjs(modelo, carpeta_salida):
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    input_shape = tuple(modelo.inputs[0].shape[1:])

    capas, pesos, manifiesto = [], [], []
    for i, capa in enumerate(modelo.layers):
        capas.append(_capa_a_config(capa, i == 0, input_shape))
        nombres = ["kernel", "bias"]
        for nombre, w in zip(nombres, capa.get_weights()):
            w = np.asarray(w, dtype="<f4")
            pesos.append(w.ravel())
            manifiesto.append({"name": f"{capa.name}/{nombre}",
                               "shape": list(w.shape), "dtype": "float32"})

    datos = np.concatenate(pesos).tobytes()
    for viejo in carpeta.glob("group1-shard*.bin"):
        viejo.unlink()
    n = max(1, int(np.ceil(len(datos) / SHARD_BYTES)))
    rutas = []
    for k in range(n):
        nombre = f"group1-shard{k + 1}of{n}.bin"
        (carpeta / nombre).write_bytes(datos[k * SHARD_BYTES:(k + 1) * SHARD_BYTES])
        rutas.append(nombre)

    model_json = {
        "format": "layers-model",
        "generatedBy": "keras v2.15.0",
        "convertedBy": "exportar_tfjs.py",
        "modelTopology": {
            "keras_version": "2.15.0", "backend": "tensorflow",
            "model_config": {"class_name": "Sequential",
                             "config": {"name": "sequential", "layers": capas}},
            "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"],
                                "optimizer_config": {"class_name": "Adam", "config": {}}}},
        "weightsManifest": [{"paths": rutas, "weights": manifiesto}],
    }
    (carpeta / "model.json").write_text(json.dumps(model_json), encoding="utf-8")
    print(f"Exportado en {carpeta.resolve()}: model.json + {n} archivo(s) .bin "
          f"({len(datos) / 1e6:.1f} MB, {modelo.count_params():,} parámetros)")
