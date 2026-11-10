# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_all

datas = [('ui_components.py.', '.'), ('rag_engine.py.', '.'), ('llm_client.py.', '.'), ('session_manager.py.', '.'), ('feedback_manager.py.', '.'), ('config.py.', '.'), ('analytics.py.', '.'), ('model_manager.py.', '.'), ('D:/EqpAI - IndustryX/.venv/Lib/site-packages/setuptools/_vendor', 'setuptools/_vendor'), ('C:/tmp/eqpai_tf_models', 'transformers/models')]
binaries = []
hiddenimports = ['llama_cpp', 'faiss', 'torch']
datas += collect_data_files('sentence_transformers')
datas += collect_data_files('transformers')
binaries += collect_dynamic_libs('llama_cpp')
binaries += collect_dynamic_libs('faiss')
tmp_ret = collect_all('huggingface_hub')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('qtawesome')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['ui_main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_equipai.py'],
    excludes=['tensorflow', 'keras', 'tensorboard', 'torchvision', 'torchaudio', 'onnxruntime', 'cv2', 'av', 'soundfile', 'librosa', 'pydub', 'nltk', 'matplotlib', 'altair', 'jsonschema', 'boto3', 'botocore', 'gevent', 'zope.interface', 'httplib2', 'grpc', 'orjson', 'uvicorn', 'gradio', 'gradio_client', 'streamlit', 'tkinter', 'pytest', 'torch.testing._internal.opinfo', 'notebook'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='EquipAI_Enterprise',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='version_info.txt',
    icon=['D:/EqpAI - IndustryX/media/equipai.ico'],
)
