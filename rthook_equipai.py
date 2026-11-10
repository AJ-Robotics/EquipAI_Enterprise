# Runtime hook for frozen EquipAI builds.
#
# PyInstaller's pyi_rth_inspect hook rewrites relative co_filename values
# ("transformers/image_processing_utils.py") into "<sys._MEIPASS>/transformers/
# image_processing_utils.pyc". transformers' auto_docstring machinery then calls
# inspect.getsourcefile() and tests os.path.basename(...).endswith(".py").
# Because the freeze uses 'pyz+py' collection mode the real source file sits at
# "...py" (not "...pyc"), so that test fails and imports die with
# "ValueError: Unrolling kwargs is not supported for preprocess of None class".
# This hook maps the generated "....pyc"/"....pyo" path back to the physical
# "....py" when the latter exists.


def _pyi_rthook():
    import inspect
    import os
    import sys

    _orig_getsourcefile = inspect.getsourcefile

    def _equipai_getsourcefile(object):
        try:
            filename = _orig_getsourcefile(object)
        except TypeError:
            filename = None
        if filename:
            bundle = getattr(sys, "_MEIPASS", None)
            if bundle and filename.startswith(bundle) and filename.endswith((".pyc", ".pyo")):
                source_path = filename[:-1]  # strip the trailing 'c' / 'o' -> "...py"
                if os.path.isfile(source_path):
                    return source_path
        return filename

    inspect.getsourcefile = _equipai_getsourcefile


_pyi_rthook()
del _pyi_rthook