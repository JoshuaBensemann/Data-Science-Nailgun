import importlib
import pkgutil

# Optional: List submodules to be exposed via `from my_package import *`
# This list is separate from the dynamic import and only affects star imports.
__all__ = []

# Dynamically import all submodules
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    # This ensures that both modules (is_pkg=False) and sub-packages (is_pkg=True) are imported.
    full_module_name = f"{__name__}.{module_name}"
    importlib.import_module(full_module_name)

    # If using `__all__`, you can add the module name to it.
    # The module has already been loaded, but this makes it available for star imports.
    __all__.append(module_name)
