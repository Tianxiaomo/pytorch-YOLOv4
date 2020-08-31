import sys
from importlib.abc import MetaPathFinder, Loader
import importlib


class ToolLoader(Loader):
    def module_repr(self, module):
        return repr(module)

    def load_module(self, fullname):
        old_name = fullname
        fullname = "yolov4." + fullname
        module = importlib.import_module(fullname)
        sys.modules[old_name] = module
        return module


class ToolImport(MetaPathFinder):
    def find_module(self, fullname, path=None):
        names = fullname.split(".")
        if len(names) >= 1 and names[0] == "tool":
            return ToolLoader()


sys.meta_path.append(ToolImport())
