import importlib
import pkgutil

import textattack.transformations
import textattack.constraints
import textattack.search_methods


def import_submodules(package, recursive=True):
    """
    Import all submodules of a module, recursively, including subpackages.
    
    Args:
        package: The package to import submodules for.
        recursive: Whether to import submodules recursively.
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = f"{package.__name__}.{name}"
        try:
            module = importlib.import_module(full_name)
            if recursive and is_pkg:
                import_submodules(module)
        except ImportError as e:
            print(f"Failed to import module: {full_name} - {e}")


def register_classes(module, class_dict, base_class):
    """
    Register all subclasses of a base class within a module and its submodules.
    
    Args:
        module: The module to search within.
        class_dict: The dictionary to store the class references.
        base_class: The base class to check against for subclasses.
    """
    if isinstance(module, str):
        module = importlib.import_module(module)
    for _, name, __ in pkgutil.walk_packages(module.__path__, module.__name__ + '.'):
        sub_module = importlib.import_module(name)
        for attribute_name in dir(sub_module):
            attribute = getattr(sub_module, attribute_name)
            if (isinstance(attribute, type) and issubclass(attribute, base_class) and 
                attribute is not base_class and attribute.__name__ not in class_dict):
                class_dict[attribute.__name__] = attribute

CLASS_REGISTRY = {
    'transformations': {},
    'constraints': {},
    'search_methods': {}
}

for category, base_class in [('transformations', textattack.transformations.Transformation),
                             ('constraints', textattack.constraints.Constraint),
                             ('search_methods', textattack.search_methods.SearchMethod)]:
    module = getattr(textattack, category)
    import_submodules(module)
    register_classes(module, CLASS_REGISTRY[category], base_class)

