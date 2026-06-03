# by xueqianyue

import os
import sys
import yaml
import inspect
import importlib
import warnings
from typing import Dict, Any

__all__ = ['GLOBAL_CONFIG', 'register', 'create', 'load_config', 'merge_config', 'merge_dict']


GLOBAL_CONFIG: Dict[str, Any] = dict()
REGISTRY: Dict[str, object] = dict()
INCLUDE_KEY = '__include__'


def _extract_schema(cls: type) -> Dict[str, Any]:

    argspec = inspect.getfullargspec(cls.__init__)

    pos_args = [arg for arg in (argspec.args or []) if arg != 'self']

    kwonly_args = list(argspec.kwonlyargs or [])

    defaults_map: Dict[str, Any] = {}

    if argspec.defaults:

        tail_pos = pos_args[-len(argspec.defaults):] if len(argspec.defaults) else []
        for name, val in zip(tail_pos, argspec.defaults):
            defaults_map[name] = val

    if argspec.kwonlydefaults:
        defaults_map.update(argspec.kwonlydefaults)

    schema: Dict[str, Any] = dict()
    schema['_name'] = cls.__name__

    module_obj = sys.modules.get(cls.__module__)
    if module_obj is None:
        module_obj = importlib.import_module(cls.__module__)
    schema['_pymodule'] = module_obj

    schema['_inject'] = list(getattr(cls, '__inject__', []))
    schema['_share']  = list(getattr(cls, '__share__', []))


    for name in pos_args + kwonly_args:
        value = defaults_map.get(name, None)

        if name in schema['_share'] and name not in defaults_map:
            warnings.warn(f"[yaml_utils] share param '{name}' of {cls.__name__} "
                          f"has no default value; it will be None unless provided in GLOBAL_CONFIG.")
        schema[name] = value

    return schema


def _copy_schema_shallow(schema: Dict[str, Any]) -> Dict[str, Any]:

    out = dict(schema)
    if '_inject' in out and isinstance(out['_inject'], list):
        out['_inject'] = list(out['_inject'])
    if '_share' in out and isinstance(out['_share'], list):
        out['_share'] = list(out['_share'])
    return out


def register(obj):

    name = getattr(obj, "__name__", str(obj))
    existed = REGISTRY.get(name, None)

    if existed is None:
        REGISTRY[name] = obj
        if inspect.isclass(obj):
            schema = _extract_schema(obj)


            if not isinstance(GLOBAL_CONFIG.get(name), dict):
                GLOBAL_CONFIG[name] = schema
        else:

            GLOBAL_CONFIG[name] = obj
    elif existed is not obj:
        warnings.warn(f"{name} already registered, keep the first one and skip the duplicate.")

        if name not in GLOBAL_CONFIG:
            GLOBAL_CONFIG[name] = existed
    return obj


def create(type_or_name, **kwargs):

    assert isinstance(type_or_name, (type, str)), 'create should be class or name.'
    name = type_or_name if isinstance(type_or_name, str) else type_or_name.__name__

    if name not in GLOBAL_CONFIG:
        raise ValueError(f'The module {name} is not registered')

    cfg = GLOBAL_CONFIG[name]


    if not isinstance(cfg, dict):
        return cfg


    if 'type' in cfg:
        target_type = str(cfg['type'])
        if target_type not in GLOBAL_CONFIG:
            raise ValueError(f'Missing {target_type} in inspect stage.')
        base_schema = _copy_schema_shallow(GLOBAL_CONFIG[target_type])
        local_override = dict(cfg); local_override.pop('type', None)
        local_override.update(kwargs)
        return _instantiate_from_schema(target_type, base_schema, local_override)


    return _instantiate_from_schema(name, _copy_schema_shallow(cfg), kwargs)


def _instantiate_from_schema(type_name: str, schema: Dict[str, Any], override: Dict[str, Any]):

    cls = getattr(schema['_pymodule'], type_name)
    if not inspect.isclass(cls):

        return cls

    argspec = inspect.getfullargspec(cls.__init__)
    arg_names = [arg for arg in (argspec.args or []) if arg != 'self']

    cls_kwargs = dict(schema)
    cls_kwargs.update(override)


    for k in schema.get('_share', []):
        if k in GLOBAL_CONFIG:
            cls_kwargs[k] = GLOBAL_CONFIG[k]
        else:
            cls_kwargs[k] = schema.get(k, None)


    for k in schema.get('_inject', []):
        if k not in cls_kwargs:
            continue
        _k = cls_kwargs[k]
        if _k is None:
            continue

        if isinstance(_k, str):
            if _k not in GLOBAL_CONFIG:
                raise ValueError(f'Missing inject config of `{_k}`.')
            ref_cfg = GLOBAL_CONFIG[_k]
            if isinstance(ref_cfg, dict):
                cls_kwargs[k] = create(_k)
            else:
                cls_kwargs[k] = ref_cfg

        elif isinstance(_k, dict):
            if 'type' not in _k:
                raise ValueError('Missing inject for `type` style.')
            _type = str(_k['type'])
            if _type not in GLOBAL_CONFIG:
                raise ValueError(f'Missing {_type} in inspect stage.')
            base_schema = _copy_schema_shallow(GLOBAL_CONFIG[_type])
            local_override = dict(_k); local_override.pop('type', None)
            cls_kwargs[k] = _instantiate_from_schema(_type, base_schema, local_override)

        else:
            raise ValueError(f'Inject does not support: {type(_k).__name__}')


    final_kwargs = {n: cls_kwargs[n] for n in arg_names if n in cls_kwargs}
    return cls(**final_kwargs)


def load_config(file_path, cfg=dict()):

    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"

    with open(file_path, encoding='utf-8') as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)
        if file_cfg is None:
            return {}

    if INCLUDE_KEY in file_cfg:
        base_yamls = list(file_cfg[INCLUDE_KEY])
        for base_yaml in base_yamls:
            if base_yaml.startswith('~'):
                base_yaml = os.path.expanduser(base_yaml)
            if not base_yaml.startswith('/'):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)
            base_cfg = load_config(base_yaml, cfg)
            merge_config(base_cfg, cfg)

    return merge_config(file_cfg, cfg)


def merge_dict(dct, another_dct):

    for k in another_dct:
        if k in dct and isinstance(dct[k], dict) and isinstance(another_dct[k], dict):
            merge_dict(dct[k], another_dct[k])
        else:
            dct[k] = another_dct[k]
    return dct


def merge_config(config, another_cfg=None):

    dct = GLOBAL_CONFIG if another_cfg is None else another_cfg
    return merge_dict(dct, config)
