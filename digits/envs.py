# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import platform

__ENV_KEYS = ['PATH', 'PYTHONPATH', 'PKG_CONFIG_PATH', 'LD_LIBRARY_PATH']

def is_windows():
    return platform.system() == "Windows"
    
    
def is_linux():
    return platform.system() == "Linux"
    
    
def env_sep():
    return ':' if not is_windows() else ';'
    
    
def get_env_dict():
    envs = os.environ
    print(envs)
    d = {}
    sep = env_sep()
    for key in __ENV_KEYS:
        if not key in envs:
            continue
        value = envs[key]
        strs = value.split(sep)
        # filter invalid
        valids = []
        for s in strs:
            s = s.strip()
            if len(s) > 0:
                valids.append(s)
        d[key] = valids
    return d
    
    
def print_env_dict(envs):
    for k in envs.keys():
        v = envs[k]
        print('%s\t\t%d' % (k, len(v)))
        for s in v:
            print('\t%s'% (s))


if __name__ == '__main__':
    envs = get_env_dict()
    print_env_dict(envs)
    
