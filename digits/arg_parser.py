# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import platform
import sys
import argparse
import re
from run_cmd import run_cmd

class ArgParser():
    def __init__(self):
        self.lines = []
        self.arg_type = 'none'
        self.name=''
        self.help=''
        self.usages = []
        self.arg_dict = {'pos':[], 'opt':[]}
        self.action = None
        # arg pattern
        exp = r'^\x20{2}([a-z0-9-_]+)(.*\x20{2,}(.*))?'
        self.pattern = re.compile(exp, re.I)   # re.I Ignore case
        
    
    def get_usage(self):
        help_str = ' '.join(self.usages)
        return help_str
        
        
    def get_args(self, type):
        return self.arg_dict[type]
      
      
    def _print_list(self, args):
        for arg in args:
            if arg['help']:
                print('  %s\t\t\t%s' % (arg['name'], arg['help']))
            else:
                print('  %s' % (arg['name']))
        
        
    def verbose(self):
        # print usages
        print(self.get_usage())
        if len(self.arg_dict['pos']) > 0:
            print('positional arguments:')
            self._print_list(self.arg_dict['pos'])
        if len(self.arg_dict['opt']) > 0:
            print('optional arguments:')
            self._print_list(self.arg_dict['opt'])
        
        
    def __call__(self, line):
        print(line)
        self.lines.append(line)
        # parse
        if line.startswith('positional arguments'):
            self.arg_type = 'pos'
            return True
        if line.startswith('optional arguments'):
            self.arg_type = 'opt'
            return True
        m = self.pattern.match(line)
        if m:
            #print(m.groups())
            name = m.group(1)
            help = m.group(3)
            action = {'name': name, 'help': help}
            if name == '-h':
                #print('Ignore -h')
                pass
            else:
                self.arg_dict[self.arg_type].append(action)
                self.action = action
        else:
            if self.arg_type == 'none':
                self.usages.append(line)
            else:
                # append help to last action
                if self.action:
                    old_help = self.action['help'] if self.action['help'] else ''
                    self.action['help'] = old_help + line.strip()
        return True
    
    
if __name__ == '__main__':
    '''
    exp = r'^\W{2}([a-z0-9-_]+)(.*\W{2,}.*)?'
    pattern = re.compile(exp, re.I)   # re.I Ignore case
    m = pattern.match('Hello World Wide Web')
    '''
    #args = 'python D:/work/gitlab/DeepLoader/eval/run_verify.py -h'
    args = 'python /home/ysten/tzk/run_verify.py -h'
    parser = ArgParser()
    result = run_cmd('/home/ysten/tzk/gitlab/DIGITS', args, process_output=parser, name='jobdir')
    parser.verbose()
    #print(parser.arg_dict)
    print(1111)
    print(parser.get_args('opt'))
    #print(result)
    
    
