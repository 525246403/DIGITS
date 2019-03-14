# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import platform
import sys
from io import BlockingIOError
import locale
import signal
import subprocess
import logging
import time

if not platform.system() == 'Windows':
    import fcntl
else:
    import gevent.os

def nonblocking_readlines(f):
    """Generator which yields lines from F (a file object, used only for
       its fileno()) without blocking.  If there is no data, you get an
       endless stream of empty strings until there is data again (caller
       is expected to sleep for a while).
       Newlines are normalized to the Unix standard.
    """
    fd = f.fileno()
    if not platform.system() == 'Windows':
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    enc = locale.getpreferredencoding(False)

    buf = bytearray()
    while True:
        try:
            if not platform.system() == 'Windows':
                block = os.read(fd, 8192)
            else:
                block = gevent.os.tp_read(fd, 8192)
        except (BlockingIOError, OSError):
            yield ""
            continue

        if not block:
            if buf:
                yield buf.decode(enc)
            break

        buf.extend(block)

        while True:
            r = buf.find(b'\r')
            n = buf.find(b'\n')
            if r == -1 and n == -1:
                break

            if r == -1 or r > n:
                yield buf[:(n + 1)].decode(enc)
                buf = buf[(n + 1):]
            elif n == -1 or n > r:
                yield buf[:r].decode(enc) + '\n'
                if n == r + 1:
                    buf = buf[(r + 2):]
                else:
                    buf = buf[(r + 1):]

                    
def join_args(args):
    args = args.encode('utf-8')
    # Convert them all to strings
    if isinstance(args, str):
        print(1231)
	print(type(args))
        return args
    args = [str(x) for x in args]
    # https://docs.python.org/2/library/subprocess.html#converting-argument-sequence
    if platform.system() == 'Windows':
        args = ' '.join(args)
        print('Task subprocess args: "{}"'.format(args))
    else:
	args = ''.join(args)
        print('Task subprocess args: "%s"' % ''.join(args))
    return args 

def output_printer(line):
    print(line)
    return True
    
    
def run_cmd(job_dir, args, process_output=output_printer, after_run=None, aborted=None, name=''):
    """
    Execute the task

    Arguments:
    -- the resources assigned by the scheduler for this task
    """
    status = ""
    env = os.environ.copy()
    logger = logging.getLogger(name)
    ret = False
    if not args:
        logger.error('Could not create the arguments for Popen')
        status = 'ERROR'
        return {'ret': ret, 'status': status}

    print('%s task started.' % name)
    status = 'RUN'

    unrecognized_output = []

    env['PYTHONPATH'] = os.pathsep.join(['.', job_dir, env.get('PYTHONPATH', '')] + sys.path)

    args = join_args(args)
    print(args)
    p = subprocess.Popen( args,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          cwd=job_dir,
                          close_fds=False if platform.system() == 'Windows' else True,
                          shell=True,
                          env=env,
                          )

    try:
        sigterm_time = None  # When was the SIGTERM signal sent
        sigterm_timeout = 2  # When should the SIGKILL signal be sent
        while p.poll() is None:
            for line in nonblocking_readlines(p.stdout):
                if aborted and aborted.is_set():
                    if sigterm_time is None:
                        # Attempt graceful shutdown
                        p.send_signal(signal.SIGTERM)
                        sigterm_time = time.time()
                        status = 'ABORT'
                    break

                if line is not None:
                    # Remove whitespace
                    line = line.rstrip()

                if line:
                    if process_output and not process_output(line):
                        logger.warning('%s unrecognized output: %s' % (name, line.strip()))
                        unrecognized_output.append(line)
                else:
                    time.sleep(0.05)
            if sigterm_time is not None and (time.time() - sigterm_time > sigterm_timeout):
                p.send_signal(signal.SIGKILL)
                logger.warning('Sent SIGKILL to task "%s"' % name)
                time.sleep(0.1)
            time.sleep(0.01)
    except:
        p.terminate()
        if after_run:
            after_run()
        raise

    if after_run:
        after_run()

    if p.returncode != 0:
        logger.error('%s task failed with error code %d' % (name, p.returncode))
        status = 'ERROR'
        ret = False
    else:
        logger.info('%s task completed.' % name)
        status = 'DONE'
        ret = True
        
    return {'ret': ret, 'status': status, 'returncode':p.returncode, 'output':unrecognized_output}
    
    
if __name__ == '__main__':
    result = run_cmd('.', 'dir', name='jobdir')
    print(result)
