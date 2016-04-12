#!/usr/bin/env python

# Copyright 2016 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Caches requests to the GCE metadata server.

Reduces load on metadata server to once for most requests and once per
~10m for access tokens.

See README.md for instructions for the whole system.

Usage:
  screen python metadata-cache.py
"""
import collections
import json
import logging
import logging.handlers
import socket
import threading
import time

import flask
import requests

app = flask.Flask(__name__)

LOCK = threading.Lock()
SESSION = requests
URL = 'http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token'
logger = None
_cached_tokens = collections.defaultdict(dict)
_global_cache = {}


def fetch_token():
  """Fetch a new token from the metadata server, retrying any errors."""
  pause = False
  while True:
    if pause:
      time.sleep(1)
    pause = True
    seconds = time.time()
    try:
      logger.info('GET: %s' % URL)
      resp = SESSION.get(
          URL,
          headers={
              'Metadata-Flavor': 'Google',
              'Host': 'metadata.google.internal',
          },
          allow_redirects=False,
      )
      logger.info('GET: %d %s' % (resp.status_code, URL))
      resp.raise_for_status()
    except IOError:
      logger.exception('Error reading response from metadata server')
      continue
    try:
      content = resp.content
    except IOError:
      logger.exception('Error reading response')
      continue
    safe_content = content.encode('utf-8', errors='ignore')
    try:
      data = json.loads(content)
    except ValueError:
      logger.exception('Could not decode response: %s' % safe_content)
      continue
    if data.get('token_type') != 'Bearer':
      logger.error('Not a bearer token: %s' % json.dumps(data, indent=1))
      continue
    return seconds, data


def cached_token(uri):
  """Return the access token, adjusting expires_in and potentially fetching."""
  while time.time() + 10 > _cached_tokens[uri].get('expiration', 0):
    logger.info('Refreshing expired token: %s' % _cached_tokens[uri])
    seconds, token = fetch_token()
    logger.info('New token: %s' % json.dumps(token, indent=1))
    token['expiration'] = seconds + token['expires_in']
    _cached_tokens[uri].clear()
    _cached_tokens[uri].update(token)
  this_token = {k: v for (k, v) in _cached_tokens[uri].items() if k != 'expiration'}
  this_token['expires_in'] = int(_cached_tokens[uri]['expiration'] - time.time())
  return json.dumps(this_token)


def cache_request(uri):
  if uri not in _global_cache:
    with LOCK:
      if uri not in _global_cache:
        r2, ok = proxy_request(uri)
        if not ok:
          logger.warn('Request failed: %s %s' % (uri, r2))
          return r2
        _global_cache[uri] = r2
  return _global_cache[uri]


def proxy_request(uri):
  """Proxy a request to uri using a connection to 169.254.169.254."""
  logger.info('GET: %s' % uri)
  headers = dict(flask.request.headers)
  headers['Host'] = 'metadata.google.internal'
  resp = SESSION.get(
      'http://169.254.169.254/%s' % uri,
      headers=headers,
      allow_redirects=False,
  )
  logger.info('GET: %d %s' % (resp.status_code, uri))
  r2 = flask.make_response(resp.content, resp.status_code)
  for k, v in resp.headers.items():
    r2.headers.set(k, v)
  return r2, resp.ok


@app.route('/')
def get_root_response():
  return cache_request('')


@app.route('/<path:uri>')
def get_path_response(uri):
  """Return the cached token as a string."""
  if uri.endswith('/token'):
    return cached_token(uri)
  return cache_request(uri)


def listen_address():
  """Return the ip address to bind, which should be an internal one."""
  ip = socket.gethostbyname(socket.gethostname())
  if not ip.startswith('10.'): raise ValueError('Not a private ip', ip)
  return ip


def setup_logger():
  """Configure to log everything to the screen and /var/log/syslog."""
  logs = logging.getLogger('metadata-cache')
  logs.setLevel(logging.DEBUG)
  handler = logging.handlers.SysLogHandler(
      address='/dev/log',
      facility=logging.handlers.SysLogHandler.LOG_SYSLOG)
  formatter = logging.Formatter('metadata-cache: %(levelname)s %(message)s')
  handler.setFormatter(formatter)
  handler.setLevel(logging.DEBUG)
  logs.addHandler(handler)
  sh = logging.StreamHandler()
  sh.setFormatter(formatter)
  sh.setLevel(logging.DEBUG)
  logs.addHandler(sh)
  return logs


if __name__ == '__main__':
  logger = setup_logger()
  app.run(host=listen_address(), port=80)
