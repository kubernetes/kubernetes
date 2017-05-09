import os.path
import os
import logging
import base64
import json

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger.info('BASE_DIR %s', BASE_DIR)
for base, dirs, files, in os.walk(BASE_DIR):
    if base == BASE_DIR:
        continue
    if not os.path.dirname(base) == BASE_DIR:
        logger.warning('No nested dirs supported in secrets: %r', base)
        continue
    logger.info('Processing dir %r with files %r', base, files)
    content = {
        'kind': 'Secret',
        'apiVersion': 'v1',
        'metadata': os.path.basename(base),
        'data': {}
    }
    for file in files:
        fp = os.path.join(base, file)
        with open(fp, mode='rb') as fd:
            fc = fd.read()
        content['data'][file] = base64.b64encode(fc).decode()

    with open('%s.json' % base, mode='wt') as fd:
        fd.write(json.dumps(content))
