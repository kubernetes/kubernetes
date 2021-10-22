from __future__ import print_function

import sys
from google.auth.credentials import AnonymousCredentials
from google.cloud import storage


if len(sys.argv)-1 != 3:
    print('args: PROJECT BUCKET record|replay')
    sys.exit(1)
project = sys.argv[1]
bucket_name = sys.argv[2]
mode = sys.argv[3]

if mode == 'record':
    creds = None  # use default creds for demo purposes; not recommended
    client = storage.Client(project=project)
elif mode == 'replay':
    creds = AnonymousCredentials()
else:
    print('want record or replay')
    sys.exit(1)

client = storage.Client(project=project, credentials=creds)
bucket = client.get_bucket(bucket_name)
print('bucket %s created %s' %(bucket.id, bucket.time_created))
