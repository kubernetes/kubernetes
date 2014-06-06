# This is a quick script that adds AllUsers as READER to a JSON file
# representing an ACL on a GCS object.  This is a quick workaround for a bug in
# gsutil.
import json
import sys

acl = json.load(sys.stdin)
acl.append({
    "entity": "allUsers",
    "role": "READER"
  })
json.dump(acl, sys.stdout)
