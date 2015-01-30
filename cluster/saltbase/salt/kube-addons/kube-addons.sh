#!/bin/bash
#
# The business logic for whether a given object should be created
# was already enforced by salt, and /etc/kubernetes/addons is the
# managed result is of that. Start everything below that directory.
echo "== Kubernetes addon manager started at $(date -Is) =="
KUBECTL=/usr/local/bin/kubectl
for obj in $(find /etc/kubernetes/addons -name \*.yaml); do
  ${KUBECTL} --server="127.0.0.1:8080" create -f ${obj} &
  echo "++ addon ${obj} started in pid $! ++"
done
noerrors="true"
for pid in $(jobs -p); do
  wait ${pid} || noerrors="false"
  echo "++ pid ${pid} complete ++"
done
if [ ${noerrors} == "true" ]; then
  echo "== Kubernetes addon manager completed successfully at $(date -Is) =="
else
  echo "== Kubernetes addon manager completed with errors at $(date -Is) =="
fi

# We stay around so that status checks by salt make it look like
# the service is good. (We could do this is other ways, but this
# is simple.)
sleep infinity
