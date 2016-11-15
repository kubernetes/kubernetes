#!/bin/sh

set -o nounset

fail() {
  echo FAIL
  exit 1
}

pass() {
  echo PASS
  exit 0
}

migrate_if_needed() {
  export TARGET_STORAGE="$1"
  export TARGET_VERSION="$2"
  echo "Migrating to ${TARGET_STORAGE}/${TARGET_VERSION}"
  t0=$(date +%s)
  /usr/local/bin/migrate-if-needed.sh >> /var/log/migrate.log 2>&1
  if [ "$?" != 0 ]; then
    echo "Migrate to ${TARGET_STORAGE}/${TARGET_VERSION} failed."
    exit 1
  fi
  t=$(date +%s)
  echo "Took $(( t - t0 )) seconds."
}

start_etcd() {
  echo "Starting etcd..."
  /usr/local/bin/etcd --debug --name etcd-master --listen-peer-urls http://127.0.0.1:2380 --initial-advertise-peer-urls http://127.0.0.1:2380 --advertise-client-urls http://127.0.0.1:2379 --listen-client-urls http://127.0.0.1:2379 --data-dir /var/etcd/data --initial-cluster-state new --initial-cluster etcd-master=http://127.0.0.1:2380 1>>/var/log/etcd.log 2>&1 &
  ETCD_PID=$!
}

start_apiserver() {
  echo "Starting apiserver..."
  /usr/local/bin/kube-apiserver --v=2 --runtime-config=extensions/v1beta1 --delete-collection-workers=1 --address=127.0.0.1 --allow-privileged=true --etcd-servers=http://127.0.0.1:2379 --etcd-servers-overrides=/events#http://127.0.0.1:4002 --storage-backend=etcd2 --target-ram-mb=120 --service-cluster-ip-range=10.0.0.0/16 --min-request-timeout=300 --runtime-config=api/all=true --authorization-mode=AlwaysAllow --allow-privileged=true 1>>/var/log/kube-apiserver.log 2>&1 &
  API_PID=$!
}

kill_em_all() {
  echo -n "Killing etcd and apiserver..."
  kill $ETCD_PID
  kill $API_PID
  wait
  echo "done"
}

try_dump() {
  echo -n "Trying to dump"
  for i in $(seq 1 300); do
    {
      echo -n '.'
      /usr/local/bin/kubectl --server=http://127.0.0.1:8080/ get -oyaml nodes > /var/log/kubectl-nodes-${1}-${TARGET_VERSION}-${TARGET_STORAGE}.log 2>> /var/log/kubectl-err.log
      if [ "$?" == "0" ]; then
        /usr/local/bin/kubectl --server=http://127.0.0.1:8080/ get --all-namespaces -oyaml pods > /var/log/kubectl-pods-${1}-${TARGET_VERSION}-${TARGET_STORAGE}.log 2>> /var/log/kubectl-err.log
        if [ "$?" == "0" ]; then
          echo "dumped"
          return
        fi
      fi
      sleep 1
    }
  done
}

migrate_if_needed etcd2 2.2.1
start_etcd
start_apiserver
try_dump start
kill_em_all

migrate_if_needed etcd2 2.3.7
start_etcd
start_apiserver
try_dump 237
kill_em_all

migrate_if_needed etcd3 3.0.14
start_etcd
start_apiserver
try_dump upgraded
kill_em_all

for what in nodes pods; do
  echo "Checking ${what}..."
  for dump in 237 upgraded; do
    echo -n " start vs ${dump}..."
    diff -u /var/log/kubectl-${what}-start* /var/log/kubectl-${what}-${dump}* || fail
    echo OK
  done
done

pass

# TODO(mml): Currently busted: https://github.com/kubernetes/kubernetes/issues/36555
# migrate_if_needed etcd2 2.3.7
# start_etcd
# start_apiserver
# try_dump rollback
# kill_em_all
