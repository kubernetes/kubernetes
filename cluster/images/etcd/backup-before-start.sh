#!/bin/bash

set -o errexit
set -o nounset

DATA_DIRECTORY=${DATA_DIRECTORY:-/var/etcd/data/}

etcd_init_backup_dir="${DATA_DIRECTORY}"/initial-data-backups/
etcd_db_file="${DATA_DIRECTORY}"/member/snap/db

if [ -f ${etcd_db_file} ]; then
  echo "Found etcd db file: "${etcd_db_file}". Creating a copy."
  mkdir -p ${etcd_init_backup_dir}
  file_count=$(ls -1q ${etcd_init_backup_dir} | wc -l)
  if [ ${file_count} -ge 3 ]; then
    echo "Number of backup files already reached max value of ${file_count}. Skip the current backup."
  else
    ts=$(date +%s)
    cp ${etcd_db_file} ${etcd_init_backup_dir}/$(expr $ts / 300)_snapshot.db
  fi
else
  echo "No etcd db file found under "${etcd_db_file}"."
fi
