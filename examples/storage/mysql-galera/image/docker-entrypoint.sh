#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

# 
# This script does the following:
# 
# 1. Sets up database privileges by building an SQL script
# 2. MySQL is initially started with this script a first time 
# 3. Modify my.cnf and cluster.cnf to reflect available nodes to join
# 

# if NUM_NODES not passed, default to 3
if [ -z "$NUM_NODES" ]; then
  NUM_NODES=3
fi

if [ "${1:0:1}" = '-' ]; then
  set -- mysqld "$@"
fi

# if the command passed is 'mysqld' via CMD, then begin processing. 
if [ "$1" = 'mysqld' ]; then
  # read DATADIR from the MySQL config
  DATADIR="$("$@" --verbose --help 2>/dev/null | awk '$1 == "datadir" { print $2; exit }')"
 
  # only check if system tables not created from mysql_install_db and permissions 
  # set with initial SQL script before proceeding to build SQL script
  if [ ! -d "$DATADIR/mysql" ]; then
  # fail if user didn't supply a root password  
    if [ -z "$MYSQL_ROOT_PASSWORD" -a -z "$MYSQL_ALLOW_EMPTY_PASSWORD" ]; then
      echo >&2 'error: database is uninitialized and MYSQL_ROOT_PASSWORD not set'
      echo >&2 '  Did you forget to add -e MYSQL_ROOT_PASSWORD=... ?'
      exit 1
    fi

    # mysql_install_db installs system tables
    echo 'Running mysql_install_db ...'
        mysql_install_db --datadir="$DATADIR"
        echo 'Finished mysql_install_db'
    
    # this script will be run once when MySQL first starts to set up
    # prior to creating system tables and will ensure proper user permissions 
    tempSqlFile='/tmp/mysql-first-time.sql'
    cat > "$tempSqlFile" <<-EOSQL
DELETE FROM mysql.user ;
CREATE USER 'root'@'%' IDENTIFIED BY '${MYSQL_ROOT_PASSWORD}' ;
GRANT ALL ON *.* TO 'root'@'%' WITH GRANT OPTION ;
EOSQL
    
    if [ "$MYSQL_DATABASE" ]; then
      echo "CREATE DATABASE IF NOT EXISTS \`$MYSQL_DATABASE\` ;" >> "$tempSqlFile"
    fi
    
    if [ "$MYSQL_USER" -a "$MYSQL_PASSWORD" ]; then
      echo "CREATE USER '$MYSQL_USER'@'%' IDENTIFIED BY '$MYSQL_PASSWORD' ;" >> "$tempSqlFile"
      
      if [ "$MYSQL_DATABASE" ]; then
        echo "GRANT ALL ON \`$MYSQL_DATABASE\`.* TO '$MYSQL_USER'@'%' ;" >> "$tempSqlFile"
      fi
    fi

    # Add SST (Single State Transfer) user if Clustering is turned on
    if [ -n "$GALERA_CLUSTER" ]; then
    # this is the Single State Transfer user (SST, initial dump or xtrabackup user)
      WSREP_SST_USER=${WSREP_SST_USER:-"sst"}
      if [ -z "$WSREP_SST_PASSWORD" ]; then
        echo >&2 'error: Galera cluster is enabled and WSREP_SST_PASSWORD is not set'
        echo >&2 '  Did you forget to add -e WSREP_SST__PASSWORD=... ?'
        exit 1
      fi
      # add single state transfer (SST) user privileges
      echo "CREATE USER '${WSREP_SST_USER}'@'localhost' IDENTIFIED BY '${WSREP_SST_PASSWORD}';" >> "$tempSqlFile"
      echo "GRANT RELOAD, LOCK TABLES, REPLICATION CLIENT ON *.* TO '${WSREP_SST_USER}'@'localhost';" >> "$tempSqlFile"
    fi

    echo 'FLUSH PRIVILEGES ;' >> "$tempSqlFile"
    
    # Add the SQL file to mysqld's command line args
    set -- "$@" --init-file="$tempSqlFile"
  fi
  
  chown -R mysql:mysql "$DATADIR"
fi

# if cluster is turned on, then proceed to build cluster setting strings
# that will be interpolated into the config files
if [ -n "$GALERA_CLUSTER" ]; then
  # this is the Single State Transfer user (SST, initial dump or xtrabackup user)
  WSREP_SST_USER=${WSREP_SST_USER:-"sst"}
  if [ -z "$WSREP_SST_PASSWORD" ]; then
    echo >&2 'error: database is uninitialized and WSREP_SST_PASSWORD not set'
    echo >&2 '  Did you forget to add -e WSREP_SST_PASSWORD=xxx ?'
    exit 1
  fi

  # user/password for SST user
  sed -i -e "s|^wsrep_sst_auth=sstuser:changethis|wsrep_sst_auth=${WSREP_SST_USER}:${WSREP_SST_PASSWORD}|" /etc/mysql/conf.d/cluster.cnf

  # set nodes own address
  WSREP_NODE_ADDRESS=`ip addr show | grep -E '^[ ]*inet' | grep -m1 global | awk '{ print $2 }' | sed -e 's/\/.*//'`
  if [ -n "$WSREP_NODE_ADDRESS" ]; then
    sed -i -e "s|^wsrep_node_address=.*$|wsrep_node_address=${WSREP_NODE_ADDRESS}|" /etc/mysql/conf.d/cluster.cnf
  fi
  
  # if the string is not defined or it only is 'gcomm://', this means bootstrap
  if [ -z "$WSREP_CLUSTER_ADDRESS" -o "$WSREP_CLUSTER_ADDRESS" == "gcomm://" ]; then
    # if empty, set to 'gcomm://'
    # NOTE: this list does not imply membership. 
    # It only means "obtain SST and join from one of these..."
    if [ -z "$WSREP_CLUSTER_ADDRESS" ]; then
      WSREP_CLUSTER_ADDRESS="gcomm://"
    fi

    # loop through number of nodes
    for NUM in `seq 1 $NUM_NODES`; do
      NODE_SERVICE_HOST="PXC_NODE${NUM}_SERVICE_HOST"
  
      # if set
      if [ -n "${!NODE_SERVICE_HOST}" ]; then
        # if not its own IP, then add it
        if [ $(expr "$HOSTNAME" : "pxc-node${NUM}") -eq 0 ]; then
          # if not the first bootstrap node add comma
          if [ $WSREP_CLUSTER_ADDRESS != "gcomm://" ]; then
            WSREP_CLUSTER_ADDRESS="${WSREP_CLUSTER_ADDRESS},"
          fi
          # append
          # if user specifies USE_IP, use that
          if [ -n "${USE_IP}" ]; then
            WSREP_CLUSTER_ADDRESS="${WSREP_CLUSTER_ADDRESS}"${!NODE_SERVICE_HOST}
          # otherwise use DNS
          else
            WSREP_CLUSTER_ADDRESS="${WSREP_CLUSTER_ADDRESS}pxc-node${NUM}"
          fi
        fi
      fi
    done
  fi

  # WSREP_CLUSTER_ADDRESS is now complete and will be interpolated into the 
  # cluster address string (wsrep_cluster_address) in the cluster
  # configuration file, cluster.cnf
  if [ -n "$WSREP_CLUSTER_ADDRESS" -a "$WSREP_CLUSTER_ADDRESS" != "gcomm://" ]; then
    sed -i -e "s|^wsrep_cluster_address=gcomm://|wsrep_cluster_address=${WSREP_CLUSTER_ADDRESS}|" /etc/mysql/conf.d/cluster.cnf
  fi
fi

# random server ID needed
sed -i -e "s/^server\-id=.*$/server-id=${RANDOM}/" /etc/mysql/my.cnf

# finally, start mysql 
exec "$@"
