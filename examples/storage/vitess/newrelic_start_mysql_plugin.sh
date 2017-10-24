#!/bin/bash

# Copyright 2017 Google Inc.
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

NEWRELIC_LICENSE_KEY=$1
AGENT_NAME=$2

export TERM=screen

apt-get install openjdk-7-jdk -y

yes | LICENSE_KEY=$NEWRELIC_LICENSE_KEY bash -c "$(curl -sSL https://download.newrelic.com/npi/release/install-npi-linux-debian-x64.sh)"

vttablet_dir=`ls /vt/vtdataroot | grep -o vt_.* | tr '\r' ' ' | xargs`
mysql_sock=/vt/vtdataroot/${vttablet_dir}/mysql.sock
mysql -S $mysql_sock -u vt_dba -e "UPDATE mysql.user SET Password=PASSWORD('password') WHERE User='root'; FLUSH PRIVILEGES;"
mysql -S $mysql_sock -u vt_dba -e "create user newrelic@localhost identified by 'password'"
mysql -S $mysql_sock -u vt_dba -e "create user newrelic@127.0.0.1 identified by 'password'"
mysql -S $mysql_sock -u root --password=password -e "grant all on vt_test_keyspace.* to newrelic@localhost"
mysql -S $mysql_sock -u root --password=password -e "grant all on vt_test_keyspace.* to newrelic@127.0.0.1"
mysql -S $mysql_sock -u vt_dba -e "flush privileges;"


cd ~/newrelic-npi
./npi set user root
./npi config set license_key=$NEWRELIC_LICENSE_KEY
./npi fetch com.newrelic.plugins.mysql.instance -y
./npi prepare com.newrelic.plugins.mysql.instance -n

echo "{
  \"agents\": [
    {
      \"name\"    : \"$AGENT_NAME\",
      \"host\"    : \"localhost\",
      \"metrics\" : \"status,newrelic\",
      \"user\"    : \"root\",
      \"passwd\"  : \"password\",
      \"properties\": \"mysql?socket=$mysql_sock\"
    }
  ]
}" > ~/newrelic-npi/plugins/com.newrelic.plugins.mysql.instance/newrelic_mysql_plugin-2.0.0/config/plugin.json

sleep 10
./npi add-service com.newrelic.plugins.mysql.instance --start
sleep 10
./npi start com.newrelic.plugins.mysql.instance
