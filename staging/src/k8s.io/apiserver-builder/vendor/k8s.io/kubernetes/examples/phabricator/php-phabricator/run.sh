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

echo "MySQL host IP ${MYSQL_SERVICE_IP} port ${MYSQL_SERVICE_PORT}."
/home/www-data/phabricator/bin/config set mysql.host $MYSQL_SERVICE_IP
/home/www-data/phabricator/bin/config set mysql.port $MYSQL_SERVICE_PORT
/home/www-data/phabricator/bin/config set mysql.pass $MYSQL_PASSWORD

echo "Running storage upgrade"
/home/www-data/phabricator/bin/storage --force upgrade || exit 1

source /etc/apache2/envvars
echo "Starting Apache2"
apache2 -D FOREGROUND

