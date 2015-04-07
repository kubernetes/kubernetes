#!/bin/bash

echo "MySQL host IP ${MYSQL_SERVICE_IP} port ${MYSQL_SERVICE_PORT}."
/home/www-data/phabricator/bin/config set mysql.host $MYSQL_SERVICE_IP
/home/www-data/phabricator/bin/config set mysql.port $MYSQL_SERVICE_PORT
/home/www-data/phabricator/bin/config set mysql.pass $MYSQL_PASSWORD

echo "Running storage upgrade"
/home/www-data/phabricator/bin/storage --force upgrade || exit 1

source /etc/apache2/envvars
echo "Starting Apache2"
apache2 -D FOREGROUND

