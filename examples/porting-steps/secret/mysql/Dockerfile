FROM mysql:latest
CMD export MYSQL_ROOT_PASSWORD=$(cat /etc/mysql-password/password); /entrypoint.sh mysqld
