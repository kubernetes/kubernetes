#!/bin/sh
set -e

hostname="localregistry"
authhostname="auth.$hostname"

set_etc_hosts() {
	hostentry=$1
	IP=$(ifconfig eth0|grep "inet addr:"| cut -d: -f2 | awk '{ print $1}')
	echo "$IP $hostentry" >> /etc/hosts
	# TODO: Check if record already exists in /etc/hosts
}

install_ca() {
	mkdir -p $1/$hostname:$2
	cp ./nginx/ssl/registry-ca+ca.pem $1/$hostname:$2/ca.crt
	if [ "$3" != "" ]; then
		cp ./nginx/ssl/registry-$3+client-cert.pem $1/$hostname:$2/client.cert
		cp ./nginx/ssl/registry-$3+client-key.pem $1/$hostname:$2/client.key
	fi
}

install_test_certs() {
	install_ca $1 5440
	install_ca $1 5441
	install_ca $1 5442 ca
	install_ca $1 5443 noca
	install_ca $1 5444 ca
	install_ca $1 5447 ca
	# For test remove CA
	rm $1/${hostname}:5447/ca.crt
	install_ca $1 5448
}

set_etc_hosts $hostname
set_etc_hosts $authhostname

install_test_certs /etc/docker/certs.d
install_test_certs /root/.docker/tls

# Malevolent server
mkdir -p /etc/docker/certs.d/$hostname:6666
cp ./malevolent-certs/ca.pem /etc/docker/certs.d/$hostname:6666/ca.crt

# Token server
install_file ./tokenserver/certs/ca.pem $1 5555
install_file ./tokenserver/certs/ca.pem $1 5554
install_file ./tokenserver/certs/ca.pem $1 5557
install_file ./tokenserver/certs/ca.pem $1 5558
