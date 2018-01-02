#!/bin/sh
set -e

hostname="localregistry"
installdir="$1"

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
	install_ca $1 5600
}

install_ca_file() {
	mkdir -p $2
	cp $1 $2/ca.crt
}

append_ca_file() {
	mkdir -p $2
	cat $1 >> $2/ca.crt
}

install_test_certs $installdir

# Malevolent server
install_ca_file ./malevolent-certs/ca.pem $installdir/$hostname:6666

# Token server
install_ca_file ./tokenserver/certs/ca.pem $installdir/$hostname:5554
install_ca_file ./tokenserver/certs/ca.pem $installdir/$hostname:5555
install_ca_file ./tokenserver/certs/ca.pem $installdir/$hostname:5557
install_ca_file ./tokenserver/certs/ca.pem $installdir/$hostname:5558
append_ca_file ./tokenserver/certs/ca.pem $installdir/$hostname:5600

