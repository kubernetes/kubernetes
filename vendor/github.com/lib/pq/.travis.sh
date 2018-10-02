#!/bin/bash

set -eu

client_configure() {
	sudo chmod 600 $PQSSLCERTTEST_PATH/postgresql.key
}

pgdg_repository() {
	local sourcelist='sources.list.d/postgresql.list'

	curl -sS 'https://www.postgresql.org/media/keys/ACCC4CF8.asc' | sudo apt-key add -
	echo deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main $PGVERSION | sudo tee "/etc/apt/$sourcelist"
	sudo apt-get -o Dir::Etc::sourcelist="$sourcelist" -o Dir::Etc::sourceparts='-' -o APT::Get::List-Cleanup='0' update
}

postgresql_configure() {
	sudo tee /etc/postgresql/$PGVERSION/main/pg_hba.conf > /dev/null <<-config
		local     all         all                               trust
		hostnossl all         pqgossltest 127.0.0.1/32          reject
		hostnossl all         pqgosslcert 127.0.0.1/32          reject
		hostssl   all         pqgossltest 127.0.0.1/32          trust
		hostssl   all         pqgosslcert 127.0.0.1/32          cert
		host      all         all         127.0.0.1/32          trust
		hostnossl all         pqgossltest ::1/128               reject
		hostnossl all         pqgosslcert ::1/128               reject
		hostssl   all         pqgossltest ::1/128               trust
		hostssl   all         pqgosslcert ::1/128               cert
		host      all         all         ::1/128               trust
	config

	xargs sudo install -o postgres -g postgres -m 600 -t /var/lib/postgresql/$PGVERSION/main/ <<-certificates
		certs/root.crt
		certs/server.crt
		certs/server.key
	certificates

	sort -VCu <<-versions ||
		$PGVERSION
		9.2
	versions
	sudo tee -a /etc/postgresql/$PGVERSION/main/postgresql.conf > /dev/null <<-config
		ssl_ca_file   = 'root.crt'
		ssl_cert_file = 'server.crt'
		ssl_key_file  = 'server.key'
	config

	echo 127.0.0.1 postgres | sudo tee -a /etc/hosts > /dev/null

	sudo service postgresql restart
}

postgresql_install() {
	xargs sudo apt-get -y -o Dpkg::Options::='--force-confdef' -o Dpkg::Options::='--force-confnew' install <<-packages
		postgresql-$PGVERSION
		postgresql-server-dev-$PGVERSION
		postgresql-contrib-$PGVERSION
	packages
}

postgresql_uninstall() {
	sudo service postgresql stop
	xargs sudo apt-get -y --purge remove <<-packages
		libpq-dev
		libpq5
		postgresql
		postgresql-client-common
		postgresql-common
	packages
	sudo rm -rf /var/lib/postgresql
}

megacheck_install() {
	# Megacheck is Go 1.6+, so skip if Go 1.5.
	if [[ "$(go version)" =~ "go1.5" ]]
	then
		echo "megacheck not supported, skipping installation"
		return 0
	fi
	# Lock megacheck version at $MEGACHECK_VERSION to prevent spontaneous
	# new error messages in old code.
	go get -d honnef.co/go/tools/...
	git -C $GOPATH/src/honnef.co/go/tools/ checkout $MEGACHECK_VERSION
	go install honnef.co/go/tools/cmd/megacheck
	megacheck --version
}

golint_install() {
	# Golint is Go 1.6+, so skip if Go 1.5.
	if [[ "$(go version)" =~ "go1.5" ]]
	then
		echo "golint not supported, skipping installation"
		return 0
	fi
	go get github.com/golang/lint/golint
}

$1
