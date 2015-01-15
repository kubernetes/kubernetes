#!/bin/sh
# We only need the XDR parser for gss/proxy.

mkdir -p bin

echo proxy-client
go build -o bin/proxy-client cmd/proxy-client/proxy-client.go
echo proxy-server
go build -o bin/proxy-server cmd/proxy-server/proxy-server.go
# We need development files for krb5 1.12 or newer for gss.
if pkg-config krb5-gssapi 2> /dev/null ; then
	echo gss-client
	go build -o bin/gss-client cmd/gss-client/gss-client.go
	echo gss-server
	go build -o bin/gss-server cmd/gss-server/gss-server.go
	echo www-authenticate
	go build -o bin/www-authenticate cmd/www-authenticate/www-authenticate.go
fi
