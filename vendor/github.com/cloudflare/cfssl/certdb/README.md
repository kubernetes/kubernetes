# certdb usage

Using a database enables additional functionality for existing commands when a
db config is provided:

 - `sign` and `gencert` add a certificate to the certdb after signing it
 - `serve` enables database functionality for the sign and revoke endpoints

A database is required for the following:

 - `revoke` marks certificates revoked in the database with an optional reason
 - `ocsprefresh` refreshes the table of cached OCSP responses
 - `ocspdump` outputs cached OCSP responses in a concatenated base64-encoded format

## Setup/Migration

This directory stores [goose](https://bitbucket.org/liamstask/goose/) db migration scripts for various DB backends.
Currently supported:
 - MySQL in mysql
 - PostgreSQL in pg
 - SQLite in sqlite

### Get goose

    go get bitbucket.org/liamstask/goose/cmd/goose

### Use goose to start and terminate a MySQL DB
To start a MySQL using goose:

    goose -path $GOPATH/src/github.com/cloudflare/cfssl/certdb/mysql up

To tear down a MySQL DB using goose

    goose -path $GOPATH/src/github.com/cloudflare/cfssl/certdb/mysql down

Note: the administration of MySQL DB is not included. We assume
the databases being connected to are already created and access control
is properly handled.

### Use goose to start and terminate a PostgreSQL DB
To start a PostgreSQL using goose:

    goose -path $GOPATH/src/github.com/cloudflare/cfssl/certdb/pg up

To tear down a PostgreSQL DB using goose

    goose -path $GOPATH/src/github.com/cloudflare/cfssl/certdb/pg down

Note: the administration of PostgreSQL DB is not included. We assume
the databases being connected to are already created and access control
is properly handled.

### Use goose to start and terminate a SQLite DB
To start a SQLite DB using goose:

    goose -path $GOPATH/src/github.com/cloudflare/cfssl/certdb/sqlite up

To tear down a SQLite DB using goose

    goose -path $GOPATH/src/github.com/cloudflare/cfssl/certdb/sqlite down

## CFSSL Configuration

Several cfssl commands take a -db-config flag. Create a file with a
JSON dictionary:

    {"driver":"sqlite3","data_source":"certs.db"}

or

    {"driver":"postgres","data_source":"postgres://user:password@host/db"}
 
or

    {"driver":"mysql","data_source":"user:password@tcp(hostname:3306)/db?parseTime=true"}
