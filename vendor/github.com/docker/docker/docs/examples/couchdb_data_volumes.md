<!--[metadata]>
+++
title = "Dockerizing a CouchDB service"
description = "Sharing data between 2 couchdb databases"
keywords = ["docker, example, package installation, networking, couchdb,  data volumes"]
[menu.main]
parent = "smn_applied"
+++
<![end-metadata]-->

# Dockerizing a CouchDB service

> **Note**: 
> - **If you don't like sudo** then see [*Giving non-root
>   access*](/installation/binaries/#giving-non-root-access)

Here's an example of using data volumes to share the same data between
two CouchDB containers. This could be used for hot upgrades, testing
different versions of CouchDB on the same data, etc.

## Create first database

Note that we're marking `/var/lib/couchdb` as a data volume.

    $ COUCH1=$(docker run -d -p 5984 -v /var/lib/couchdb shykes/couchdb:2013-05-03)

## Add data to the first database

We're assuming your Docker host is reachable at `localhost`. If not,
replace `localhost` with the public IP of your Docker host.

    $ HOST=localhost
    $ URL="http://$HOST:$(docker port $COUCH1 5984 | grep -o '[1-9][0-9]*$')/_utils/"
    $ echo "Navigate to $URL in your browser, and use the couch interface to add data"

## Create second database

This time, we're requesting shared access to `$COUCH1`'s volumes.

    $ COUCH2=$(docker run -d -p 5984 --volumes-from $COUCH1 shykes/couchdb:2013-05-03)

## Browse data on the second database

    $ HOST=localhost
    $ URL="http://$HOST:$(docker port $COUCH2 5984 | grep -o '[1-9][0-9]*$')/_utils/"
    $ echo "Navigate to $URL in your browser. You should see the same data as in the first database"'!'

Congratulations, you are now running two Couchdb containers, completely
isolated from each other *except* for their data.
