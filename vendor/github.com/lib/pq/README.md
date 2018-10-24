# pq - A pure Go postgres driver for Go's database/sql package

[![GoDoc](https://godoc.org/github.com/lib/pq?status.svg)](https://godoc.org/github.com/lib/pq)
[![Build Status](https://travis-ci.org/lib/pq.svg?branch=master)](https://travis-ci.org/lib/pq)

## Install

	go get github.com/lib/pq

## Docs

For detailed documentation and basic usage examples, please see the package
documentation at <http://godoc.org/github.com/lib/pq>.

## Tests

`go test` is used for testing.  A running PostgreSQL server is
required, with the ability to log in.  The default database to connect
to test with is "pqgotest," but it can be overridden using environment
variables.

Example:

	PGHOST=/run/postgresql go test github.com/lib/pq

Optionally, a benchmark suite can be run as part of the tests:

	PGHOST=/run/postgresql go test -bench .

## Features

* SSL
* Handles bad connections for `database/sql`
* Scan `time.Time` correctly (i.e. `timestamp[tz]`, `time[tz]`, `date`)
* Scan binary blobs correctly (i.e. `bytea`)
* Package for `hstore` support
* COPY FROM support
* pq.ParseURL for converting urls to connection strings for sql.Open.
* Many libpq compatible environment variables
* Unix socket support
* Notifications: `LISTEN`/`NOTIFY`
* pgpass support

## Future / Things you can help with

* Better COPY FROM / COPY TO (see discussion in #181)

## Thank you (alphabetical)

Some of these contributors are from the original library `bmizerany/pq.go` whose
code still exists in here.

* Andy Balholm (andybalholm)
* Ben Berkert (benburkert)
* Benjamin Heatwole (bheatwole)
* Bill Mill (llimllib)
* Bjørn Madsen (aeons)
* Blake Gentry (bgentry)
* Brad Fitzpatrick (bradfitz)
* Charlie Melbye (cmelbye)
* Chris Bandy (cbandy)
* Chris Gilling (cgilling)
* Chris Walsh (cwds)
* Dan Sosedoff (sosedoff)
* Daniel Farina (fdr)
* Eric Chlebek (echlebek)
* Eric Garrido (minusnine)
* Eric Urban (hydrogen18)
* Everyone at The Go Team
* Evan Shaw (edsrzf)
* Ewan Chou (coocood)
* Fazal Majid (fazalmajid)
* Federico Romero (federomero)
* Fumin (fumin)
* Gary Burd (garyburd)
* Heroku (heroku)
* James Pozdena (jpoz)
* Jason McVetta (jmcvetta)
* Jeremy Jay (pbnjay)
* Joakim Sernbrant (serbaut)
* John Gallagher (jgallagher)
* Jonathan Rudenberg (titanous)
* Joël Stemmer (jstemmer)
* Kamil Kisiel (kisielk)
* Kelly Dunn (kellydunn)
* Keith Rarick (kr)
* Kir Shatrov (kirs)
* Lann Martin (lann)
* Maciek Sakrejda (uhoh-itsmaciek)
* Marc Brinkmann (mbr)
* Marko Tiikkaja (johto)
* Matt Newberry (MattNewberry)
* Matt Robenolt (mattrobenolt)
* Martin Olsen (martinolsen)
* Mike Lewis (mikelikespie)
* Nicolas Patry (Narsil)
* Oliver Tonnhofer (olt)
* Patrick Hayes (phayes)
* Paul Hammond (paulhammond)
* Ryan Smith (ryandotsmith)
* Samuel Stauffer (samuel)
* Timothée Peignier (cyberdelia)
* Travis Cline (tmc)
* TruongSinh Tran-Nguyen (truongsinh)
* Yaismel Miranda (ympons)
* notedit (notedit)
