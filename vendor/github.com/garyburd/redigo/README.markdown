Redigo
======

[![Build Status](https://travis-ci.org/garyburd/redigo.svg?branch=master)](https://travis-ci.org/garyburd/redigo)
[![GoDoc](https://godoc.org/github.com/garyburd/redigo/redis?status.svg)](https://godoc.org/github.com/garyburd/redigo/redis)

Redigo is a [Go](http://golang.org/) client for the [Redis](http://redis.io/) database.

Features
-------

* A [Print-like](http://godoc.org/github.com/garyburd/redigo/redis#hdr-Executing_Commands) API with support for all Redis commands.
* [Pipelining](http://godoc.org/github.com/garyburd/redigo/redis#hdr-Pipelining), including pipelined transactions.
* [Publish/Subscribe](http://godoc.org/github.com/garyburd/redigo/redis#hdr-Publish_and_Subscribe).
* [Connection pooling](http://godoc.org/github.com/garyburd/redigo/redis#Pool).
* [Script helper type](http://godoc.org/github.com/garyburd/redigo/redis#Script) with optimistic use of EVALSHA.
* [Helper functions](http://godoc.org/github.com/garyburd/redigo/redis#hdr-Reply_Helpers) for working with command replies.

Documentation
-------------

- [API Reference](http://godoc.org/github.com/garyburd/redigo/redis)
- [FAQ](https://github.com/garyburd/redigo/wiki/FAQ)

Installation
------------

Install Redigo using the "go get" command:

    go get github.com/garyburd/redigo/redis

The Go distribution is Redigo's only dependency.

Related Projects
----------------

- [rafaeljusto/redigomock](https://godoc.org/github.com/rafaeljusto/redigomock) - A mock library for Redigo.
- [chasex/redis-go-cluster](https://github.com/chasex/redis-go-cluster) - A Redis cluster client implementation.
- [FZambia/go-sentinel](https://github.com/FZambia/go-sentinel) - Redis Sentinel support for Redigo
- [PuerkitoBio/redisc](https://github.com/PuerkitoBio/redisc) - Redis Cluster client built on top of Redigo

Contributing
------------

Send email to Gary Burd (address in GitHub profile) before doing any work on Redigo.

License
-------

Redigo is available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0.html).
