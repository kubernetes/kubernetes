Redigo
======

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

Contributing
------------

Contributions are welcome. 

Before writing code, send mail to gary@beagledreams.com to discuss what you
plan to do. This gives me a chance to validate the design, avoid duplication of
effort and ensure that the changes fit the goals of the project. Do not start
the discussion with a pull request. 

License
-------

Redigo is available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0.html).
