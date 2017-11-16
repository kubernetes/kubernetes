# Overview

The goal of this Go project is to consolidate all low-level
network testing "daemons" into one place. In network testing we
frequently have need of simple daemons (common/Runner) that perform
some "trivial" set of actions on a socket.

# Usage

* A package for each general area that is being tested, for example
  `nat/` will contain Runners that test various NAT features.
* Every runner should be registered via `main.go:makeRunnerMap()`.
* Runners receive a JSON options structure as to their configuration. `Run()`
  should return the disposition of the test.

Runners can be executed into two different ways, either through the
the command-line or via an HTTP request:

## Command-line

````
$ ./net -runner <runner> -options <json>
./net \
  -runner nat-closewait-client \
  -options '{"RemoteAddr":"127.0.0.1:9999"}'
````

## HTTP server
````
$ ./net --serve :8889
$ curl -v -X POST localhost:8889/run/nat-closewait-server \
  -d '{"LocalAddr":"127.0.0.1:9999"}'
````


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/test/images/net/README.md?pixel)]()
