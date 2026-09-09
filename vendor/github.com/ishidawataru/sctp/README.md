Stream Control Transmission Protocol (SCTP)
----

[![Build Status](https://travis-ci.org/ishidawataru/sctp.svg?branch=master)](https://travis-ci.org/ishidawataru/sctp/builds)

Examples
----

See `example/sctp.go`

```go
$ cd example
$ go build
$ # run example SCTP server
$ ./example -server -port 1000 -ip 10.10.0.1,10.20.0.1
$ # run example SCTP client
$ ./example -port 1000 -ip 10.10.0.1,10.20.0.1
```
