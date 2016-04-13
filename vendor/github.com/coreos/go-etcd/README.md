# go-etcd

[![GoDoc](https://godoc.org/github.com/coreos/go-etcd/etcd?status.png)](https://godoc.org/github.com/coreos/go-etcd/etcd)

# DEPRECATED

etcd now has an [official Go client](https://github.com/coreos/etcd/tree/master/client), which has
a nicer API and better support.

We strongly suggest you use the official Go client instead of go-etcd in your new projects.
For existing projects, we suggest you migrate to the official Go client.

## Usage

The current version of go-etcd supports etcd v2.0+, if you need support for etcd v0.4 please use go-etcd from the [release-0.4](https://github.com/coreos/go-etcd/tree/release-0.4) branch.

```
package main

import (
    "log"

    "github.com/coreos/go-etcd/etcd"
)

func main() {
    machines := []string{"http://127.0.0.1:2379"}
    client := etcd.NewClient(machines)

    if _, err := client.Set("/foo", "bar", 0); err != nil {
        log.Fatal(err)
    }
}
```

## Install

```bash
go get github.com/coreos/go-etcd/etcd
```

## Caveat

1. go-etcd always talks to one member if the member works well. This saves socket resources, and improves efficiency for both client and server side. It doesn't hurt the consistent view of the client because each etcd member has data replication.

2. go-etcd does round-robin rotation when it fails to connect the member in use. For example, if the member that go-etcd connects to is hard killed, go-etcd will fail on the first attempt with the killed member, and succeed on the second attempt with another member. The default CheckRetry function does 2*machine_number retries before returning error.

3. The default transport in go-etcd sets 1s DialTimeout and 1s TCP keepalive period. A customized transport could be set by calling `Client.SetTransport`.

4. Default go-etcd cannot handle the case that the remote server is SIGSTOPed now. TCP keepalive mechanism doesn't help in this scenario because operating system may still send TCP keep-alive packets. We will improve it, but it is not in high priority because we don't see a solid real-life case which server is stopped but connection is alive.

5. go-etcd is not thread-safe, and it may have race when switching member or updating cluster.

6. go-etcd cannot detect whether the member in use is healthy when doing read requests. If the member is isolated from the cluster, go-etcd may retrieve outdated data. We will improve this.

## License

See LICENSE file.
