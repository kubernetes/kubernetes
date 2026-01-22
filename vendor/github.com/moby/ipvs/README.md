# ipvs - networking for containers

![Test](https://github.com/moby/ipvs/workflows/Test/badge.svg) [![GoDoc](https://godoc.org/github.com/moby/ipvs?status.svg)](https://godoc.org/github.com/moby/ipvs) [![Go Report Card](https://goreportcard.com/badge/github.com/moby/ipvs)](https://goreportcard.com/report/github.com/moby/ipvs)

ipvs provides a native Go implementation for communicating with IPVS kernel module using a netlink socket.


#### Using ipvs

```go
import (
	"log"

	"github.com/moby/ipvs"
)

func main() {
	handle, err := ipvs.New("")
	if err != nil {
		log.Fatalf("ipvs.New: %s", err)
	}
	svcs, err := handle.GetServices()
	if err != nil {
		log.Fatalf("handle.GetServices: %s", err)
	}
}
```

## Contributing

Want to hack on ipvs? [Docker's contributions guidelines](https://github.com/docker/docker/blob/master/CONTRIBUTING.md) apply.

## Copyright and license

Copyright 2015 Docker, inc. Code released under the [Apache 2.0 license](LICENSE).
