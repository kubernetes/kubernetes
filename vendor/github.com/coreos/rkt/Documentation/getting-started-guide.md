# Getting Started with rkt

The following guide will show you how to build and run a self-contained Go app using rkt, the reference implementation of the [App Container Specification](https://github.com/appc/spec).
If you're not on Linux, you should do all of this inside [the rkt Vagrant](https://github.com/coreos/rkt/blob/master/Documentation/trying-out-rkt.md#rkt-using-vagrant).

## Create a hello go application

```go
package main

import (
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("request from %v\n", r.RemoteAddr)
		w.Write([]byte("hello\n"))
	})
	log.Fatal(http.ListenAndServe(":5000", nil))
}
```

### Build a statically linked Go binary

Next we need to build our application.
We are going to statically link our app so we can ship an App Container Image with no external dependencies.

With [Go 1.4](https://github.com/golang/go/issues/9344#issuecomment-69944514):

```
$ CGO_ENABLED=0 GOOS=linux go build -o hello -a -installsuffix cgo .
```

or with Go 1.5:

```
$ CGO_ENABLED=0 GOOS=linux go build -o hello -a -tags netgo -ldflags '-w' .
```

Before proceeding, verify that the produced binary is statically linked:

```
$ file hello
hello: ELF 64-bit LSB executable, x86-64, version 1 (SYSV), statically linked, not stripped
$ ldd hello
	not a dynamic executable
```

## Create the image

To create the image, we can use `acbuild`, which can be downloaded via one of the [releases in the acbuild repository](https://github.com/appc/acbuild/releases)

The following commands will create an ACI containing our application and important metadata.

```bash
acbuild begin
acbuild set-name example.com/hello
acbuild copy hello /bin/hello
acbuild set-exec /bin/hello
acbuild port add www tcp 5000
acbuild label add version 0.0.1
acbuild label add arch amd64
acbuild label add os linux
acbuild annotation add authors "Carly Container <carly@example.com>"
acbuild write hello-0.0.1-linux-amd64.aci
acbuild end
```

## Run

### Launch a local application image

```
# rkt --insecure-options=image run hello-0.0.1-linux-amd64.aci
```

Note that `--insecure-options=image` is required because, by default, rkt expects our images to be signed.
See the [Signing and Verification Guide](https://github.com/coreos/rkt/blob/master/Documentation/signing-and-verification-guide.md) for more details.

At this point our hello app is running and ready to handle HTTP requests.

You can also [run rkt as a daemon](https://github.com/coreos/rkt/blob/master/Documentation/subcommands/run.md#run-rkt-as-a-daemon).

### Test with curl

By default, rkt will assign the running container an IP address. Use `rkt list` to discover what it is:

```
# rkt list
UUID		APP	IMAGE NAME		STATE	NETWORKS
885876b0	hello	example.com/hello:0.0.1	running	default:ip4=172.16.28.2
```

Then you can `curl` that IP on port 5000:

```
$ curl 172.16.28.2:5000
hello
```
