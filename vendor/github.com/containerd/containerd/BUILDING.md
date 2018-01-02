# Build containerd from source

This guide is useful if you intend to contribute on containerd. Thanks for your
effort. Every contribution is very appreciated.

## Build the development environment

In first you need to setup your Go development environment. You can follow this
guideline [How to write go code](https://golang.org/doc/code.html) and at the
end you need to have `GOPATH` and `GOROOT` set in your environment.

Current containerd requires Go 1.9.x or above.

At this point you can use `go` to checkout `containerd` in your `GOPATH`:

```sh
go get github.com/containerd/containerd
```

`containerd` uses [Btrfs](https://en.wikipedia.org/wiki/Btrfs) it means that you
need to satisfy this dependencies in your system:

* CentOS/Fedora: `yum install btrfs-progs-devel`
* Debian/Ubuntu: `apt-get install btrfs-tools`

At this point you are ready to build `containerd` yourself.

## In your local environment

`containerd` uses `make` to create a repeatable build flow. It means that you
can run:

```sudo
make
```

This is going to build all the binaries provided by this project in the `./bin`
directory.

You can move them in your global path with:

```sudo
sudo make install
```

## Via Docker Container

### Build containerd

You can build `containerd` via Docker container. You can build an image from
this `Dockerfile`:

```
FROM golang

RUN apt-get update && \
    apt-get install btrfs-tools
```

Let's suppose that you built an image called `containerd/build` and you are into
the containerd root directory you can run the following command:

```sh
docker run -it --rm \
    -v ${PWD}:/go/src/github.com/containerd/containerd \
    -e GOPATH=/go \
    -w /go/src/github.com/containerd/containerd containerd/build make
```

At this point you can find your binaries in the `./bin` directory in your host.
You can move the binaries in your `$PATH` with the command:

```sh
sudo make install
```

### Build runc and containerd

To have complete core container runtime, you will need both `containerd` and `runc`. It is possible to build both of these via Docker container.

You can use `go` to checkout `runc` in your `GOPATH`:

```sh
go get github.com/opencontainers/runc
```

We can build an image from this `Dockerfile`

```sh
FROM golang

RUN apt-get update && \
    apt-get install -y btrfs-tools libapparmor-dev libseccomp-dev

```

In our Docker container we will use a specific `runc` build which includes [seccomp](https://en.wikipedia.org/wiki/seccomp) and [apparmor](https://en.wikipedia.org/wiki/AppArmor) support. Hence why our Dockerfile includes these dependencies: `libapparmor-dev` `libseccomp-dev`.

Let's suppose you build an image called `containerd/build` from the above Dockerfile. You can run the following command:

```sh
docker run -it --privileged \
    -v /var/lib/containerd \
    -v ${GOPATH}/src/github.com/opencontainers/runc:/go/src/github.com/opencontainers/runc \
    -v ${GOPATH}/src/github.com/containerd/containerd:/go/src/github.com/containerd/containerd \
    -e GOPATH=/go
    -w /go/src/github.com/containerd/containerd containerd/build sh
```

This mounts both `runc` and `containerd` repositories in our Docker container.

From within our Docker container let's build `containerd`

```sh
cd /go/src/github.com/containerd/containerd
make && make install
```

These binaries can be found in the `./bin` directory in your host.
`make install` will move the binaries in your `$PATH`.

Next, let's build `runc`

```sh
cd /go/src/github.com/opencontainers/runc
make BUILDTAGS='seccomp apparmor' && make install
```

When working with `ctr`, the containerd CLI we just built, don't forget to start `containerd`!

```sh
containerd --config config.toml
```

# Testing containerd

During the automated CI the unit tests and integration tests are run as part of the PR validation. As a developer you can run these tests locally by using any of the following `Makefile` targets:
 - `make test`: run all non-integration tests that do not require `root` privileges
 - `make root-test`: run all non-integration tests which require `root`
 - `make integration`: run all tests, including integration tests and those which require `root`
 - `make integration-parallel`: run all tests (integration and root-required included) in parallel mode

To execute a specific test or set of tests you can use the `go test` capabilities
without using the `Makefile` targets. The following examples show how to specify a test
name and also how to use the flag directly against `go test` to run root-requiring tests.

```sh
# run the test <TEST_NAME>:
go test	-v -run "<TEST_NAME>" .
# enable the root-requiring tests:
go test -v -run . -test.root
```

Example output from directly running `go test` to execute the `TestContainerList` test:
```sh
sudo go test -v -run "TestContainerList" . -test.root
INFO[0000] running tests against containerd revision=f2ae8a020a985a8d9862c9eb5ab66902c2888361 version=v1.0.0-beta.2-49-gf2ae8a0
=== RUN   TestContainerList
--- PASS: TestContainerList (0.00s)
PASS
ok  	github.com/containerd/containerd	4.778s
```

## Additional tools

### containerd-stress
In addition to `go test`-based testing executed via the `Makefile` targets, the `containerd-stress` tool is available and built with the `all` or `binaries` targets and installed during `make install`.

With this tool you can stress a running containerd daemon for a specified period of time, selecting a concurrency level to generate stress against the daemon. The following command is an example of having five workers running for two hours against a default containerd gRPC socket address:

```sh
containerd-stress -c 5 -t 120
```

For more information on this tool's options please run `containerd-stress --help`.

### bucketbench
[Bucketbench](https://github.com/estesp/bucketbench) is an external tool which can be used to drive load against a container runtime, specifying a particular set of lifecycle operations to run with a specified amount of concurrency. Bucketbench is more focused on generating performance details than simply inducing load against containerd.

Bucketbench differs from the `containerd-stress` tool in a few ways:
 - Bucketbench has support for testing the Docker engine, the `runc` binary, and containerd 0.2.x (via `ctr`) and 1.0 (via the client library) branches.
 - Bucketbench is driven via configuration file that allows specifying a list of lifecycle operations to execute. This can be used to generate detailed statistics per-command (e.g. start, stop, pause, delete).
 - Bucketbench generates detailed reports and timing data at the end of the configured test run.

More details on how to install and run `bucketbench` are available at the [GitHub project page](https://github.com/estesp/bucketbench).
