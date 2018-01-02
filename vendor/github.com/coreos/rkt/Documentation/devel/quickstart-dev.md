# Preparing development environment and first rkt build

This is an example configuration and quick start guide for the installation of rkt from source on Ubuntu 16.04 GNOME. For a detailed developer's reference, see [the rkt hacking guide][rkt-hacking].

## Get rkt repo and install dependencies

In this example ~/Repos is a personal workspace where all repos are stored

```sh
$ mkdir ~/Repos && cd ~/Repos
$ mkdir -p ~/.local/gopath/src/github.com/coreos
$ sudo apt-get install git
$ git -C ~/.local/gopath/src/github.com/coreos clone https://github.com/coreos/rkt.git
$ ln -s ~/.local/gopath/src/github.com/coreos/rkt rkt
```

On a fresh system installation, few additional software packages are needed to correctly build rkt:

```sh
$ sudo ~/Repos/rkt/scripts/install-deps-debian-sid.sh
```

See also [the dependencies page][rkt-dependencies].

## Installing Go Programming Language for a single-user

```
$ cd ~/Downloads
$ wget https://storage.googleapis.com/golang/go1.6.1.linux-amd64.tar.gz
$ tar -xvf go1.6.1.linux-amd64.tar.gz
$ mv go ~/.local
```

Add GO variables to .bashrc file:

```sh
export PATH=~/.local/bin:~/.local/go/bin:$PATH
export GOPATH=~/.local/gopath
export GOROOT=~/.local/go
```

## Install ccache (optional step)

Ccache can save a lot of time. If you build a kernel once, most of the compiled code can just be taken from the cache.
Ccache can be configured in a few easy steps:

```sh
$ sudo apt-get install ccache
$ ccache --max-size=10G
$ sudo ln -s /usr/bin/ccache /usr/local/bin/gcc
```

The maximum cache size is 10GB now (the default value is too small to cache kernel compilation).

## Building rkt

Run the autogen and configure commands with the relevant arguments, for example (kvm as flavor):

```sh
$ cd ~/Repos/rkt
$ ./autogen.sh && ./configure --enable-functional-tests --enable-incremental-build --with-stage1-flavors=kvm
```

Now build rkt with:

```sh
$ make V=2 -j
```

REMEMBER: If you want to test somebody else's changes:

```sh
$ git checkout <branch>
$ make clean
$ ./autogen.sh && ./configure <proper arguments>
```

## A few useful commands

### Just build and run tests:

```sh
$ ./tests/build-and-run-tests.sh -f kvm
```

### Run only functional tests after build:

```sh
$ make functional-check
```

### Check only one test:

```sh
$ make functional-check GO_TEST_FUNC_ARGS='-run TEST_NAME_HERE'
```

See more in [the tests readme][rkt-tests-readme] page.

### Simple usage of rkt container (run, exit, remove):

```sh
$ sudo ./build-rkt-*/bin/rkt run --insecure-options=image --interactive docker://busybox
$ exit
$ sudo ./build-rkt-*/bin/rkt gc --grace-period=0
```

### Remove all network interfaces created by rkt:

```sh
for link in $(ip link | grep rkt | cut -d':' -f2 | cut -d'@' -f1);
    sudo ip link del "${link}"
done
```

### Simplify changes in go files, before commit:

```sh
gofmt -s -w file.go
```

[rkt-hacking]: ../hacking.md
[rkt-dependencies]: ../dependencies.md
[rkt-tests-readme]: ../../tests/README.md
