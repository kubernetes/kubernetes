# Go Wrapper for ZFS #

Simple wrappers for ZFS command line tools.

[![GoDoc](https://godoc.org/github.com/mistifyio/go-zfs?status.svg)](https://godoc.org/github.com/mistifyio/go-zfs)

## Requirements ##

You need a working ZFS setup.  To use on Ubuntu 14.04, setup ZFS:

    sudo apt-get install python-software-properties
    sudo apt-add-repository ppa:zfs-native/stable
    sudo apt-get update
    sudo apt-get install ubuntu-zfs libzfs-dev

Developed using Go 1.3, but currently there isn't anything 1.3 specific. Don't use Ubuntu packages for Go, use http://golang.org/doc/install

Generally you need root privileges to use anything zfs related.

## Status ##

This has been only been tested on Ubuntu 14.04

In the future, we hope to work directly with libzfs.

# Hacking #

The tests have decent examples for most functions.

```go
//assuming a zpool named test
//error handling omitted


f, err := zfs.CreateFilesystem("test/snapshot-test", nil)
ok(t, err)

s, err := f.Snapshot("test", nil)
ok(t, err)

// snapshot is named "test/snapshot-test@test"

c, err := s.Clone("test/clone-test", nil)

err := c.Destroy()
err := s.Destroy()
err := f.Destroy()

```

# Contributing #

See the [contributing guidelines](./CONTRIBUTING.md)

