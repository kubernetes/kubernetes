[![Build Status](https://travis-ci.org/godbus/dbus.svg?branch=master)](https://travis-ci.org/godbus/dbus)

dbus
----

dbus is a simple library that implements native Go client bindings for the
D-Bus message bus system.

### Features

* Complete native implementation of the D-Bus message protocol
* Go-like API (channels for signals / asynchronous method calls, Goroutine-safe connections)
* Subpackages that help with the introspection / property interfaces

### Installation

This packages requires Go 1.7. If you installed it and set up your GOPATH, just run:

```
go get github.com/godbus/dbus
```

If you want to use the subpackages, you can install them the same way.

### Usage

The complete package documentation and some simple examples are available at
[godoc.org](http://godoc.org/github.com/godbus/dbus). Also, the
[_examples](https://github.com/godbus/dbus/tree/master/_examples) directory
gives a short overview over the basic usage. 

#### Projects using godbus
- [notify](https://github.com/esiqveland/notify) provides desktop notifications over dbus into a library.
- [go-bluetooth](https://github.com/muka/go-bluetooth) provides a bluetooth client over bluez dbus API.

Please note that the API is considered unstable for now and may change without
further notice.

### License

go.dbus is available under the Simplified BSD License; see LICENSE for the full
text.

Nearly all of the credit for this library goes to github.com/guelfey/go.dbus.
