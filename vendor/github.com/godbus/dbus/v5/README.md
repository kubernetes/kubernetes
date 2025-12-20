![Build Status](https://github.com/godbus/dbus/workflows/Go/badge.svg)

dbus
----

dbus is a simple library that implements native Go client bindings for the
D-Bus message bus system.

### Features

* Complete native implementation of the D-Bus message protocol
* Go-like API (channels for signals / asynchronous method calls, Goroutine-safe connections)
* Subpackages that help with the introspection / property interfaces

### Installation

This packages requires Go 1.20 or later. It can be installed by running the command below:

```
go get github.com/godbus/dbus/v5
```

### Usage

The complete package documentation and some simple examples are available at
[pkg.go.dev](https://pkg.go.dev/github.com/godbus/dbus/v5). Also, the
[_examples](https://github.com/godbus/dbus/tree/master/_examples) directory
gives a short overview over the basic usage. 

#### Projects using godbus
- [fyne](https://github.com/fyne-io/fyne) a cross platform GUI in Go inspired by Material Design.
- [fynedesk](https://github.com/fyne-io/fynedesk) a full desktop environment for Linux/Unix using Fyne.
- [go-bluetooth](https://github.com/muka/go-bluetooth) provides a bluetooth client over bluez dbus API.
- [iwd](https://github.com/shibumi/iwd) go bindings for the internet wireless daemon "iwd".
- [notify](https://github.com/esiqveland/notify) provides desktop notifications over dbus into a library.
- [playerbm](https://github.com/altdesktop/playerbm) a bookmark utility for media players.
- [rpic](https://github.com/stephenhu/rpic) lightweight web app and RESTful API for managing a Raspberry Pi

Please note that the API is considered unstable for now and may change without
further notice.

### License

go.dbus is available under the Simplified BSD License; see LICENSE for the full
text.

Nearly all of the credit for this library goes to github.com/guelfey/go.dbus.
