# go-systemd

[![Build Status](https://travis-ci.org/coreos/go-systemd.png?branch=master)](https://travis-ci.org/coreos/go-systemd)
[![godoc](https://godoc.org/github.com/coreos/go-systemd?status.svg)](http://godoc.org/github.com/coreos/go-systemd)

Go bindings to systemd. The project has several packages:

- `activation` - for writing and using socket activation from Go
- `dbus` - for starting/stopping/inspecting running services and units
- `journal` - for writing to systemd's logging service, journald
- `sdjournal` - for reading from journald by wrapping its C API
- `machine1` - for registering machines/containers with systemd
- `unit` - for (de)serialization and comparison of unit files

## Socket Activation

An example HTTP server using socket activation can be quickly set up by following this README on a Linux machine running systemd:

https://github.com/coreos/go-systemd/tree/master/examples/activation/httpserver

## Journal

Using the pure-Go `journal` package you can submit journal entries directly to systemd's journal, taking advantage of features like indexed key/value pairs for each log entry.
The `sdjournal` package provides read access to the journal by wrapping around journald's native C API; consequently it requires cgo and the journal headers to be available.

## D-Bus

The `dbus` package connects to the [systemd D-Bus API](http://www.freedesktop.org/wiki/Software/systemd/dbus/) and lets you start, stop and introspect systemd units. The API docs are here:

http://godoc.org/github.com/coreos/go-systemd/dbus

### Debugging

Create `/etc/dbus-1/system-local.conf` that looks like this:

```
<!DOCTYPE busconfig PUBLIC
"-//freedesktop//DTD D-Bus Bus Configuration 1.0//EN"
"http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd">
<busconfig>
    <policy user="root">
        <allow eavesdrop="true"/>
        <allow eavesdrop="true" send_destination="*"/>
    </policy>
</busconfig>
```

## machined

The `machine1` package allows interaction with the [systemd machined D-Bus API](http://www.freedesktop.org/wiki/Software/systemd/machined/).

## Units

The `unit` package provides various functions for working with [systemd unit files](http://www.freedesktop.org/software/systemd/man/systemd.unit.html).
