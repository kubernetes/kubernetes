# go-systemd

Go bindings to systemd. The project has three packages:

- activation - for writing and using socket activation from Go
- journal - for writing to systemd's logging service, journal
- dbus - for starting/stopping/inspecting running services and units

Go docs for the entire project are here:

http://godoc.org/github.com/coreos/go-systemd

## Socket Activation

An example HTTP server using socket activation can be quickly setup by
following this README on a Linux machine running systemd:

https://github.com/coreos/go-systemd/tree/master/examples/activation/httpserver

## Journal

Using this package you can submit journal entries directly to systemd's journal taking advantage of features like indexed key/value pairs for each log entry.

## D-Bus

The D-Bus API lets you start, stop and introspect systemd units. The API docs are here:

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
