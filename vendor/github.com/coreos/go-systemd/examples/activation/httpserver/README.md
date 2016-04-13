## socket activated http server

This is a simple example of using socket activation with systemd to serve a
simple HTTP server on http://127.0.0.1:8076

To try it out `go get` the httpserver and run it under the systemd-activate helper

```
export GOPATH=`pwd`
go get github.com/coreos/go-systemd/examples/activation/httpserver
sudo /usr/lib/systemd/systemd-activate -l 127.0.0.1:8076 ./bin/httpserver
```

Then curl the URL and you will notice that it starts up:

```
curl 127.0.0.1:8076
hello socket activated world!
```
