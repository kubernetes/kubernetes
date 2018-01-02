# Chat Example

This application shows how to use use the
[websocket](https://github.com/gorilla/websocket) package and
[jQuery](http://jquery.com) to implement a simple web chat application.

## Running the example

The example requires a working Go development environment. The [Getting
Started](http://golang.org/doc/install) page describes how to install the
development environment.

Once you have Go up and running, you can download, build and run the example
using the following commands.

    $ go get github.com/gorilla/websocket
    $ cd `go list -f '{{.Dir}}' github.com/gorilla/websocket/examples/chat`
    $ go run *.go

To use the chat example, open http://localhost:8080/ in your browser.
