# Test Server

This package contains a server for the [Autobahn WebSockets Test Suite](http://autobahn.ws/testsuite).

To test the server, run

    go run server.go

and start the client test driver

    wstest -m fuzzingclient -s fuzzingclient.json

When the client completes, it writes a report to reports/clients/index.html.
