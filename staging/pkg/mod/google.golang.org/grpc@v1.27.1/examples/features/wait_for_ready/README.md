# Wait for ready example

This example shows how to enable "wait for ready" in RPC calls.

This code starts a server with a 2 seconds delay. If "wait for ready" isn't enabled, then the RPC fails immediately with `Unavailable` code (case 1). If "wait for ready" is enabled, then the RPC waits for the server. If context dies before the server is available, then it fails with `DeadlineExceeded` (case 3). Otherwise it succeeds (case 2).

## Run the example

```
go run main.go
```
