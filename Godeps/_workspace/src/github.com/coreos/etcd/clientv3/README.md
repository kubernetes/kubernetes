# etcd/clientv3

[![Godoc](http://img.shields.io/badge/go-documentation-blue.svg?style=flat-square)](https://godoc.org/github.com/coreos/etcd/clientv3)

`etcd/clientv3` is the official Go etcd client for v3.

## Install

```bash
go get github.com/coreos/etcd/clientv3
```

## Get started

Create client using `clientv3.New`:

```go
cli, err := clientv3.New(clientv3.Config{
	Endpoints:   []string{"localhost:2379", "localhost:22379", "localhost:32379"},
	DialTimeout: 5 * time.Second,
})
if err != nil {
	// handle error!
}
defer cli.Close()
```

etcd v3 uses [`gRPC`](http://www.grpc.io) for remote procedure calls. And `clientv3` uses
[`grpc-go`](https://github.com/grpc/grpc-go) to connect to etcd. Make sure to close the client after using it. 
If the client is not closed, the connection will have leaky goroutines. To specify client request timeout,
pass `context.WithTimeout` to APIs:

```go
ctx, cancel := context.WithTimeout(context.Background(), timeout)
resp, err := kvc.Put(ctx, "sample_key", "sample_value")
cancel()
if err != nil {
    // handle error!
}
// use the response
```

etcd uses go's `vendor` directory to manage external dependencies. If `clientv3` is imported
outside of etcd, simply copy `clientv3` to the `vendor` directory or use tools like godep to
manage your own dependency, as in [vendor directories](https://golang.org/cmd/go/#hdr-Vendor_Directories).
For more detail, please read [Go vendor design](https://golang.org/s/go15vendor).

## Error Handling

etcd client returns 2 types of errors:

1. context error: canceled or deadline exceeded.
2. gRPC error: see [v3rpc/error](https://github.com/coreos/etcd/blob/master/etcdserver/api/v3rpc/error.go).

Here is the example code to handle client errors:

```go
resp, err := kvc.Put(ctx, "", "")
if err != nil {
	if err == context.Canceled {
		// ctx is canceled by another routine
	} else if err == context.DeadlineExceeded {
		// ctx is attached with a deadline and it exceeded
	} else if verr, ok := err.(*v3rpc.ErrEmptyKey); ok {
		// process (verr.Errors)
	} else {
		// bad cluster endpoints, which are not etcd servers
	}
}
```

## Examples

More code examples can be found at [GoDoc](https://godoc.org/github.com/coreos/etcd/clientv3).
