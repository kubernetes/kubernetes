## go-httpclient

**requires Go 1.1+** as of `v0.4.0` the API has been completely re-written for Go 1.1 (for a Go
1.0.x compatible release see [1adef50](https://github.com/mreiferson/go-httpclient/tree/1adef50))

[![Build
Status](https://secure.travis-ci.org/mreiferson/go-httpclient.png?branch=master)](http://travis-ci.org/mreiferson/go-httpclient)

Provides an HTTP Transport that implements the `RoundTripper` interface and
can be used as a built in replacement for the standard library's, providing:

 * connection timeouts
 * request timeouts

This is a thin wrapper around `http.Transport` that sets dial timeouts and uses
Go's internal timer scheduler to call the Go 1.1+ `CancelRequest()` API.

### Example

```go
transport := &httpclient.Transport{
    ConnectTimeout:        1*time.Second,
    RequestTimeout:        10*time.Second,
    ResponseHeaderTimeout: 5*time.Second,
}
defer transport.Close()

client := &http.Client{Transport: transport}
req, _ := http.NewRequest("GET", "http://127.0.0.1/test", nil)
resp, err := client.Do(req)
if err != nil {
    return err
}
defer resp.Body.Close()
```

*Note:* you will want to re-use a single client object rather than creating one for each request, otherwise you will end up [leaking connections](https://code.google.com/p/go/issues/detail?id=4049#c3).

### Reference Docs

For API docs see [godoc](http://godoc.org/github.com/mreiferson/go-httpclient).
