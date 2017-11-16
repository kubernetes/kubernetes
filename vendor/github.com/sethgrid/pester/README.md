# pester

`pester` wraps Go's standard lib http client to provide several options to increase resiliency in your request. If you experience poor network conditions or requests could experience varied delays, you can now pester the endpoint for data.
- Send out multiple requests and get the first back (only used for GET calls)
- Retry on errors
- Backoff

### Simple Example
Use `pester` where you would use the http client calls. By default, pester will use a concurrency of 1, and retry the endpoint 3 times with the `DefaultBackoff` strategy of waiting 1 second between retries.
```go
/* swap in replacement, just switch
   http.{Get|Post|PostForm|Head|Do} to
   pester.{Get|Post|PostForm|Head|Do}
*/
resp, err := pester.Get("http://sethammons.com")
```

### Backoff Strategy
Provide your own backoff strategy, or use one of the provided built in strategies:
- `DefaultBackoff`: 1 second
- `LinearBackoff`: n seconds where n is the retry number
- `LinearJitterBackoff`: n seconds where n is the retry number, +/- 0-33%
- `ExponentialBackoff`: n seconds where n is 2^(retry number)
- `ExponentialJitterBackoff`: n seconds where n is 2^(retry number), +/- 0-33%

```go
client := pester.New()
client.Backoff = func(retry int) time.Duration {
    // set up something dynamic or use a look up table
    return time.Duration(retry) * time.Minute
}
```

### Complete example
For a complete and working example, see the sample directory.
`pester` allows you to use a constructor to control:
- backoff strategy
- retries
- concurrency
- keeping a log for debugging
```go
package main

import (
    "log"
    "net/http"
    "strings"

    "github.com/sethgrid/pester"
)

func main() {
    log.Println("Starting...")

    { // drop in replacement for http.Get and other client methods
        resp, err := pester.Get("http://example.com")
        if err != nil {
            log.Println("error GETing example.com", err)
        }
        defer resp.Body.Close()
        log.Printf("example.com %s", resp.Status)
    }

    { // control the resiliency
        client := pester.New()
        client.Concurrency = 3
        client.MaxRetries = 5
        client.Backoff = pester.ExponentialBackoff
        client.KeepLog = true

        resp, err := client.Get("http://example.com")
        if err != nil {
            log.Println("error GETing example.com", client.LogString())
        }
        defer resp.Body.Close()
        log.Printf("example.com %s", resp.Status)
    }

    { // use the pester version of http.Client.Do
        req, err := http.NewRequest("POST", "http://example.com", strings.NewReader("data"))
        if err != nil {
            log.Fatal("Unable to create a new http request", err)
        }
        resp, err := pester.Do(req)
        if err != nil {
            log.Println("error POSTing example.com", err)
        }
        defer resp.Body.Close()
        log.Printf("example.com %s", resp.Status)
    }
}

```

### Example Log
`pester` also allows you to control the resiliency and can optionally log the errors.
```go
c := pester.New()
c.KeepLog = true

nonExistantURL := "http://localhost:9000/foo"
_, _ = c.Get(nonExistantURL)

fmt.Println(c.LogString())
/*
Output:

1432402837 Get [GET] http://localhost:9000/foo request-0 retry-0 error: Get http://localhost:9000/foo: dial tcp 127.0.0.1:9000: connection refused
1432402838 Get [GET] http://localhost:9000/foo request-0 retry-1 error: Get http://localhost:9000/foo: dial tcp 127.0.0.1:9000: connection refused
1432402839 Get [GET] http://localhost:9000/foo request-0 retry-2 error: Get http://localhost:9000/foo: dial tcp 127.0.0.1:9000: connection refused
*/
```

### Tests

You can run tests in the root directory with `$ go test`. There is a benchmark-like test available with `$ cd benchmarks; go test`.
You can see `pester` in action with `$ cd sample; go run main.go`.

For watching open file descriptors, you can run `watch "lsof -i -P | grep main"` if you started the app with `go run main.go`.
I did this for watching for FD leaks. My method was to alter `sample/main.go` to only run one case (`pester.Get with set backoff stategy, concurrency and retries increased`)
and adding a sleep after the result came back. This let me verify if FDs were getting left open when they should have closed. If you know a better way, let me know!
I was able to see that FDs are now closing when they should :)

![Are we there yet?](http://butchbellah.com/wp-content/uploads/2012/06/Are-We-There-Yet.jpg)

Are we there yet? Are we there yet? Are we there yet? Are we there yet? ...
