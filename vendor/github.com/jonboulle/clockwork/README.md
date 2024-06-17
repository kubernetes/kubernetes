# clockwork

[![Mentioned in Awesome Go](https://awesome.re/mentioned-badge-flat.svg)](https://github.com/avelino/awesome-go#utilities)

[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/jonboulle/clockwork/CI?style=flat-square)](https://github.com/jonboulle/clockwork/actions?query=workflow%3ACI)
[![Go Report Card](https://goreportcard.com/badge/github.com/jonboulle/clockwork?style=flat-square)](https://goreportcard.com/report/github.com/jonboulle/clockwork)
![Go Version](https://img.shields.io/badge/go%20version-%3E=1.11-61CFDD.svg?style=flat-square)
[![go.dev reference](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white&style=flat-square)](https://pkg.go.dev/mod/github.com/jonboulle/clockwork)

**A simple fake clock for Go.**


## Usage

Replace uses of the `time` package with the `clockwork.Clock` interface instead.

For example, instead of using `time.Sleep` directly:

```go
func myFunc() {
	time.Sleep(3 * time.Second)
	doSomething()
}
```

Inject a clock and use its `Sleep` method instead:

```go
func myFunc(clock clockwork.Clock) {
	clock.Sleep(3 * time.Second)
	doSomething()
}
```

Now you can easily test `myFunc` with a `FakeClock`:

```go
func TestMyFunc(t *testing.T) {
	c := clockwork.NewFakeClock()

	// Start our sleepy function
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		myFunc(c)
		wg.Done()
	}()

	// Ensure we wait until myFunc is sleeping
	c.BlockUntil(1)

	assertState()

	// Advance the FakeClock forward in time
	c.Advance(3 * time.Second)

	// Wait until the function completes
	wg.Wait()

	assertState()
}
```

and in production builds, simply inject the real clock instead:

```go
myFunc(clockwork.NewRealClock())
```

See [example_test.go](example_test.go) for a full example.


# Credits

clockwork is inspired by @wickman's [threaded fake clock](https://gist.github.com/wickman/3840816), and the [Golang playground](https://blog.golang.org/playground#TOC_3.1.)


## License

Apache License, Version 2.0. Please see [License File](LICENSE) for more information.
