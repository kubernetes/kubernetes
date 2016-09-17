# go-reap

Provides a super simple set of functions for reaping child processes. This is
useful for running applications as PID 1 in a Docker container.

Note that a mutex is supplied to allow your application to prevent reaping of
child processes during certain periods. You need to use care in order to
prevent the reaper from stealing your return values from uses of packages like
Go's exec. We use an `RWMutex` so that we don't serialize all of your
application's execution of sub processes with each other, but we do serialize
them with reaping. Your application should get a read lock when it wants to do
a wait and be safe from the reaper.

This should be supported on most UNIX flavors, but is not supported on Windows
or Solaris. Unsupported platforms have a stub implementation that's safe to call,
as well as an API to check if reaping is supported so that you can produce an
error in your application code.

Documentation
=============

The full documentation is available on [Godoc](http://godoc.org/github.com/hashicorp/go-reap).

Example
=======

Below is a simple example of usage

```go
// Reap children with no control or feedback.
go ReapChildren(nil, nil, nil)

// Get feedback on reaped children and errors.
if reap.IsSupported() {
	pids := make(reap.PidCh, 1)
	errors := make(reap.ErrorCh, 1)
	done := make(chan struct{})
	var reapLock sync.RWMutex
	go ReapChildren(pids, errors, done, &reapLock)
	// ...
	close(done)
} else {
	fmt.Println("Sorry, go-reap isn't supported on your platform.")
}
```

