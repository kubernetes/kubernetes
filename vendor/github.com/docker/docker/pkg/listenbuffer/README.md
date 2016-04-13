# listenbuffer

listenbuffer uses the kernel's listening backlog functionality to queue
connections, allowing applications to start listening immediately and handle
connections later. This is signaled by closing the activation channel passed to
the constructor.

The maximum amount of queued connections depends on the configuration of your
kernel (typically called SOMAXXCON) and cannot be configured in Go with the
net package. See `src/net/sock_platform.go` in the Go tree or consult your
kernel's manual.

	activator := make(chan struct{})
	buffer, err := NewListenBuffer("tcp", "localhost:4000", activator)
	if err != nil {
		panic(err)
	}

	// will block until activator has been closed or is sent an event
	client, err := buffer.Accept()

Somewhere else in your application once it's been booted:

	close(activator)

`buffer.Accept()` will return the first client in the kernel listening queue, or
continue to block until a client connects or an error occurs.
