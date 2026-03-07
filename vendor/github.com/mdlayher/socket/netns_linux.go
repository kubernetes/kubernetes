//go:build linux
// +build linux

package socket

import (
	"errors"
	"fmt"
	"os"
	"runtime"

	"golang.org/x/sync/errgroup"
	"golang.org/x/sys/unix"
)

// errNetNSDisabled is returned when network namespaces are unavailable on
// a given system.
var errNetNSDisabled = errors.New("socket: Linux network namespaces are not enabled on this system")

// withNetNS invokes fn within the context of the network namespace specified by
// fd, while also managing the logic required to safely do so by manipulating
// thread-local state.
func withNetNS(fd int, fn func() (*Conn, error)) (*Conn, error) {
	var (
		eg   errgroup.Group
		conn *Conn
	)

	eg.Go(func() error {
		// Retrieve and store the calling OS thread's network namespace so the
		// thread can be reassigned to it after creating a socket in another network
		// namespace.
		runtime.LockOSThread()

		ns, err := threadNetNS()
		if err != nil {
			// No thread-local manipulation, unlock.
			runtime.UnlockOSThread()
			return err
		}
		defer ns.Close()

		// Beyond this point, the thread's network namespace is poisoned. Do not
		// unlock the OS thread until all network namespace manipulation completes
		// to avoid returning to the caller with altered thread-local state.

		// Assign the current OS thread the goroutine is locked to to the given
		// network namespace.
		if err := ns.Set(fd); err != nil {
			return err
		}

		// Attempt Conn creation and unconditionally restore the original namespace.
		c, err := fn()
		if nerr := ns.Restore(); nerr != nil {
			// Failed to restore original namespace. Return an error and allow the
			// runtime to terminate the thread.
			if err == nil {
				_ = c.Close()
			}

			return nerr
		}

		// No more thread-local state manipulation; return the new Conn.
		runtime.UnlockOSThread()
		conn = c
		return nil
	})

	if err := eg.Wait(); err != nil {
		return nil, err
	}

	return conn, nil
}

// A netNS is a handle that can manipulate network namespaces.
//
// Operations performed on a netNS must use runtime.LockOSThread before
// manipulating any network namespaces.
type netNS struct {
	// The handle to a network namespace.
	f *os.File

	// Indicates if network namespaces are disabled on this system, and thus
	// operations should become a no-op or return errors.
	disabled bool
}

// threadNetNS constructs a netNS using the network namespace of the calling
// thread. If the namespace is not the default namespace, runtime.LockOSThread
// should be invoked first.
func threadNetNS() (*netNS, error) {
	return fileNetNS(fmt.Sprintf("/proc/self/task/%d/ns/net", unix.Gettid()))
}

// fileNetNS opens file and creates a netNS. fileNetNS should only be called
// directly in tests.
func fileNetNS(file string) (*netNS, error) {
	f, err := os.Open(file)
	switch {
	case err == nil:
		return &netNS{f: f}, nil
	case os.IsNotExist(err):
		// Network namespaces are not enabled on this system. Use this signal
		// to return errors elsewhere if the caller explicitly asks for a
		// network namespace to be set.
		return &netNS{disabled: true}, nil
	default:
		return nil, err
	}
}

// Close releases the handle to a network namespace.
func (n *netNS) Close() error {
	return n.do(func() error { return n.f.Close() })
}

// FD returns a file descriptor which represents the network namespace.
func (n *netNS) FD() int {
	if n.disabled {
		// No reasonable file descriptor value in this case, so specify a
		// non-existent one.
		return -1
	}

	return int(n.f.Fd())
}

// Restore restores the original network namespace for the calling thread.
func (n *netNS) Restore() error {
	return n.do(func() error { return n.Set(n.FD()) })
}

// Set sets a new network namespace for the current thread using fd.
func (n *netNS) Set(fd int) error {
	return n.do(func() error {
		return os.NewSyscallError("setns", unix.Setns(fd, unix.CLONE_NEWNET))
	})
}

// do runs fn if network namespaces are enabled on this system.
func (n *netNS) do(fn func() error) error {
	if n.disabled {
		return errNetNSDisabled
	}

	return fn()
}
