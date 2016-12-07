// +build !windows,!solaris

package reap

import (
	"os"
	"os/signal"
	"sync"

	"golang.org/x/sys/unix"
)

// IsSupported returns true if child process reaping is supported on this
// platform.
func IsSupported() bool {
	return true
}

// ReapChildren is a long-running routine that blocks waiting for child
// processes to exit and reaps them, reporting reaped process IDs to the
// optional pids channel and any errors to the optional errors channel.
//
// The optional reapLock will be used to prevent reaping during periods
// when you know your application is waiting for subprocesses to return.
// You need to use care in order to prevent the reaper from stealing your
// return values from uses of packages like Go's exec. We use an RWMutex
// so that we don't serialize all of the application's execution of sub
// processes with each other, but we do serialize them with reaping. The
// application should get a read lock when it wants to do a wait.
func ReapChildren(pids PidCh, errors ErrorCh, done chan struct{}, reapLock *sync.RWMutex) {
	c := make(chan os.Signal, 1)
	signal.Notify(c, unix.SIGCHLD)

	for {
		// Block for an incoming signal that a child has exited.
		select {
		case <-c:
			// Got a child signal, drop out and reap.
		case <-done:
			return
		}

		// Attempt to reap all abandoned child processes after getting
		// the reap lock, which makes sure the application isn't doing
		// any waiting of its own. Note that we do the full write lock
		// here.
		func() {
			if reapLock != nil {
				reapLock.Lock()
				defer reapLock.Unlock()
			}

		POLL:
			// Try to reap children until there aren't any more. We
			// never block in here so that we are always responsive
			// to signals, at the expense of possibly leaving a
			// child behind if we get here too quickly. Any
			// stragglers should get reaped the next time we see a
			// signal, so we won't leak in the long run.
			var status unix.WaitStatus
			pid, err := unix.Wait4(-1, &status, unix.WNOHANG, nil)
			switch err {
			case nil:
				// Got a child, clean this up and poll again.
				if pid > 0 {
					if pids != nil {
						pids <- pid
					}
					goto POLL
				}
				return

			case unix.ECHILD:
				// No more children, we are done.
				return

			case unix.EINTR:
				// We got interrupted, try again. This likely
				// can't happen since we are calling Wait4 in a
				// non-blocking fashion, but it's good to be
				// complete and handle this case rather than
				// fail.
				goto POLL

			default:
				// We got some other error we didn't expect.
				// Wait for another SIGCHLD so we don't
				// potentially spam in here and chew up CPU.
				if errors != nil {
					errors <- err
				}
				return
			}
		}()
	}
}
