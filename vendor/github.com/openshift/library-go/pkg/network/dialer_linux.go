//go:build linux
// +build linux

package network

import (
	"net"
	"os"
	"syscall"
	"time"

	"golang.org/x/sys/unix"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

func dialerWithDefaultOptions() DialContext {
	nd := &net.Dialer{
		// TCP_USER_TIMEOUT does affect the behaviour of connect() which is controlled by this field so we set it to the same value
		Timeout: 25 * time.Second,
		// KeepAlive must to be set to a negative value to stop std library from applying the default values
		// by doing so we ensure that the options we are interested in won't be overwritten
		KeepAlive: time.Duration(-1),
		Control: func(network, address string, con syscall.RawConn) error {
			var errs []error
			err := con.Control(func(fd uintptr) {
				optionsErr := setDefaultSocketOptions(int(fd))
				if optionsErr != nil {
					errs = append(errs, optionsErr)
				}
			})
			if err != nil {
				errs = append(errs, err)
			}
			return utilerrors.NewAggregate(errs)
		},
	}
	return nd.DialContext
}

// setDefaultSocketOptions sets custom socket options so that we can detect connections to an unhealthy (dead) peer quickly.
// In particular we set TCP_USER_TIMEOUT that specifies the maximum amount of time that transmitted data may remain
// unacknowledged before TCP will forcibly close the connection.
//
// Note
// TCP_USER_TIMEOUT can't be too low because a single dropped packet might drop the entire connection.
// Ideally it should be set to: TCP_KEEPIDLE + TCP_KEEPINTVL * TCP_KEEPCNT
func setDefaultSocketOptions(fd int) error {
	// specifies the maximum amount of time in milliseconds that transmitted data may remain
	// unacknowledged before TCP will forcibly close the corresponding connection and return ETIMEDOUT to the application
	tcpUserTimeoutInMilliSeconds := int(25 * time.Second / time.Millisecond)

	// specifies the interval at which probes are sent in seconds
	tcpKeepIntvl := int(roundDuration(5*time.Second, time.Second))

	// specifies the threshold for sending the first KEEP ALIVE probe in seconds
	tcpKeepIdle := int(roundDuration(2*time.Second, time.Second))

	// enable keep-alive probes
	if err := syscall.SetsockoptInt(int(fd), syscall.SOL_SOCKET, syscall.SO_KEEPALIVE, 1); err != nil {
		return wrapSyscallError("setsockopt", err)
	}

	if err := syscall.SetsockoptInt(int(fd), syscall.IPPROTO_TCP, unix.TCP_USER_TIMEOUT, tcpUserTimeoutInMilliSeconds); err != nil {
		return wrapSyscallError("setsockopt", err)
	}

	if err := syscall.SetsockoptInt(int(fd), syscall.IPPROTO_TCP, syscall.TCP_KEEPINTVL, tcpKeepIntvl); err != nil {
		return wrapSyscallError("setsockopt", err)
	}

	if err := syscall.SetsockoptInt(int(fd), syscall.IPPROTO_TCP, syscall.TCP_KEEPIDLE, tcpKeepIdle); err != nil {
		return wrapSyscallError("setsockopt", err)
	}
	return nil
}

// roundDurationUp rounds d to the next multiple of to.
//
// note that it was copied from the std library
func roundDuration(d time.Duration, to time.Duration) time.Duration {
	return (d + to - 1) / to
}

// wrapSyscallError takes an error and a syscall name. If the error is
// a syscall.Errno, it wraps it in a os.SyscallError using the syscall name.
//
// note that it was copied from the std library
func wrapSyscallError(name string, err error) error {
	if _, ok := err.(syscall.Errno); ok {
		err = os.NewSyscallError(name, err)
	}
	return err
}
