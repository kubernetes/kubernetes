// +build linux

package network

import (
	"context"
	"net"
	"os"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

func dialerWithDefaultOptions() DialContext {
	nd := &net.Dialer{
		// TCP_USER_TIMEOUT does affect the behaviour of connect() which is controlled by this field so we set it to the same value
		Timeout: 25 * time.Second,
	}
	return wrapDialContext(nd.DialContext)
}

func wrapDialContext(dc DialContext) DialContext {
	return func(ctx context.Context, network, address string) (net.Conn, error) {
		conn, err := dc(ctx, network, address)
		if err != nil {
			return conn, err
		}

		if tcpCon, ok := conn.(*net.TCPConn); ok {
			tcpFD, err := tcpCon.File()
			if err != nil {
				return conn, err
			}
			if err := setDefaultSocketOptions(int(tcpFD.Fd())); err != nil {
				return conn, err
			}
		}
		return conn, nil
	}
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
