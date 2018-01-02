// +build !windows,!solaris

package listeners

import (
	"crypto/tls"
	"fmt"
	"net"
	"os"
	"strconv"

	"github.com/coreos/go-systemd/activation"
	"github.com/docker/go-connections/sockets"
	"github.com/sirupsen/logrus"
)

// Init creates new listeners for the server.
// TODO: Clean up the fact that socketGroup and tlsConfig aren't always used.
func Init(proto, addr, socketGroup string, tlsConfig *tls.Config) ([]net.Listener, error) {
	ls := []net.Listener{}

	switch proto {
	case "fd":
		fds, err := listenFD(addr, tlsConfig)
		if err != nil {
			return nil, err
		}
		ls = append(ls, fds...)
	case "tcp":
		l, err := sockets.NewTCPSocket(addr, tlsConfig)
		if err != nil {
			return nil, err
		}
		ls = append(ls, l)
	case "unix":
		gid, err := lookupGID(socketGroup)
		if err != nil {
			if socketGroup != "" {
				if socketGroup != defaultSocketGroup {
					return nil, err
				}
				logrus.Warnf("could not change group %s to %s: %v", addr, defaultSocketGroup, err)
			}
			gid = os.Getgid()
		}
		l, err := sockets.NewUnixSocket(addr, gid)
		if err != nil {
			return nil, fmt.Errorf("can't create unix socket %s: %v", addr, err)
		}
		ls = append(ls, l)
	default:
		return nil, fmt.Errorf("invalid protocol format: %q", proto)
	}

	return ls, nil
}

// listenFD returns the specified socket activated files as a slice of
// net.Listeners or all of the activated files if "*" is given.
func listenFD(addr string, tlsConfig *tls.Config) ([]net.Listener, error) {
	var (
		err       error
		listeners []net.Listener
	)
	// socket activation
	if tlsConfig != nil {
		listeners, err = activation.TLSListeners(false, tlsConfig)
	} else {
		listeners, err = activation.Listeners(false)
	}
	if err != nil {
		return nil, err
	}

	if len(listeners) == 0 {
		return nil, fmt.Errorf("no sockets found via socket activation: make sure the service was started by systemd")
	}

	// default to all fds just like unix:// and tcp://
	if addr == "" || addr == "*" {
		return listeners, nil
	}

	fdNum, err := strconv.Atoi(addr)
	if err != nil {
		return nil, fmt.Errorf("failed to parse systemd fd address: should be a number: %v", addr)
	}
	fdOffset := fdNum - 3
	if len(listeners) < int(fdOffset)+1 {
		return nil, fmt.Errorf("too few socket activated files passed in by systemd")
	}
	if listeners[fdOffset] == nil {
		return nil, fmt.Errorf("failed to listen on systemd activated file: fd %d", fdOffset+3)
	}
	for i, ls := range listeners {
		if i == fdOffset || ls == nil {
			continue
		}
		if err := ls.Close(); err != nil {
			return nil, fmt.Errorf("failed to close systemd activated file: fd %d: %v", fdOffset+3, err)
		}
	}
	return []net.Listener{listeners[fdOffset]}, nil
}
