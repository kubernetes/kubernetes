package systemd

import (
	"errors"
	"net"
	"strconv"

	"github.com/coreos/go-systemd/activation"
)

// ListenFD returns the specified socket activated files as a slice of
// net.Listeners or all of the activated files if "*" is given.
func ListenFD(addr string) ([]net.Listener, error) {
	// socket activation
	listeners, err := activation.Listeners(false)
	if err != nil {
		return nil, err
	}

	if listeners == nil || len(listeners) == 0 {
		return nil, errors.New("No sockets found")
	}

	// default to all fds just like unix:// and tcp://
	if addr == "" {
		addr = "*"
	}

	fdNum, _ := strconv.Atoi(addr)
	fdOffset := fdNum - 3
	if (addr != "*") && (len(listeners) < int(fdOffset)+1) {
		return nil, errors.New("Too few socket activated files passed in")
	}

	if addr == "*" {
		return listeners, nil
	}

	return []net.Listener{listeners[fdOffset]}, nil
}
