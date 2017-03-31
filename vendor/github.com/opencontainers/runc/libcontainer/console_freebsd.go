// +build freebsd

package libcontainer

import (
	"errors"
)

// newConsole returns an initialized console that can be used within a container by copying bytes
// from the master side to the slave that is attached as the tty for the container's init process.
func newConsole() (Console, error) {
	return nil, errors.New("libcontainer console is not supported on FreeBSD")
}
