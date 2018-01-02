// +build windows

package sys

import (
	"net"

	"github.com/Microsoft/go-winio"
)

// GetLocalListener returns a Listernet out of a named pipe.
// `path` must be of the form of `\\.\pipe\<pipename>`
// (see https://msdn.microsoft.com/en-us/library/windows/desktop/aa365150)
func GetLocalListener(path string, uid, gid int) (net.Listener, error) {
	return winio.ListenPipe(path, nil)
}
