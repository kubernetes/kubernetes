// +build !windows

package opts

import "fmt"

var DefaultHost = fmt.Sprintf("unix://%s", DefaultUnixSocket)
