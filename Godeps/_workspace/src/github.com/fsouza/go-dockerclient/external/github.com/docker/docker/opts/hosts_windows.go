// +build windows

package opts

import "fmt"

var DefaultHost = fmt.Sprintf("tcp://%s:%d", DefaultHTTPHost, DefaultHTTPPort)
