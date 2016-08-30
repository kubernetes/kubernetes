// +build windows plan9 netbsd

package gsyslog

import (
	"fmt"
)

// NewLogger is used to construct a new Syslogger
func NewLogger(p Priority, facility, tag string) (Syslogger, error) {
	return nil, fmt.Errorf("Platform does not support syslog")
}
