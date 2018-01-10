// +build appengine

package logrus

import "io"

// IsTerminal returns true if stderr's file descriptor is a terminal.
func IsTerminal(f io.Writer) bool {
	return true
}
