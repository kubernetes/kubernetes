//go:build js
// +build js

package logrus

func isTerminal(fd int) bool {
	return false
}
