//go:build wasip1
// +build wasip1

package logrus

func isTerminal(fd int) bool {
	return false
}
