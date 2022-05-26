// +build windows
// +build !appengine

package runewidth

import (
	"syscall"
)

var (
	kernel32               = syscall.NewLazyDLL("kernel32")
	procGetConsoleOutputCP = kernel32.NewProc("GetConsoleOutputCP")
)

// IsEastAsian return true if the current locale is CJK
func IsEastAsian() bool {
	r1, _, _ := procGetConsoleOutputCP.Call()
	if r1 == 0 {
		return false
	}

	switch int(r1) {
	case 932, 51932, 936, 949, 950:
		return true
	}

	return false
}
