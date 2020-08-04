// +build windows

package ole

import (
	"syscall"
)

var (
	modcombase     = syscall.NewLazyDLL("combase.dll")
	modkernel32, _ = syscall.LoadDLL("kernel32.dll")
	modole32, _    = syscall.LoadDLL("ole32.dll")
	modoleaut32, _ = syscall.LoadDLL("oleaut32.dll")
	modmsvcrt, _   = syscall.LoadDLL("msvcrt.dll")
	moduser32, _   = syscall.LoadDLL("user32.dll")
)
