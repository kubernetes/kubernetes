// +build windows

package ole

import (
	"golang.org/x/sys/windows"
)

var (
	modcombase  = windows.NewLazySystemDLL("combase.dll")
	modkernel32 = windows.NewLazySystemDLL("kernel32.dll")
	modole32    = windows.NewLazySystemDLL("ole32.dll")
	modoleaut32 = windows.NewLazySystemDLL("oleaut32.dll")
	moduser32   = windows.NewLazySystemDLL("user32.dll")
)
