// +build windows

package speakeasy

import (
	"syscall"
)

// SetConsoleMode function can be used to change value of ENABLE_ECHO_INPUT:
// http://msdn.microsoft.com/en-us/library/windows/desktop/ms686033(v=vs.85).aspx
const ENABLE_ECHO_INPUT = 0x0004

func getPassword() (password string, err error) {
	var oldMode uint32

	err = syscall.GetConsoleMode(syscall.Stdin, &oldMode)
	if err != nil {
		return
	}

	var newMode uint32 = (oldMode &^ ENABLE_ECHO_INPUT)

	err = setConsoleMode(syscall.Stdin, newMode)
	defer setConsoleMode(syscall.Stdin, oldMode)
	if err != nil {
		return
	}

	return readline()
}

func setConsoleMode(console syscall.Handle, mode uint32) (err error) {
	dll := syscall.MustLoadDLL("kernel32")
	proc := dll.MustFindProc("SetConsoleMode")
	r, _, err := proc.Call(uintptr(console), uintptr(mode))

	if r == 0 {
		return err
	}
	return nil
}
