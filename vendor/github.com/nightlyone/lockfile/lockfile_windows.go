package lockfile

import (
	"syscall"
)

//For some reason these consts don't exist in syscall.
const (
	error_invalid_parameter = 87
	code_still_active       = 259
)

func isRunning(pid int) (bool, error) {
	procHnd, err := syscall.OpenProcess(syscall.PROCESS_QUERY_INFORMATION, true, uint32(pid))
	if err != nil {
		if scerr, ok := err.(syscall.Errno); ok {
			if uintptr(scerr) == error_invalid_parameter {
				return false, nil
			}
		}
	}

	var code uint32
	err = syscall.GetExitCodeProcess(procHnd, &code)
	if err != nil {
		return false, err
	}

	return code == code_still_active, nil
}
