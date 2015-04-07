package system

import (
	"fmt"
	"runtime"
	"syscall"
)

// Via http://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/commit/?id=7b21fddd087678a70ad64afc0f632e0f1071b092
//
// We need different setns values for the different platforms and arch
// We are declaring the macro here because the SETNS syscall does not exist in th stdlib
var setNsMap = map[string]uintptr{
	"linux/386":     346,
	"linux/amd64":   308,
	"linux/arm":     374,
	"linux/ppc64":   350,
	"linux/ppc64le": 350,
	"linux/s390x":   339,
}

func Setns(fd uintptr, flags uintptr) error {
	ns, exists := setNsMap[fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH)]
	if !exists {
		return fmt.Errorf("unsupported platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	_, _, err := syscall.RawSyscall(ns, fd, flags, 0)
	if err != 0 {
		return err
	}

	return nil
}
