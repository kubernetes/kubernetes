package daemon

import (
	"fmt"
	"strconv"
	"syscall"
	"unsafe"

	"github.com/go-check/check"
	"golang.org/x/sys/windows"
)

func openEvent(desiredAccess uint32, inheritHandle bool, name string, proc *windows.LazyProc) (handle windows.Handle, err error) {
	namep, _ := windows.UTF16PtrFromString(name)
	var _p2 uint32
	if inheritHandle {
		_p2 = 1
	}
	r0, _, e1 := proc.Call(uintptr(desiredAccess), uintptr(_p2), uintptr(unsafe.Pointer(namep)))
	handle = windows.Handle(r0)
	if handle == windows.InvalidHandle {
		err = e1
	}
	return
}

func pulseEvent(handle windows.Handle, proc *windows.LazyProc) (err error) {
	r0, _, _ := proc.Call(uintptr(handle))
	if r0 != 0 {
		err = syscall.Errno(r0)
	}
	return
}

// SignalDaemonDump sends a signal to the daemon to write a dump file
func SignalDaemonDump(pid int) {
	modkernel32 := windows.NewLazySystemDLL("kernel32.dll")
	procOpenEvent := modkernel32.NewProc("OpenEventW")
	procPulseEvent := modkernel32.NewProc("PulseEvent")

	ev := "Global\\docker-daemon-" + strconv.Itoa(pid)
	h2, _ := openEvent(0x0002, false, ev, procOpenEvent)
	if h2 == 0 {
		return
	}
	pulseEvent(h2, procPulseEvent)
}

func signalDaemonReload(pid int) error {
	return fmt.Errorf("daemon reload not supported")
}

func cleanupExecRoot(c *check.C, execRoot string) {
}
