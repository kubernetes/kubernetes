package system

// This file implements syscalls for Win32 events which are not implemented
// in golang.

import (
	"syscall"
	"unsafe"
)

const (
	EVENT_ALL_ACCESS    = 0x1F0003
	EVENT_MODIFY_STATUS = 0x0002
)

var (
	procCreateEvent = modkernel32.NewProc("CreateEventW")
	procOpenEvent   = modkernel32.NewProc("OpenEventW")
	procSetEvent    = modkernel32.NewProc("SetEvent")
	procResetEvent  = modkernel32.NewProc("ResetEvent")
	procPulseEvent  = modkernel32.NewProc("PulseEvent")
)

func CreateEvent(eventAttributes *syscall.SecurityAttributes, manualReset bool, initialState bool, name string) (handle syscall.Handle, err error) {
	namep, _ := syscall.UTF16PtrFromString(name)
	var _p1 uint32 = 0
	if manualReset {
		_p1 = 1
	}
	var _p2 uint32 = 0
	if initialState {
		_p2 = 1
	}
	r0, _, e1 := procCreateEvent.Call(uintptr(unsafe.Pointer(eventAttributes)), uintptr(_p1), uintptr(_p2), uintptr(unsafe.Pointer(namep)))
	use(unsafe.Pointer(namep))
	handle = syscall.Handle(r0)
	if handle == syscall.InvalidHandle {
		err = e1
	}
	return
}

func OpenEvent(desiredAccess uint32, inheritHandle bool, name string) (handle syscall.Handle, err error) {
	namep, _ := syscall.UTF16PtrFromString(name)
	var _p1 uint32 = 0
	if inheritHandle {
		_p1 = 1
	}
	r0, _, e1 := procOpenEvent.Call(uintptr(desiredAccess), uintptr(_p1), uintptr(unsafe.Pointer(namep)))
	use(unsafe.Pointer(namep))
	handle = syscall.Handle(r0)
	if handle == syscall.InvalidHandle {
		err = e1
	}
	return
}

func SetEvent(handle syscall.Handle) (err error) {
	return setResetPulse(handle, procSetEvent)
}

func ResetEvent(handle syscall.Handle) (err error) {
	return setResetPulse(handle, procResetEvent)
}

func PulseEvent(handle syscall.Handle) (err error) {
	return setResetPulse(handle, procPulseEvent)
}

func setResetPulse(handle syscall.Handle, proc *syscall.LazyProc) (err error) {
	r0, _, _ := proc.Call(uintptr(handle))
	if r0 != 0 {
		err = syscall.Errno(r0)
	}
	return
}

var temp unsafe.Pointer

// use ensures a variable is kept alive without the GC freeing while still needed
func use(p unsafe.Pointer) {
	temp = p
}
