package system

// This file implements syscalls for Win32 events which are not implemented
// in golang.

import (
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	procCreateEvent = modkernel32.NewProc("CreateEventW")
	procOpenEvent   = modkernel32.NewProc("OpenEventW")
	procSetEvent    = modkernel32.NewProc("SetEvent")
	procResetEvent  = modkernel32.NewProc("ResetEvent")
	procPulseEvent  = modkernel32.NewProc("PulseEvent")
)

// CreateEvent implements win32 CreateEventW func in golang. It will create an event object.
func CreateEvent(eventAttributes *windows.SecurityAttributes, manualReset bool, initialState bool, name string) (handle windows.Handle, err error) {
	namep, _ := windows.UTF16PtrFromString(name)
	var _p1 uint32
	if manualReset {
		_p1 = 1
	}
	var _p2 uint32
	if initialState {
		_p2 = 1
	}
	r0, _, e1 := procCreateEvent.Call(uintptr(unsafe.Pointer(eventAttributes)), uintptr(_p1), uintptr(_p2), uintptr(unsafe.Pointer(namep)))
	use(unsafe.Pointer(namep))
	handle = windows.Handle(r0)
	if handle == windows.InvalidHandle {
		err = e1
	}
	return
}

// OpenEvent implements win32 OpenEventW func in golang. It opens an event object.
func OpenEvent(desiredAccess uint32, inheritHandle bool, name string) (handle windows.Handle, err error) {
	namep, _ := windows.UTF16PtrFromString(name)
	var _p1 uint32
	if inheritHandle {
		_p1 = 1
	}
	r0, _, e1 := procOpenEvent.Call(uintptr(desiredAccess), uintptr(_p1), uintptr(unsafe.Pointer(namep)))
	use(unsafe.Pointer(namep))
	handle = windows.Handle(r0)
	if handle == windows.InvalidHandle {
		err = e1
	}
	return
}

// SetEvent implements win32 SetEvent func in golang.
func SetEvent(handle windows.Handle) (err error) {
	return setResetPulse(handle, procSetEvent)
}

// ResetEvent implements win32 ResetEvent func in golang.
func ResetEvent(handle windows.Handle) (err error) {
	return setResetPulse(handle, procResetEvent)
}

// PulseEvent implements win32 PulseEvent func in golang.
func PulseEvent(handle windows.Handle) (err error) {
	return setResetPulse(handle, procPulseEvent)
}

func setResetPulse(handle windows.Handle, proc *windows.LazyProc) (err error) {
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
