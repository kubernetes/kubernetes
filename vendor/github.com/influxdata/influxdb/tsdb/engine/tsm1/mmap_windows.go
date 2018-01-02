package tsm1

import (
	"errors"
	"os"
	"reflect"
	"sync"
	"syscall"
	"unsafe"
)

// mmap implementation for Windows
// Based on: https://github.com/edsrzf/mmap-go
// Based on: https://github.com/boltdb/bolt/bolt_windows.go
// Ref: https://groups.google.com/forum/#!topic/golang-nuts/g0nLwQI9www

// We keep this map so that we can get back the original handle from the memory address.
var handleLock sync.Mutex
var handleMap = map[uintptr]syscall.Handle{}
var fileMap = map[uintptr]*os.File{}

func openSharedFile(f *os.File) (file *os.File, err error) {

	var access, createmode, sharemode uint32
	var sa *syscall.SecurityAttributes

	access = syscall.GENERIC_READ
	sharemode = uint32(syscall.FILE_SHARE_READ | syscall.FILE_SHARE_WRITE | syscall.FILE_SHARE_DELETE)
	createmode = syscall.OPEN_EXISTING
	fileName := f.Name()

	pathp, err := syscall.UTF16PtrFromString(fileName)
	if err != nil {
		return nil, err
	}

	h, e := syscall.CreateFile(pathp, access, sharemode, sa, createmode, syscall.FILE_ATTRIBUTE_NORMAL, 0)

	if e != nil {
		return nil, e
	}
	//NewFile does not add finalizer, need to close this manually
	return os.NewFile(uintptr(h), fileName), nil
}

func mmap(f *os.File, offset int64, length int) (out []byte, err error) {
	// Open a file mapping handle.
	sizelo := uint32(length >> 32)
	sizehi := uint32(length) & 0xffffffff

	sharedHandle, errno := openSharedFile(f)
	if errno != nil {
		return nil, os.NewSyscallError("CreateFile", errno)
	}

	h, errno := syscall.CreateFileMapping(syscall.Handle(sharedHandle.Fd()), nil, syscall.PAGE_READONLY, sizelo, sizehi, nil)
	if h == 0 {
		return nil, os.NewSyscallError("CreateFileMapping", errno)
	}

	// Create the memory map.
	addr, errno := syscall.MapViewOfFile(h, syscall.FILE_MAP_READ, 0, 0, uintptr(length))
	if addr == 0 {
		return nil, os.NewSyscallError("MapViewOfFile", errno)
	}

	handleLock.Lock()
	handleMap[addr] = h
	fileMap[addr] = sharedHandle
	handleLock.Unlock()

	// Convert to a byte array.
	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&out))
	hdr.Data = uintptr(unsafe.Pointer(addr))
	hdr.Len = length
	hdr.Cap = length

	return
}

// munmap Windows implementation
// Based on: https://github.com/edsrzf/mmap-go
// Based on: https://github.com/boltdb/bolt/bolt_windows.go
func munmap(b []byte) (err error) {
	handleLock.Lock()
	defer handleLock.Unlock()

	addr := (uintptr)(unsafe.Pointer(&b[0]))
	if err := syscall.UnmapViewOfFile(addr); err != nil {
		return os.NewSyscallError("UnmapViewOfFile", err)
	}

	handle, ok := handleMap[addr]
	if !ok {
		// should be impossible; we would've seen the error above
		return errors.New("unknown base address")
	}
	delete(handleMap, addr)

	e := syscall.CloseHandle(syscall.Handle(handle))
	if e != nil {
		return os.NewSyscallError("CloseHandle", e)
	}

	file, ok := fileMap[addr]
	if !ok {
		// should be impossible; we would've seen the error above
		return errors.New("unknown base address")
	}
	delete(fileMap, addr)

	e = file.Close()
	if e != nil {
		return errors.New("close file" + e.Error())
	}
	return nil
}
