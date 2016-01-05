package bolt

import (
	"fmt"
	"os"
	"syscall"
	"time"
	"unsafe"
)

var odirect int

// fdatasync flushes written data to a file descriptor.
func fdatasync(db *DB) error {
	return db.file.Sync()
}

// flock acquires an advisory lock on a file descriptor.
func flock(f *os.File, _ bool, _ time.Duration) error {
	return nil
}

// funlock releases an advisory lock on a file descriptor.
func funlock(f *os.File) error {
	return nil
}

// mmap memory maps a DB's data file.
// Based on: https://github.com/edsrzf/mmap-go
func mmap(db *DB, sz int) error {
	if !db.readOnly {
		// Truncate the database to the size of the mmap.
		if err := db.file.Truncate(int64(sz)); err != nil {
			return fmt.Errorf("truncate: %s", err)
		}
	}

	// Open a file mapping handle.
	sizelo := uint32(sz >> 32)
	sizehi := uint32(sz) & 0xffffffff
	h, errno := syscall.CreateFileMapping(syscall.Handle(db.file.Fd()), nil, syscall.PAGE_READONLY, sizelo, sizehi, nil)
	if h == 0 {
		return os.NewSyscallError("CreateFileMapping", errno)
	}

	// Create the memory map.
	addr, errno := syscall.MapViewOfFile(h, syscall.FILE_MAP_READ, 0, 0, uintptr(sz))
	if addr == 0 {
		return os.NewSyscallError("MapViewOfFile", errno)
	}

	// Close mapping handle.
	if err := syscall.CloseHandle(syscall.Handle(h)); err != nil {
		return os.NewSyscallError("CloseHandle", err)
	}

	// Convert to a byte array.
	db.data = ((*[maxMapSize]byte)(unsafe.Pointer(addr)))
	db.datasz = sz

	return nil
}

// munmap unmaps a pointer from a file.
// Based on: https://github.com/edsrzf/mmap-go
func munmap(db *DB) error {
	if db.data == nil {
		return nil
	}

	addr := (uintptr)(unsafe.Pointer(&db.data[0]))
	if err := syscall.UnmapViewOfFile(addr); err != nil {
		return os.NewSyscallError("UnmapViewOfFile", err)
	}
	return nil
}
