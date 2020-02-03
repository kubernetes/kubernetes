// +build linux darwin dragonfly freebsd netbsd openbsd
// +build !appengine

package msgp

import (
	"os"
	"syscall"
)

// ReadFile reads a file into 'dst' using
// a read-only memory mapping. Consequently,
// the file must be mmap-able, and the
// Unmarshaler should never write to
// the source memory. (Methods generated
// by the msgp tool obey that constraint, but
// user-defined implementations may not.)
//
// Reading and writing through file mappings
// is only efficient for large files; small
// files are best read and written using
// the ordinary streaming interfaces.
//
func ReadFile(dst Unmarshaler, file *os.File) error {
	stat, err := file.Stat()
	if err != nil {
		return err
	}
	data, err := syscall.Mmap(int(file.Fd()), 0, int(stat.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return err
	}
	adviseRead(data)
	_, err = dst.UnmarshalMsg(data)
	uerr := syscall.Munmap(data)
	if err == nil {
		err = uerr
	}
	return err
}

// MarshalSizer is the combination
// of the Marshaler and Sizer
// interfaces.
type MarshalSizer interface {
	Marshaler
	Sizer
}

// WriteFile writes a file from 'src' using
// memory mapping. It overwrites the entire
// contents of the previous file.
// The mapping size is calculated
// using the `Msgsize()` method
// of 'src', so it must produce a result
// equal to or greater than the actual encoded
// size of the object. Otherwise,
// a fault (SIGBUS) will occur.
//
// Reading and writing through file mappings
// is only efficient for large files; small
// files are best read and written using
// the ordinary streaming interfaces.
//
// NOTE: The performance of this call
// is highly OS- and filesystem-dependent.
// Users should take care to test that this
// performs as expected in a production environment.
// (Linux users should run a kernel and filesystem
// that support fallocate(2) for the best results.)
func WriteFile(src MarshalSizer, file *os.File) error {
	sz := src.Msgsize()
	err := fallocate(file, int64(sz))
	if err != nil {
		return err
	}
	data, err := syscall.Mmap(int(file.Fd()), 0, sz, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return err
	}
	adviseWrite(data)
	chunk := data[:0]
	chunk, err = src.MarshalMsg(chunk)
	if err != nil {
		return err
	}
	uerr := syscall.Munmap(data)
	if uerr != nil {
		return uerr
	}
	return file.Truncate(int64(len(chunk)))
}
