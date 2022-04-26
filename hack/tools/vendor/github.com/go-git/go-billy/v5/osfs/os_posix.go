// +build !plan9,!windows

package osfs

import (
	"os"

	"golang.org/x/sys/unix"
)

func (f *file) Lock() error {
	f.m.Lock()
	defer f.m.Unlock()

	return unix.Flock(int(f.File.Fd()), unix.LOCK_EX)
}

func (f *file) Unlock() error {
	f.m.Lock()
	defer f.m.Unlock()

	return unix.Flock(int(f.File.Fd()), unix.LOCK_UN)
}

func rename(from, to string) error {
	return os.Rename(from, to)
}
