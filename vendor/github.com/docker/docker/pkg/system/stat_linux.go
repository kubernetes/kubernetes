package system

import (
	"syscall"
)

// fromStatT converts a syscall.Stat_t type to a system.Stat_t type
func fromStatT(s *syscall.Stat_t) (*Stat_t, error) {
	return &Stat_t{size: s.Size,
		mode: s.Mode,
		uid:  s.Uid,
		gid:  s.Gid,
		rdev: s.Rdev,
		mtim: s.Mtim}, nil
}

// FromStatT exists only on linux, and loads a system.Stat_t from a
// syscal.Stat_t.
func FromStatT(s *syscall.Stat_t) (*Stat_t, error) {
	return fromStatT(s)
}

// Stat takes a path to a file and returns
// a system.Stat_t type pertaining to that file.
//
// Throws an error if the file does not exist
func Stat(path string) (*Stat_t, error) {
	s := &syscall.Stat_t{}
	if err := syscall.Stat(path, s); err != nil {
		return nil, err
	}
	return fromStatT(s)
}
