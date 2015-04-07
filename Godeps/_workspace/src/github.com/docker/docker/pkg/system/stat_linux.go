package system

import (
	"syscall"
)

func fromStatT(s *syscall.Stat_t) (*Stat_t, error) {
	return &Stat_t{size: s.Size,
		mode: s.Mode,
		uid:  s.Uid,
		gid:  s.Gid,
		rdev: s.Rdev,
		mtim: s.Mtim}, nil
}

func Stat(path string) (*Stat_t, error) {
	s := &syscall.Stat_t{}
	err := syscall.Stat(path, s)
	if err != nil {
		return nil, err
	}
	return fromStatT(s)
}
