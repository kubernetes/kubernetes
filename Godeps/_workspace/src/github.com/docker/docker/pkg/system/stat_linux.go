package system

import (
	"syscall"
)

func fromStatT(s *syscall.Stat_t) (*Stat, error) {
	return &Stat{size: s.Size,
		mode: s.Mode,
		uid:  s.Uid,
		gid:  s.Gid,
		rdev: s.Rdev,
		mtim: s.Mtim}, nil
}
