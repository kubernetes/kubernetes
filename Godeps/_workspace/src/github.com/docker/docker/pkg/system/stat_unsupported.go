// +build !linux,!windows

package system

import (
	"syscall"
)

func fromStatT(s *syscall.Stat_t) (*Stat, error) {
	return &Stat{size: s.Size,
		mode: uint32(s.Mode),
		uid:  s.Uid,
		gid:  s.Gid,
		rdev: uint64(s.Rdev),
		mtim: s.Mtimespec}, nil
}
