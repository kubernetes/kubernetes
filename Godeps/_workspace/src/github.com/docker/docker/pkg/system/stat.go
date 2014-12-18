package system

import (
	"syscall"
)

type Stat struct {
	mode uint32
	uid  uint32
	gid  uint32
	rdev uint64
	size int64
	mtim syscall.Timespec
}

func (s Stat) Mode() uint32 {
	return s.mode
}

func (s Stat) Uid() uint32 {
	return s.uid
}

func (s Stat) Gid() uint32 {
	return s.gid
}

func (s Stat) Rdev() uint64 {
	return s.rdev
}

func (s Stat) Size() int64 {
	return s.size
}

func (s Stat) Mtim() syscall.Timespec {
	return s.mtim
}

func (s Stat) GetLastModification() syscall.Timespec {
	return s.Mtim()
}
