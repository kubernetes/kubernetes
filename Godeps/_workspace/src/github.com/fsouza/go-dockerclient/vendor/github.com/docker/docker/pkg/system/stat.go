package system

import (
	"syscall"
)

// Stat_t type contains status of a file. It contains metadata
// like permission, owner, group, size, etc about a file
type Stat_t struct {
	mode uint32
	uid  uint32
	gid  uint32
	rdev uint64
	size int64
	mtim syscall.Timespec
}

func (s Stat_t) Mode() uint32 {
	return s.mode
}

func (s Stat_t) Uid() uint32 {
	return s.uid
}

func (s Stat_t) Gid() uint32 {
	return s.gid
}

func (s Stat_t) Rdev() uint64 {
	return s.rdev
}

func (s Stat_t) Size() int64 {
	return s.size
}

func (s Stat_t) Mtim() syscall.Timespec {
	return s.mtim
}

func (s Stat_t) GetLastModification() syscall.Timespec {
	return s.Mtim()
}
