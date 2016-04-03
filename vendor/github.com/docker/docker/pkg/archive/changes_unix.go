// +build !windows

package archive

import (
	"syscall"

	"github.com/docker/docker/pkg/system"
)

func statDifferent(oldStat *system.Stat_t, newStat *system.Stat_t) bool {
	// Don't look at size for dirs, its not a good measure of change
	if oldStat.Mode() != newStat.Mode() ||
		oldStat.Uid() != newStat.Uid() ||
		oldStat.Gid() != newStat.Gid() ||
		oldStat.Rdev() != newStat.Rdev() ||
		// Don't look at size for dirs, its not a good measure of change
		(oldStat.Mode()&syscall.S_IFDIR != syscall.S_IFDIR &&
			(!sameFsTimeSpec(oldStat.Mtim(), newStat.Mtim()) || (oldStat.Size() != newStat.Size()))) {
		return true
	}
	return false
}

func (info *FileInfo) isDir() bool {
	return info.parent == nil || info.stat.Mode()&syscall.S_IFDIR != 0
}
