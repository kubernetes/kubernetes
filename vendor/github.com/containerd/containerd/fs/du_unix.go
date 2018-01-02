// +build !windows

package fs

import (
	"os"
	"path/filepath"
	"syscall"
)

func diskUsage(roots ...string) (Usage, error) {
	type inode struct {
		// TODO(stevvooe): Can probably reduce memory usage by not tracking
		// device, but we can leave this right for now.
		dev, ino uint64
	}

	var (
		size   int64
		inodes = map[inode]struct{}{} // expensive!
	)

	for _, root := range roots {
		if err := filepath.Walk(root, func(path string, fi os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			stat := fi.Sys().(*syscall.Stat_t)

			inoKey := inode{dev: uint64(stat.Dev), ino: uint64(stat.Ino)}
			if _, ok := inodes[inoKey]; !ok {
				inodes[inoKey] = struct{}{}
				size += fi.Size()
			}

			return nil
		}); err != nil {
			return Usage{}, err
		}
	}

	return Usage{
		Inodes: int64(len(inodes)),
		Size:   size,
	}, nil
}
