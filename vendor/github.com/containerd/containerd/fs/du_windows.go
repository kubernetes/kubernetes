// +build windows

package fs

import (
	"os"
	"path/filepath"
)

func diskUsage(roots ...string) (Usage, error) {
	var (
		size int64
	)

	// TODO(stevvooe): Support inodes (or equivalent) for windows.

	for _, root := range roots {
		if err := filepath.Walk(root, func(path string, fi os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			size += fi.Size()
			return nil
		}); err != nil {
			return Usage{}, err
		}
	}

	return Usage{
		Size: size,
	}, nil
}
