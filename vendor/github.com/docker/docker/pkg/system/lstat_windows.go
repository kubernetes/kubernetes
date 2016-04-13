// +build windows

package system

import (
	"os"
)

// Some explanation for my own sanity, and hopefully maintainers in the
// future.
//
// Lstat calls os.Lstat to get a fileinfo interface back.
// This is then copied into our own locally defined structure.
// Note the Linux version uses fromStatT to do the copy back,
// but that not strictly necessary when already in an OS specific module.

func Lstat(path string) (*Stat_t, error) {
	fi, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}

	return &Stat_t{
		name:    fi.Name(),
		size:    fi.Size(),
		mode:    fi.Mode(),
		modTime: fi.ModTime(),
		isDir:   fi.IsDir()}, nil
}
