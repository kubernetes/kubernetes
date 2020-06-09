package fileutils

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"syscall"
)

// CopyFile copies the file at source to dest
func CopyFile(source string, dest string) error {
	si, err := os.Lstat(source)
	if err != nil {
		return err
	}

	st, ok := si.Sys().(*syscall.Stat_t)
	if !ok {
		return fmt.Errorf("could not convert to syscall.Stat_t")
	}

	uid := int(st.Uid)
	gid := int(st.Gid)

	// Handle symlinks
	if si.Mode()&os.ModeSymlink != 0 {
		target, err := os.Readlink(source)
		if err != nil {
			return err
		}
		if err := os.Symlink(target, dest); err != nil {
			return err
		}
	}

	// Handle device files
	if st.Mode&syscall.S_IFMT == syscall.S_IFBLK || st.Mode&syscall.S_IFMT == syscall.S_IFCHR {
		devMajor := int64(major(uint64(st.Rdev)))
		devMinor := int64(minor(uint64(st.Rdev)))
		mode := uint32(si.Mode() & 07777)
		if st.Mode&syscall.S_IFMT == syscall.S_IFBLK {
			mode |= syscall.S_IFBLK
		}
		if st.Mode&syscall.S_IFMT == syscall.S_IFCHR {
			mode |= syscall.S_IFCHR
		}
		if err := syscall.Mknod(dest, mode, int(mkdev(devMajor, devMinor))); err != nil {
			return err
		}
	}

	// Handle regular files
	if si.Mode().IsRegular() {
		sf, err := os.Open(source)
		if err != nil {
			return err
		}
		defer sf.Close()

		df, err := os.Create(dest)
		if err != nil {
			return err
		}
		defer df.Close()

		_, err = io.Copy(df, sf)
		if err != nil {
			return err
		}
	}

	// Chown the file
	if err := os.Lchown(dest, uid, gid); err != nil {
		return err
	}

	// Chmod the file
	if !(si.Mode()&os.ModeSymlink == os.ModeSymlink) {
		if err := os.Chmod(dest, si.Mode()); err != nil {
			return err
		}
	}

	return nil
}

// CopyDirectory copies the files under the source directory
// to dest directory. The dest directory is created if it
// does not exist.
func CopyDirectory(source string, dest string) error {
	fi, err := os.Stat(source)
	if err != nil {
		return err
	}

	// Get owner.
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return fmt.Errorf("could not convert to syscall.Stat_t")
	}

	// We have to pick an owner here anyway.
	if err := MkdirAllNewAs(dest, fi.Mode(), int(st.Uid), int(st.Gid)); err != nil {
		return err
	}

	return filepath.Walk(source, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Get the relative path
		relPath, err := filepath.Rel(source, path)
		if err != nil {
			return nil
		}

		if info.IsDir() {
			// Skip the source directory.
			if path != source {
				// Get the owner.
				st, ok := info.Sys().(*syscall.Stat_t)
				if !ok {
					return fmt.Errorf("could not convert to syscall.Stat_t")
				}

				uid := int(st.Uid)
				gid := int(st.Gid)

				if err := os.Mkdir(filepath.Join(dest, relPath), info.Mode()); err != nil {
					return err
				}

				if err := os.Lchown(filepath.Join(dest, relPath), uid, gid); err != nil {
					return err
				}
			}
			return nil
		}

		return CopyFile(path, filepath.Join(dest, relPath))
	})
}

// Gives a number indicating the device driver to be used to access the passed device
func major(device uint64) uint64 {
	return (device >> 8) & 0xfff
}

// Gives a number that serves as a flag to the device driver for the passed device
func minor(device uint64) uint64 {
	return (device & 0xff) | ((device >> 12) & 0xfff00)
}

func mkdev(major int64, minor int64) uint32 {
	return uint32(((minor & 0xfff00) << 12) | ((major & 0xfff) << 8) | (minor & 0xff))
}
