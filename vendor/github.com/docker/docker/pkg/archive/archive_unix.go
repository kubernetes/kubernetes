// +build !windows

package archive

import (
	"archive/tar"
	"errors"
	"os"
	"syscall"

	"github.com/docker/docker/pkg/system"
)

// CanonicalTarNameForPath returns platform-specific filepath
// to canonical posix-style path for tar archival. p is relative
// path.
func CanonicalTarNameForPath(p string) (string, error) {
	return p, nil // already unix-style
}

// chmodTarEntry is used to adjust the file permissions used in tar header based
// on the platform the archival is done.

func chmodTarEntry(perm os.FileMode) os.FileMode {
	return perm // noop for unix as golang APIs provide perm bits correctly
}

func setHeaderForSpecialDevice(hdr *tar.Header, ta *tarAppender, name string, stat interface{}) (nlink uint32, inode uint64, err error) {
	s, ok := stat.(*syscall.Stat_t)

	if !ok {
		err = errors.New("cannot convert stat value to syscall.Stat_t")
		return
	}

	nlink = uint32(s.Nlink)
	inode = uint64(s.Ino)

	// Currently go does not fil in the major/minors
	if s.Mode&syscall.S_IFBLK != 0 ||
		s.Mode&syscall.S_IFCHR != 0 {
		hdr.Devmajor = int64(major(uint64(s.Rdev)))
		hdr.Devminor = int64(minor(uint64(s.Rdev)))
	}

	return
}

func major(device uint64) uint64 {
	return (device >> 8) & 0xfff
}

func minor(device uint64) uint64 {
	return (device & 0xff) | ((device >> 12) & 0xfff00)
}

// handleTarTypeBlockCharFifo is an OS-specific helper function used by
// createTarFile to handle the following types of header: Block; Char; Fifo
func handleTarTypeBlockCharFifo(hdr *tar.Header, path string) error {
	mode := uint32(hdr.Mode & 07777)
	switch hdr.Typeflag {
	case tar.TypeBlock:
		mode |= syscall.S_IFBLK
	case tar.TypeChar:
		mode |= syscall.S_IFCHR
	case tar.TypeFifo:
		mode |= syscall.S_IFIFO
	}

	if err := system.Mknod(path, mode, int(system.Mkdev(hdr.Devmajor, hdr.Devminor))); err != nil {
		return err
	}
	return nil
}

func handleLChmod(hdr *tar.Header, path string, hdrInfo os.FileInfo) error {
	if hdr.Typeflag == tar.TypeLink {
		if fi, err := os.Lstat(hdr.Linkname); err == nil && (fi.Mode()&os.ModeSymlink == 0) {
			if err := os.Chmod(path, hdrInfo.Mode()); err != nil {
				return err
			}
		}
	} else if hdr.Typeflag != tar.TypeSymlink {
		if err := os.Chmod(path, hdrInfo.Mode()); err != nil {
			return err
		}
	}
	return nil
}
