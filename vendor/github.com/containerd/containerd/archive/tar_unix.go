// +build !windows

package archive

import (
	"os"
	"sync"
	"syscall"

	"github.com/containerd/continuity/sysx"
	"github.com/dmcgowan/go-tar"
	"github.com/opencontainers/runc/libcontainer/system"
	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

func tarName(p string) (string, error) {
	return p, nil
}

func chmodTarEntry(perm os.FileMode) os.FileMode {
	return perm
}

func setHeaderForSpecialDevice(hdr *tar.Header, name string, fi os.FileInfo) error {
	s, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return errors.New("unsupported stat type")
	}

	// Currently go does not fill in the major/minors
	if s.Mode&syscall.S_IFBLK != 0 ||
		s.Mode&syscall.S_IFCHR != 0 {
		hdr.Devmajor = int64(unix.Major(uint64(s.Rdev)))
		hdr.Devminor = int64(unix.Minor(uint64(s.Rdev)))
	}

	return nil
}

func open(p string) (*os.File, error) {
	return os.Open(p)
}

func openFile(name string, flag int, perm os.FileMode) (*os.File, error) {
	f, err := os.OpenFile(name, flag, perm)
	if err != nil {
		return nil, err
	}
	// Call chmod to avoid permission mask
	if err := os.Chmod(name, perm); err != nil {
		return nil, err
	}
	return f, err
}

func mkdirAll(path string, perm os.FileMode) error {
	return os.MkdirAll(path, perm)
}

func mkdir(path string, perm os.FileMode) error {
	if err := os.Mkdir(path, perm); err != nil {
		return err
	}
	// Only final created directory gets explicit permission
	// call to avoid permission mask
	return os.Chmod(path, perm)
}

func skipFile(*tar.Header) bool {
	return false
}

var (
	inUserNS bool
	nsOnce   sync.Once
)

func setInUserNS() {
	inUserNS = system.RunningInUserNS()
}

// handleTarTypeBlockCharFifo is an OS-specific helper function used by
// createTarFile to handle the following types of header: Block; Char; Fifo
func handleTarTypeBlockCharFifo(hdr *tar.Header, path string) error {
	nsOnce.Do(setInUserNS)
	if inUserNS {
		// cannot create a device if running in user namespace
		return nil
	}

	mode := uint32(hdr.Mode & 07777)
	switch hdr.Typeflag {
	case tar.TypeBlock:
		mode |= unix.S_IFBLK
	case tar.TypeChar:
		mode |= unix.S_IFCHR
	case tar.TypeFifo:
		mode |= unix.S_IFIFO
	}

	return unix.Mknod(path, mode, int(unix.Mkdev(uint32(hdr.Devmajor), uint32(hdr.Devminor))))
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

func getxattr(path, attr string) ([]byte, error) {
	b, err := sysx.LGetxattr(path, attr)
	if err == unix.ENOTSUP || err == sysx.ENODATA {
		return nil, nil
	}
	return b, err
}

func setxattr(path, key, value string) error {
	return sysx.LSetxattr(path, key, []byte(value), 0)
}
