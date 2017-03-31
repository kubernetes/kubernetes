package archive

import (
	"archive/tar"
	"os"
	"sync"
	"syscall"

	"github.com/opencontainers/runc/libcontainer/system"
	"github.com/pkg/errors"
	"github.com/stevvooe/continuity/sysx"
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
		hdr.Devmajor = int64(major(uint64(s.Rdev)))
		hdr.Devminor = int64(minor(uint64(s.Rdev)))
	}

	return nil
}

func major(device uint64) uint64 {
	return (device >> 8) & 0xfff
}

func minor(device uint64) uint64 {
	return (device & 0xff) | ((device >> 12) & 0xfff00)
}

func mkdev(major int64, minor int64) uint32 {
	return uint32(((minor & 0xfff00) << 12) | ((major & 0xfff) << 8) | (minor & 0xff))
}

func open(p string) (*os.File, error) {
	return os.Open(p)
}

func openFile(name string, flag int, perm os.FileMode) (*os.File, error) {
	return os.OpenFile(name, flag, perm)
}

func mkdirAll(path string, perm os.FileMode) error {
	return os.MkdirAll(path, perm)
}

func prepareApply() func() {
	// Unset unmask before doing an apply operation,
	// restore unmask when complete
	oldmask := syscall.Umask(0)
	return func() {
		syscall.Umask(oldmask)
	}
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
		mode |= syscall.S_IFBLK
	case tar.TypeChar:
		mode |= syscall.S_IFCHR
	case tar.TypeFifo:
		mode |= syscall.S_IFIFO
	}

	if err := syscall.Mknod(path, mode, int(mkdev(hdr.Devmajor, hdr.Devminor))); err != nil {
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

func getxattr(path, attr string) ([]byte, error) {
	b, err := sysx.LGetxattr(path, attr)
	if err == syscall.ENOTSUP || err == syscall.ENODATA {
		return nil, nil
	}
	return b, err
}

func setxattr(path, key, value string) error {
	return sysx.LSetxattr(path, key, []byte(value), 0)
}
