package fs

import (
	"io"
	"os"
	"syscall"

	"github.com/pkg/errors"
	"github.com/stevvooe/continuity/sysx"
)

func copyFileInfo(fi os.FileInfo, name string) error {
	st := fi.Sys().(*syscall.Stat_t)
	if err := os.Lchown(name, int(st.Uid), int(st.Gid)); err != nil {
		return errors.Wrapf(err, "failed to chown %s", name)
	}

	if (fi.Mode() & os.ModeSymlink) != os.ModeSymlink {
		if err := os.Chmod(name, fi.Mode()); err != nil {
			return errors.Wrapf(err, "failed to chmod %s", name)
		}
	}

	if err := syscall.UtimesNano(name, []syscall.Timespec{st.Atim, st.Mtim}); err != nil {
		return errors.Wrapf(err, "failed to utime %s", name)
	}

	return nil
}

func copyFileContent(dst, src *os.File) error {
	st, err := src.Stat()
	if err != nil {
		return errors.Wrap(err, "unable to stat source")
	}

	n, err := sysx.CopyFileRange(src.Fd(), nil, dst.Fd(), nil, int(st.Size()), 0)
	if err != nil {
		if err != syscall.ENOSYS && err != syscall.EXDEV {
			return errors.Wrap(err, "copy file range failed")
		}

		buf := bufferPool.Get().([]byte)
		_, err = io.CopyBuffer(dst, src, buf)
		bufferPool.Put(buf)
		return err
	}

	if int64(n) != st.Size() {
		return errors.Wrapf(err, "short copy: %d of %d", int64(n), st.Size())
	}

	return nil
}

func copyXAttrs(dst, src string) error {
	xattrKeys, err := sysx.LListxattr(src)
	if err != nil {
		return errors.Wrapf(err, "failed to list xattrs on %s", src)
	}
	for _, xattr := range xattrKeys {
		data, err := sysx.LGetxattr(src, xattr)
		if err != nil {
			return errors.Wrapf(err, "failed to get xattr %q on %s", xattr, src)
		}
		if err := sysx.LSetxattr(dst, xattr, data, 0); err != nil {
			return errors.Wrapf(err, "failed to set xattr %q on %s", xattr, dst)
		}
	}

	return nil
}

func copyDevice(dst string, fi os.FileInfo) error {
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return errors.New("unsupported stat type")
	}
	return syscall.Mknod(dst, uint32(fi.Mode()), int(st.Rdev))
}
