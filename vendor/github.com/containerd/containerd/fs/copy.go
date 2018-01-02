package fs

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"

	"github.com/pkg/errors"
)

var (
	bufferPool = &sync.Pool{
		New: func() interface{} {
			return make([]byte, 32*1024)
		},
	}
)

// CopyDir copies the directory from src to dst.
// Most efficient copy of files is attempted.
func CopyDir(dst, src string) error {
	inodes := map[uint64]string{}
	return copyDirectory(dst, src, inodes)
}

func copyDirectory(dst, src string, inodes map[uint64]string) error {
	stat, err := os.Stat(src)
	if err != nil {
		return errors.Wrapf(err, "failed to stat %s", src)
	}
	if !stat.IsDir() {
		return errors.Errorf("source is not directory")
	}

	if st, err := os.Stat(dst); err != nil {
		if err := os.Mkdir(dst, stat.Mode()); err != nil {
			return errors.Wrapf(err, "failed to mkdir %s", dst)
		}
	} else if !st.IsDir() {
		return errors.Errorf("cannot copy to non-directory: %s", dst)
	} else {
		if err := os.Chmod(dst, stat.Mode()); err != nil {
			return errors.Wrapf(err, "failed to chmod on %s", dst)
		}
	}

	fis, err := ioutil.ReadDir(src)
	if err != nil {
		return errors.Wrapf(err, "failed to read %s", src)
	}

	if err := copyFileInfo(stat, dst); err != nil {
		return errors.Wrapf(err, "failed to copy file info for %s", dst)
	}

	for _, fi := range fis {
		source := filepath.Join(src, fi.Name())
		target := filepath.Join(dst, fi.Name())

		switch {
		case fi.IsDir():
			if err := copyDirectory(target, source, inodes); err != nil {
				return err
			}
			continue
		case (fi.Mode() & os.ModeType) == 0:
			link, err := getLinkSource(target, fi, inodes)
			if err != nil {
				return errors.Wrap(err, "failed to get hardlink")
			}
			if link != "" {
				if err := os.Link(link, target); err != nil {
					return errors.Wrap(err, "failed to create hard link")
				}
			} else if err := copyFile(source, target); err != nil {
				return errors.Wrap(err, "failed to copy files")
			}
		case (fi.Mode() & os.ModeSymlink) == os.ModeSymlink:
			link, err := os.Readlink(source)
			if err != nil {
				return errors.Wrapf(err, "failed to read link: %s", source)
			}
			if err := os.Symlink(link, target); err != nil {
				return errors.Wrapf(err, "failed to create symlink: %s", target)
			}
		case (fi.Mode() & os.ModeDevice) == os.ModeDevice:
			if err := copyDevice(target, fi); err != nil {
				return errors.Wrapf(err, "failed to create device")
			}
		default:
			// TODO: Support pipes and sockets
			return errors.Wrapf(err, "unsupported mode %s", fi.Mode())
		}
		if err := copyFileInfo(fi, target); err != nil {
			return errors.Wrap(err, "failed to copy file info")
		}

		if err := copyXAttrs(target, source); err != nil {
			return errors.Wrap(err, "failed to copy xattrs")
		}
	}

	return nil
}

func copyFile(source, target string) error {
	src, err := os.Open(source)
	if err != nil {
		return errors.Wrapf(err, "failed to open source %s", source)
	}
	defer src.Close()
	tgt, err := os.Create(target)
	if err != nil {
		return errors.Wrapf(err, "failed to open target %s", target)
	}
	defer tgt.Close()

	return copyFileContent(tgt, src)
}
