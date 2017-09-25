package hcsshim

import (
	"errors"
	"os"
	"path/filepath"
	"syscall"

	"github.com/Microsoft/go-winio"
)

type baseLayerWriter struct {
	root         string
	f            *os.File
	bw           *winio.BackupFileWriter
	err          error
	hasUtilityVM bool
	dirInfo      []dirInfo
}

type dirInfo struct {
	path     string
	fileInfo winio.FileBasicInfo
}

// reapplyDirectoryTimes reapplies directory modification, creation, etc. times
// after processing of the directory tree has completed. The times are expected
// to be ordered such that parent directories come before child directories.
func reapplyDirectoryTimes(dis []dirInfo) error {
	for i := range dis {
		di := &dis[len(dis)-i-1] // reverse order: process child directories first
		f, err := winio.OpenForBackup(di.path, syscall.GENERIC_READ|syscall.GENERIC_WRITE, syscall.FILE_SHARE_READ, syscall.OPEN_EXISTING)
		if err != nil {
			return err
		}

		err = winio.SetFileBasicInfo(f, &di.fileInfo)
		f.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

func (w *baseLayerWriter) closeCurrentFile() error {
	if w.f != nil {
		err := w.bw.Close()
		err2 := w.f.Close()
		w.f = nil
		w.bw = nil
		if err != nil {
			return err
		}
		if err2 != nil {
			return err2
		}
	}
	return nil
}

func (w *baseLayerWriter) Add(name string, fileInfo *winio.FileBasicInfo) (err error) {
	defer func() {
		if err != nil {
			w.err = err
		}
	}()

	err = w.closeCurrentFile()
	if err != nil {
		return err
	}

	if filepath.ToSlash(name) == `UtilityVM/Files` {
		w.hasUtilityVM = true
	}

	path := filepath.Join(w.root, name)
	path, err = makeLongAbsPath(path)
	if err != nil {
		return err
	}

	var f *os.File
	defer func() {
		if f != nil {
			f.Close()
		}
	}()

	createmode := uint32(syscall.CREATE_NEW)
	if fileInfo.FileAttributes&syscall.FILE_ATTRIBUTE_DIRECTORY != 0 {
		err := os.Mkdir(path, 0)
		if err != nil && !os.IsExist(err) {
			return err
		}
		createmode = syscall.OPEN_EXISTING
		if fileInfo.FileAttributes&syscall.FILE_ATTRIBUTE_REPARSE_POINT == 0 {
			w.dirInfo = append(w.dirInfo, dirInfo{path, *fileInfo})
		}
	}

	mode := uint32(syscall.GENERIC_READ | syscall.GENERIC_WRITE | winio.WRITE_DAC | winio.WRITE_OWNER | winio.ACCESS_SYSTEM_SECURITY)
	f, err = winio.OpenForBackup(path, mode, syscall.FILE_SHARE_READ, createmode)
	if err != nil {
		return makeError(err, "Failed to OpenForBackup", path)
	}

	err = winio.SetFileBasicInfo(f, fileInfo)
	if err != nil {
		return makeError(err, "Failed to SetFileBasicInfo", path)
	}

	w.f = f
	w.bw = winio.NewBackupFileWriter(f, true)
	f = nil
	return nil
}

func (w *baseLayerWriter) AddLink(name string, target string) (err error) {
	defer func() {
		if err != nil {
			w.err = err
		}
	}()

	err = w.closeCurrentFile()
	if err != nil {
		return err
	}

	linkpath, err := makeLongAbsPath(filepath.Join(w.root, name))
	if err != nil {
		return err
	}

	linktarget, err := makeLongAbsPath(filepath.Join(w.root, target))
	if err != nil {
		return err
	}

	return os.Link(linktarget, linkpath)
}

func (w *baseLayerWriter) Remove(name string) error {
	return errors.New("base layer cannot have tombstones")
}

func (w *baseLayerWriter) Write(b []byte) (int, error) {
	n, err := w.bw.Write(b)
	if err != nil {
		w.err = err
	}
	return n, err
}

func (w *baseLayerWriter) Close() error {
	err := w.closeCurrentFile()
	if err != nil {
		return err
	}
	if w.err == nil {
		// Restore the file times of all the directories, since they may have
		// been modified by creating child directories.
		err = reapplyDirectoryTimes(w.dirInfo)
		if err != nil {
			return err
		}

		err = ProcessBaseLayer(w.root)
		if err != nil {
			return err
		}

		if w.hasUtilityVM {
			err = ProcessUtilityVMImage(filepath.Join(w.root, "UtilityVM"))
			if err != nil {
				return err
			}
		}
	}
	return w.err
}
