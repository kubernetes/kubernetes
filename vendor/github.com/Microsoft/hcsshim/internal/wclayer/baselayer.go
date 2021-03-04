package wclayer

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"syscall"

	"github.com/Microsoft/go-winio"
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/oc"
	"github.com/Microsoft/hcsshim/internal/safefile"
	"go.opencensus.io/trace"
)

type baseLayerWriter struct {
	ctx context.Context
	s   *trace.Span

	root         *os.File
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
func reapplyDirectoryTimes(root *os.File, dis []dirInfo) error {
	for i := range dis {
		di := &dis[len(dis)-i-1] // reverse order: process child directories first
		f, err := safefile.OpenRelative(di.path, root, syscall.GENERIC_READ|syscall.GENERIC_WRITE, syscall.FILE_SHARE_READ, safefile.FILE_OPEN, safefile.FILE_DIRECTORY_FILE|syscall.FILE_FLAG_OPEN_REPARSE_POINT)
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

	var f *os.File
	defer func() {
		if f != nil {
			f.Close()
		}
	}()

	extraFlags := uint32(0)
	if fileInfo.FileAttributes&syscall.FILE_ATTRIBUTE_DIRECTORY != 0 {
		extraFlags |= safefile.FILE_DIRECTORY_FILE
		w.dirInfo = append(w.dirInfo, dirInfo{name, *fileInfo})
	}

	mode := uint32(syscall.GENERIC_READ | syscall.GENERIC_WRITE | winio.WRITE_DAC | winio.WRITE_OWNER | winio.ACCESS_SYSTEM_SECURITY)
	f, err = safefile.OpenRelative(name, w.root, mode, syscall.FILE_SHARE_READ, safefile.FILE_CREATE, extraFlags)
	if err != nil {
		return hcserror.New(err, "Failed to safefile.OpenRelative", name)
	}

	err = winio.SetFileBasicInfo(f, fileInfo)
	if err != nil {
		return hcserror.New(err, "Failed to SetFileBasicInfo", name)
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

	return safefile.LinkRelative(target, w.root, name, w.root)
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

func (w *baseLayerWriter) Close() (err error) {
	defer w.s.End()
	defer func() { oc.SetSpanStatus(w.s, err) }()
	defer func() {
		w.root.Close()
		w.root = nil
	}()

	err = w.closeCurrentFile()
	if err != nil {
		return err
	}
	if w.err == nil {
		// Restore the file times of all the directories, since they may have
		// been modified by creating child directories.
		err = reapplyDirectoryTimes(w.root, w.dirInfo)
		if err != nil {
			return err
		}

		err = ProcessBaseLayer(w.ctx, w.root.Name())
		if err != nil {
			return err
		}

		if w.hasUtilityVM {
			err := safefile.EnsureNotReparsePointRelative("UtilityVM", w.root)
			if err != nil {
				return err
			}
			err = ProcessUtilityVMImage(w.ctx, filepath.Join(w.root.Name(), "UtilityVM"))
			if err != nil {
				return err
			}
		}
	}
	return w.err
}
