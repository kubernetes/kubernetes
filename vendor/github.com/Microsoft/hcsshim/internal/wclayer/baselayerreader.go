//go:build windows

package wclayer

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/Microsoft/go-winio"
	"github.com/Microsoft/hcsshim/internal/longpath"
	"github.com/Microsoft/hcsshim/internal/oc"
	"go.opencensus.io/trace"
)

type baseLayerReader struct {
	s            *trace.Span
	root         string
	result       chan *fileEntry
	proceed      chan bool
	currentFile  *os.File
	backupReader *winio.BackupFileReader
}

func newBaseLayerReader(root string, s *trace.Span) (r *baseLayerReader) {
	r = &baseLayerReader{
		s:       s,
		root:    root,
		result:  make(chan *fileEntry),
		proceed: make(chan bool),
	}
	go r.walk()
	return r
}

func (r *baseLayerReader) walkUntilCancelled() error {
	root, err := longpath.LongAbs(r.root)
	if err != nil {
		return err
	}

	r.root = root

	err = filepath.Walk(filepath.Join(r.root, filesPath), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Indirect fix for https://github.com/moby/moby/issues/32838#issuecomment-343610048.
		// Handle failure from what may be a golang bug in the conversion of
		// UTF16 to UTF8 in files which are left in the recycle bin. Os.Lstat
		// which is called by filepath.Walk will fail when a filename contains
		// unicode characters. Skip the recycle bin regardless which is goodness.
		if strings.EqualFold(path, filepath.Join(r.root, `Files\$Recycle.Bin`)) && info.IsDir() {
			return filepath.SkipDir
		}

		r.result <- &fileEntry{path, info, nil}
		if !<-r.proceed {
			return errorIterationCanceled
		}

		return nil
	})

	if err == errorIterationCanceled { //nolint:errorlint // explicitly returned
		return nil
	}

	if err != nil {
		return err
	}

	utilityVMAbsPath := filepath.Join(r.root, UtilityVMPath)
	utilityVMFilesAbsPath := filepath.Join(r.root, UtilityVMFilesPath)

	// Ignore a UtilityVM without Files, that's not _really_ a UtiltyVM
	if _, err = os.Lstat(utilityVMFilesAbsPath); err != nil {
		if os.IsNotExist(err) {
			return io.EOF
		}
		return err
	}

	err = filepath.Walk(utilityVMAbsPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if path != utilityVMAbsPath && path != utilityVMFilesAbsPath && !hasPathPrefix(path, utilityVMFilesAbsPath) {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		r.result <- &fileEntry{path, info, nil}
		if !<-r.proceed {
			return errorIterationCanceled
		}

		return nil
	})

	if err == errorIterationCanceled { //nolint:errorlint // explicitly returned
		return nil
	}

	if err != nil {
		return err
	}

	return io.EOF
}

func (r *baseLayerReader) walk() {
	defer close(r.result)
	if !<-r.proceed {
		return
	}

	err := r.walkUntilCancelled()
	if err != nil {
		for {
			r.result <- &fileEntry{err: err}
			if !<-r.proceed {
				return
			}
		}
	}
}

func (r *baseLayerReader) reset() {
	if r.backupReader != nil {
		r.backupReader.Close()
		r.backupReader = nil
	}
	if r.currentFile != nil {
		r.currentFile.Close()
		r.currentFile = nil
	}
}

func (r *baseLayerReader) Next() (path string, size int64, fileInfo *winio.FileBasicInfo, err error) {
	r.reset()
	r.proceed <- true
	fe := <-r.result
	if fe == nil {
		err = errors.New("BaseLayerReader closed")
		return
	}
	if fe.err != nil {
		err = fe.err
		return
	}

	path, err = filepath.Rel(r.root, fe.path)
	if err != nil {
		return
	}

	f, err := openFileOrDir(fe.path, syscall.GENERIC_READ, syscall.OPEN_EXISTING)
	if err != nil {
		return
	}
	defer func() {
		if f != nil {
			f.Close()
		}
	}()

	fileInfo, err = winio.GetFileBasicInfo(f)
	if err != nil {
		return
	}

	size = fe.fi.Size()
	r.backupReader = winio.NewBackupFileReader(f, true)

	r.currentFile = f
	f = nil
	return
}

func (r *baseLayerReader) LinkInfo() (uint32, *winio.FileIDInfo, error) {
	fileStandardInfo, err := winio.GetFileStandardInfo(r.currentFile)
	if err != nil {
		return 0, nil, err
	}
	fileIDInfo, err := winio.GetFileID(r.currentFile)
	if err != nil {
		return 0, nil, err
	}
	return fileStandardInfo.NumberOfLinks, fileIDInfo, nil
}

func (r *baseLayerReader) Read(b []byte) (int, error) {
	if r.backupReader == nil {
		return 0, io.EOF
	}
	return r.backupReader.Read(b)
}

func (r *baseLayerReader) Close() (err error) {
	defer r.s.End()
	defer func() {
		oc.SetSpanStatus(r.s, err)
		close(r.proceed)
	}()
	r.proceed <- false
	// The r.result channel will be closed once walk() returns
	<-r.result
	r.reset()
	return nil
}
