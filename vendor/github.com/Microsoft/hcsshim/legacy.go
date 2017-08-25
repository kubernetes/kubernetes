package hcsshim

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/Microsoft/go-winio"
)

var errorIterationCanceled = errors.New("")

var mutatedUtilityVMFiles = map[string]bool{
	`EFI\Microsoft\Boot\BCD`:      true,
	`EFI\Microsoft\Boot\BCD.LOG`:  true,
	`EFI\Microsoft\Boot\BCD.LOG1`: true,
	`EFI\Microsoft\Boot\BCD.LOG2`: true,
}

const (
	filesPath          = `Files`
	hivesPath          = `Hives`
	utilityVMPath      = `UtilityVM`
	utilityVMFilesPath = `UtilityVM\Files`
)

func openFileOrDir(path string, mode uint32, createDisposition uint32) (file *os.File, err error) {
	return winio.OpenForBackup(path, mode, syscall.FILE_SHARE_READ, createDisposition)
}

func makeLongAbsPath(path string) (string, error) {
	if strings.HasPrefix(path, `\\?\`) || strings.HasPrefix(path, `\\.\`) {
		return path, nil
	}
	if !filepath.IsAbs(path) {
		absPath, err := filepath.Abs(path)
		if err != nil {
			return "", err
		}
		path = absPath
	}
	if strings.HasPrefix(path, `\\`) {
		return `\\?\UNC\` + path[2:], nil
	}
	return `\\?\` + path, nil
}

func hasPathPrefix(p, prefix string) bool {
	return strings.HasPrefix(p, prefix) && len(p) > len(prefix) && p[len(prefix)] == '\\'
}

type fileEntry struct {
	path string
	fi   os.FileInfo
	err  error
}

type legacyLayerReader struct {
	root         string
	result       chan *fileEntry
	proceed      chan bool
	currentFile  *os.File
	backupReader *winio.BackupFileReader
}

// newLegacyLayerReader returns a new LayerReader that can read the Windows
// container layer transport format from disk.
func newLegacyLayerReader(root string) *legacyLayerReader {
	r := &legacyLayerReader{
		root:    root,
		result:  make(chan *fileEntry),
		proceed: make(chan bool),
	}
	go r.walk()
	return r
}

func readTombstones(path string) (map[string]([]string), error) {
	tf, err := os.Open(filepath.Join(path, "tombstones.txt"))
	if err != nil {
		return nil, err
	}
	defer tf.Close()
	s := bufio.NewScanner(tf)
	if !s.Scan() || s.Text() != "\xef\xbb\xbfVersion 1.0" {
		return nil, errors.New("Invalid tombstones file")
	}

	ts := make(map[string]([]string))
	for s.Scan() {
		t := filepath.Join(filesPath, s.Text()[1:]) // skip leading `\`
		dir := filepath.Dir(t)
		ts[dir] = append(ts[dir], t)
	}
	if err = s.Err(); err != nil {
		return nil, err
	}

	return ts, nil
}

func (r *legacyLayerReader) walkUntilCancelled() error {
	root, err := makeLongAbsPath(r.root)
	if err != nil {
		return err
	}

	r.root = root
	ts, err := readTombstones(r.root)
	if err != nil {
		return err
	}

	err = filepath.Walk(r.root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if path == r.root || path == filepath.Join(r.root, "tombstones.txt") || strings.HasSuffix(path, ".$wcidirs$") {
			return nil
		}

		r.result <- &fileEntry{path, info, nil}
		if !<-r.proceed {
			return errorIterationCanceled
		}

		// List all the tombstones.
		if info.IsDir() {
			relPath, err := filepath.Rel(r.root, path)
			if err != nil {
				return err
			}
			if dts, ok := ts[relPath]; ok {
				for _, t := range dts {
					r.result <- &fileEntry{filepath.Join(r.root, t), nil, nil}
					if !<-r.proceed {
						return errorIterationCanceled
					}
				}
			}
		}
		return nil
	})
	if err == errorIterationCanceled {
		return nil
	}
	if err == nil {
		return io.EOF
	}
	return err
}

func (r *legacyLayerReader) walk() {
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

func (r *legacyLayerReader) reset() {
	if r.backupReader != nil {
		r.backupReader.Close()
		r.backupReader = nil
	}
	if r.currentFile != nil {
		r.currentFile.Close()
		r.currentFile = nil
	}
}

func findBackupStreamSize(r io.Reader) (int64, error) {
	br := winio.NewBackupStreamReader(r)
	for {
		hdr, err := br.Next()
		if err != nil {
			if err == io.EOF {
				err = nil
			}
			return 0, err
		}
		if hdr.Id == winio.BackupData {
			return hdr.Size, nil
		}
	}
}

func (r *legacyLayerReader) Next() (path string, size int64, fileInfo *winio.FileBasicInfo, err error) {
	r.reset()
	r.proceed <- true
	fe := <-r.result
	if fe == nil {
		err = errors.New("LegacyLayerReader closed")
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

	if fe.fi == nil {
		// This is a tombstone. Return a nil fileInfo.
		return
	}

	if fe.fi.IsDir() && hasPathPrefix(path, filesPath) {
		fe.path += ".$wcidirs$"
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

	if !hasPathPrefix(path, filesPath) {
		size = fe.fi.Size()
		r.backupReader = winio.NewBackupFileReader(f, false)
		if path == hivesPath || path == filesPath {
			// The Hives directory has a non-deterministic file time because of the
			// nature of the import process. Use the times from System_Delta.
			var g *os.File
			g, err = os.Open(filepath.Join(r.root, hivesPath, `System_Delta`))
			if err != nil {
				return
			}
			attr := fileInfo.FileAttributes
			fileInfo, err = winio.GetFileBasicInfo(g)
			g.Close()
			if err != nil {
				return
			}
			fileInfo.FileAttributes = attr
		}

		// The creation time and access time get reset for files outside of the Files path.
		fileInfo.CreationTime = fileInfo.LastWriteTime
		fileInfo.LastAccessTime = fileInfo.LastWriteTime

	} else {
		// The file attributes are written before the backup stream.
		var attr uint32
		err = binary.Read(f, binary.LittleEndian, &attr)
		if err != nil {
			return
		}
		fileInfo.FileAttributes = uintptr(attr)
		beginning := int64(4)

		// Find the accurate file size.
		if !fe.fi.IsDir() {
			size, err = findBackupStreamSize(f)
			if err != nil {
				err = &os.PathError{Op: "findBackupStreamSize", Path: fe.path, Err: err}
				return
			}
		}

		// Return back to the beginning of the backup stream.
		_, err = f.Seek(beginning, 0)
		if err != nil {
			return
		}
	}

	r.currentFile = f
	f = nil
	return
}

func (r *legacyLayerReader) Read(b []byte) (int, error) {
	if r.backupReader == nil {
		if r.currentFile == nil {
			return 0, io.EOF
		}
		return r.currentFile.Read(b)
	}
	return r.backupReader.Read(b)
}

func (r *legacyLayerReader) Seek(offset int64, whence int) (int64, error) {
	if r.backupReader == nil {
		if r.currentFile == nil {
			return 0, errors.New("no current file")
		}
		return r.currentFile.Seek(offset, whence)
	}
	return 0, errors.New("seek not supported on this stream")
}

func (r *legacyLayerReader) Close() error {
	r.proceed <- false
	<-r.result
	r.reset()
	return nil
}

type pendingLink struct {
	Path, Target string
}

type legacyLayerWriter struct {
	root         string
	parentRoots  []string
	destRoot     string
	currentFile  *os.File
	backupWriter *winio.BackupFileWriter
	tombstones   []string
	pathFixed    bool
	HasUtilityVM bool
	uvmDi        []dirInfo
	addedFiles   map[string]bool
	PendingLinks []pendingLink
}

// newLegacyLayerWriter returns a LayerWriter that can write the contaler layer
// transport format to disk.
func newLegacyLayerWriter(root string, parentRoots []string, destRoot string) *legacyLayerWriter {
	return &legacyLayerWriter{
		root:        root,
		parentRoots: parentRoots,
		destRoot:    destRoot,
		addedFiles:  make(map[string]bool),
	}
}

func (w *legacyLayerWriter) init() error {
	if !w.pathFixed {
		path, err := makeLongAbsPath(w.root)
		if err != nil {
			return err
		}
		for i, p := range w.parentRoots {
			w.parentRoots[i], err = makeLongAbsPath(p)
			if err != nil {
				return err
			}
		}
		destPath, err := makeLongAbsPath(w.destRoot)
		if err != nil {
			return err
		}
		w.root = path
		w.destRoot = destPath
		w.pathFixed = true
	}
	return nil
}

func (w *legacyLayerWriter) initUtilityVM() error {
	if !w.HasUtilityVM {
		err := os.Mkdir(filepath.Join(w.destRoot, utilityVMPath), 0)
		if err != nil {
			return err
		}
		// Server 2016 does not support multiple layers for the utility VM, so
		// clone the utility VM from the parent layer into this layer. Use hard
		// links to avoid unnecessary copying, since most of the files are
		// immutable.
		err = cloneTree(filepath.Join(w.parentRoots[0], utilityVMFilesPath), filepath.Join(w.destRoot, utilityVMFilesPath), mutatedUtilityVMFiles)
		if err != nil {
			return fmt.Errorf("cloning the parent utility VM image failed: %s", err)
		}
		w.HasUtilityVM = true
	}
	return nil
}

func (w *legacyLayerWriter) reset() {
	if w.backupWriter != nil {
		w.backupWriter.Close()
		w.backupWriter = nil
	}
	if w.currentFile != nil {
		w.currentFile.Close()
		w.currentFile = nil
	}
}

// copyFileWithMetadata copies a file using the backup/restore APIs in order to preserve metadata
func copyFileWithMetadata(srcPath, destPath string, isDir bool) (fileInfo *winio.FileBasicInfo, err error) {
	createDisposition := uint32(syscall.CREATE_NEW)
	if isDir {
		err = os.Mkdir(destPath, 0)
		if err != nil {
			return nil, err
		}
		createDisposition = syscall.OPEN_EXISTING
	}

	src, err := openFileOrDir(srcPath, syscall.GENERIC_READ|winio.ACCESS_SYSTEM_SECURITY, syscall.OPEN_EXISTING)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	srcr := winio.NewBackupFileReader(src, true)
	defer srcr.Close()

	fileInfo, err = winio.GetFileBasicInfo(src)
	if err != nil {
		return nil, err
	}

	dest, err := openFileOrDir(destPath, syscall.GENERIC_READ|syscall.GENERIC_WRITE|winio.WRITE_DAC|winio.WRITE_OWNER|winio.ACCESS_SYSTEM_SECURITY, createDisposition)
	if err != nil {
		return nil, err
	}
	defer dest.Close()

	err = winio.SetFileBasicInfo(dest, fileInfo)
	if err != nil {
		return nil, err
	}

	destw := winio.NewBackupFileWriter(dest, true)
	defer func() {
		cerr := destw.Close()
		if err == nil {
			err = cerr
		}
	}()

	_, err = io.Copy(destw, srcr)
	if err != nil {
		return nil, err
	}

	return fileInfo, nil
}

// cloneTree clones a directory tree using hard links. It skips hard links for
// the file names in the provided map and just copies those files.
func cloneTree(srcPath, destPath string, mutatedFiles map[string]bool) error {
	var di []dirInfo
	err := filepath.Walk(srcPath, func(srcFilePath string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(srcPath, srcFilePath)
		if err != nil {
			return err
		}
		destFilePath := filepath.Join(destPath, relPath)

		// Directories, reparse points, and files that will be mutated during
		// utility VM import must be copied. All other files can be hard linked.
		isReparsePoint := info.Sys().(*syscall.Win32FileAttributeData).FileAttributes&syscall.FILE_ATTRIBUTE_REPARSE_POINT != 0
		if info.IsDir() || isReparsePoint || mutatedFiles[relPath] {
			fi, err := copyFileWithMetadata(srcFilePath, destFilePath, info.IsDir())
			if err != nil {
				return err
			}
			if info.IsDir() && !isReparsePoint {
				di = append(di, dirInfo{path: destFilePath, fileInfo: *fi})
			}
		} else {
			err = os.Link(srcFilePath, destFilePath)
			if err != nil {
				return err
			}
		}

		// Don't recurse on reparse points.
		if info.IsDir() && isReparsePoint {
			return filepath.SkipDir
		}

		return nil
	})
	if err != nil {
		return err
	}

	return reapplyDirectoryTimes(di)
}

func (w *legacyLayerWriter) Add(name string, fileInfo *winio.FileBasicInfo) error {
	w.reset()
	err := w.init()
	if err != nil {
		return err
	}

	if name == utilityVMPath {
		return w.initUtilityVM()
	}

	if hasPathPrefix(name, utilityVMPath) {
		if !w.HasUtilityVM {
			return errors.New("missing UtilityVM directory")
		}
		if !hasPathPrefix(name, utilityVMFilesPath) && name != utilityVMFilesPath {
			return errors.New("invalid UtilityVM layer")
		}
		path := filepath.Join(w.destRoot, name)
		createDisposition := uint32(syscall.OPEN_EXISTING)
		if (fileInfo.FileAttributes & syscall.FILE_ATTRIBUTE_DIRECTORY) != 0 {
			st, err := os.Lstat(path)
			if err != nil && !os.IsNotExist(err) {
				return err
			}
			if st != nil {
				// Delete the existing file/directory if it is not the same type as this directory.
				existingAttr := st.Sys().(*syscall.Win32FileAttributeData).FileAttributes
				if (uint32(fileInfo.FileAttributes)^existingAttr)&(syscall.FILE_ATTRIBUTE_DIRECTORY|syscall.FILE_ATTRIBUTE_REPARSE_POINT) != 0 {
					if err = os.RemoveAll(path); err != nil {
						return err
					}
					st = nil
				}
			}
			if st == nil {
				if err = os.Mkdir(path, 0); err != nil {
					return err
				}
			}
			if fileInfo.FileAttributes&syscall.FILE_ATTRIBUTE_REPARSE_POINT == 0 {
				w.uvmDi = append(w.uvmDi, dirInfo{path: path, fileInfo: *fileInfo})
			}
		} else {
			// Overwrite any existing hard link.
			err = os.Remove(path)
			if err != nil && !os.IsNotExist(err) {
				return err
			}
			createDisposition = syscall.CREATE_NEW
		}

		f, err := openFileOrDir(path, syscall.GENERIC_READ|syscall.GENERIC_WRITE|winio.WRITE_DAC|winio.WRITE_OWNER|winio.ACCESS_SYSTEM_SECURITY, createDisposition)
		if err != nil {
			return err
		}
		defer func() {
			if f != nil {
				f.Close()
				os.Remove(path)
			}
		}()

		err = winio.SetFileBasicInfo(f, fileInfo)
		if err != nil {
			return err
		}

		w.backupWriter = winio.NewBackupFileWriter(f, true)
		w.currentFile = f
		w.addedFiles[name] = true
		f = nil
		return nil
	}

	path := filepath.Join(w.root, name)
	if (fileInfo.FileAttributes & syscall.FILE_ATTRIBUTE_DIRECTORY) != 0 {
		err := os.Mkdir(path, 0)
		if err != nil {
			return err
		}
		path += ".$wcidirs$"
	}

	f, err := openFileOrDir(path, syscall.GENERIC_READ|syscall.GENERIC_WRITE, syscall.CREATE_NEW)
	if err != nil {
		return err
	}
	defer func() {
		if f != nil {
			f.Close()
			os.Remove(path)
		}
	}()

	strippedFi := *fileInfo
	strippedFi.FileAttributes = 0
	err = winio.SetFileBasicInfo(f, &strippedFi)
	if err != nil {
		return err
	}

	if hasPathPrefix(name, hivesPath) {
		w.backupWriter = winio.NewBackupFileWriter(f, false)
	} else {
		// The file attributes are written before the stream.
		err = binary.Write(f, binary.LittleEndian, uint32(fileInfo.FileAttributes))
		if err != nil {
			return err
		}
	}

	w.currentFile = f
	w.addedFiles[name] = true
	f = nil
	return nil
}

func (w *legacyLayerWriter) AddLink(name string, target string) error {
	w.reset()
	err := w.init()
	if err != nil {
		return err
	}

	var roots []string
	if hasPathPrefix(target, filesPath) {
		// Look for cross-layer hard link targets in the parent layers, since
		// nothing is in the destination path yet.
		roots = w.parentRoots
	} else if hasPathPrefix(target, utilityVMFilesPath) {
		// Since the utility VM is fully cloned into the destination path
		// already, look for cross-layer hard link targets directly in the
		// destination path.
		roots = []string{w.destRoot}
	}

	if roots == nil || (!hasPathPrefix(name, filesPath) && !hasPathPrefix(name, utilityVMFilesPath)) {
		return errors.New("invalid hard link in layer")
	}

	// Find to try the target of the link in a previously added file. If that
	// fails, search in parent layers.
	var selectedRoot string
	if _, ok := w.addedFiles[target]; ok {
		selectedRoot = w.destRoot
	} else {
		for _, r := range roots {
			if _, err = os.Lstat(filepath.Join(r, target)); err != nil {
				if !os.IsNotExist(err) {
					return err
				}
			} else {
				selectedRoot = r
				break
			}
		}
		if selectedRoot == "" {
			return fmt.Errorf("failed to find link target for '%s' -> '%s'", name, target)
		}
	}
	// The link can't be written until after the ImportLayer call.
	w.PendingLinks = append(w.PendingLinks, pendingLink{
		Path:   filepath.Join(w.destRoot, name),
		Target: filepath.Join(selectedRoot, target),
	})
	w.addedFiles[name] = true
	return nil
}

func (w *legacyLayerWriter) Remove(name string) error {
	if hasPathPrefix(name, filesPath) {
		w.tombstones = append(w.tombstones, name[len(filesPath)+1:])
	} else if hasPathPrefix(name, utilityVMFilesPath) {
		err := w.initUtilityVM()
		if err != nil {
			return err
		}
		// Make sure the path exists; os.RemoveAll will not fail if the file is
		// already gone, and this needs to be a fatal error for diagnostics
		// purposes.
		path := filepath.Join(w.destRoot, name)
		if _, err := os.Lstat(path); err != nil {
			return err
		}
		err = os.RemoveAll(path)
		if err != nil {
			return err
		}
	} else {
		return fmt.Errorf("invalid tombstone %s", name)
	}

	return nil
}

func (w *legacyLayerWriter) Write(b []byte) (int, error) {
	if w.backupWriter == nil {
		if w.currentFile == nil {
			return 0, errors.New("closed")
		}
		return w.currentFile.Write(b)
	}
	return w.backupWriter.Write(b)
}

func (w *legacyLayerWriter) Close() error {
	w.reset()
	err := w.init()
	if err != nil {
		return err
	}
	tf, err := os.Create(filepath.Join(w.root, "tombstones.txt"))
	if err != nil {
		return err
	}
	defer tf.Close()
	_, err = tf.Write([]byte("\xef\xbb\xbfVersion 1.0\n"))
	if err != nil {
		return err
	}
	for _, t := range w.tombstones {
		_, err = tf.Write([]byte(filepath.Join(`\`, t) + "\n"))
		if err != nil {
			return err
		}
	}
	if w.HasUtilityVM {
		err = reapplyDirectoryTimes(w.uvmDi)
		if err != nil {
			return err
		}
	}
	return nil
}
