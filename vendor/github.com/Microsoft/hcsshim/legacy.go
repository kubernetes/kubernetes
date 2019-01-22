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

		// Indirect fix for https://github.com/moby/moby/issues/32838#issuecomment-343610048.
		// Handle failure from what may be a golang bug in the conversion of
		// UTF16 to UTF8 in files which are left in the recycle bin. Os.Lstat
		// which is called by filepath.Walk will fail when a filename contains
		// unicode characters. Skip the recycle bin regardless which is goodness.
		if strings.EqualFold(path, filepath.Join(r.root, `Files\$Recycle.Bin`)) && info.IsDir() {
			return filepath.SkipDir
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
	TargetRoot   *os.File
}

type pendingDir struct {
	Path string
	Root *os.File
}

type legacyLayerWriter struct {
	root            *os.File
	destRoot        *os.File
	parentRoots     []*os.File
	currentFile     *os.File
	currentFileName string
	currentFileRoot *os.File
	backupWriter    *winio.BackupFileWriter
	Tombstones      []string
	HasUtilityVM    bool
	uvmDi           []dirInfo
	addedFiles      map[string]bool
	PendingLinks    []pendingLink
	pendingDirs     []pendingDir
	currentIsDir    bool
}

// newLegacyLayerWriter returns a LayerWriter that can write the contaler layer
// transport format to disk.
func newLegacyLayerWriter(root string, parentRoots []string, destRoot string) (w *legacyLayerWriter, err error) {
	w = &legacyLayerWriter{
		addedFiles: make(map[string]bool),
	}
	defer func() {
		if err != nil {
			w.CloseRoots()
			w = nil
		}
	}()
	w.root, err = openRoot(root)
	if err != nil {
		return
	}
	w.destRoot, err = openRoot(destRoot)
	if err != nil {
		return
	}
	for _, r := range parentRoots {
		f, err := openRoot(r)
		if err != nil {
			return w, err
		}
		w.parentRoots = append(w.parentRoots, f)
	}
	return
}

func (w *legacyLayerWriter) CloseRoots() {
	if w.root != nil {
		w.root.Close()
		w.root = nil
	}
	if w.destRoot != nil {
		w.destRoot.Close()
		w.destRoot = nil
	}
	for i := range w.parentRoots {
		w.parentRoots[i].Close()
	}
	w.parentRoots = nil
}

func (w *legacyLayerWriter) initUtilityVM() error {
	if !w.HasUtilityVM {
		err := mkdirRelative(utilityVMPath, w.destRoot)
		if err != nil {
			return err
		}
		// Server 2016 does not support multiple layers for the utility VM, so
		// clone the utility VM from the parent layer into this layer. Use hard
		// links to avoid unnecessary copying, since most of the files are
		// immutable.
		err = cloneTree(w.parentRoots[0], w.destRoot, utilityVMFilesPath, mutatedUtilityVMFiles)
		if err != nil {
			return fmt.Errorf("cloning the parent utility VM image failed: %s", err)
		}
		w.HasUtilityVM = true
	}
	return nil
}

func (w *legacyLayerWriter) reset() error {
	if w.currentIsDir {
		r := w.currentFile
		br := winio.NewBackupStreamReader(r)
		// Seek to the beginning of the backup stream, skipping the fileattrs
		if _, err := r.Seek(4, io.SeekStart); err != nil {
			return err
		}

		for {
			bhdr, err := br.Next()
			if err == io.EOF {
				// end of backupstream data
				break
			}
			if err != nil {
				return err
			}
			switch bhdr.Id {
			case winio.BackupReparseData:
				// The current file is a `.$wcidirs$` metadata file that
				// describes a directory reparse point. Delete the placeholder
				// directory to prevent future files being added into the
				// destination of the reparse point during the ImportLayer call
				if err := removeRelative(w.currentFileName, w.currentFileRoot); err != nil {
					return err
				}
				w.pendingDirs = append(w.pendingDirs, pendingDir{Path: w.currentFileName, Root: w.currentFileRoot})
			default:
				// ignore all other stream types, as we only care about directory reparse points
			}
		}
		w.currentIsDir = false
	}
	if w.backupWriter != nil {
		w.backupWriter.Close()
		w.backupWriter = nil
	}
	if w.currentFile != nil {
		w.currentFile.Close()
		w.currentFile = nil
		w.currentFileName = ""
		w.currentFileRoot = nil
	}
	return nil
}

// copyFileWithMetadata copies a file using the backup/restore APIs in order to preserve metadata
func copyFileWithMetadata(srcRoot, destRoot *os.File, subPath string, isDir bool) (fileInfo *winio.FileBasicInfo, err error) {
	src, err := openRelative(
		subPath,
		srcRoot,
		syscall.GENERIC_READ|winio.ACCESS_SYSTEM_SECURITY,
		syscall.FILE_SHARE_READ,
		_FILE_OPEN,
		_FILE_OPEN_REPARSE_POINT)
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

	extraFlags := uint32(0)
	if isDir {
		extraFlags |= _FILE_DIRECTORY_FILE
	}
	dest, err := openRelative(
		subPath,
		destRoot,
		syscall.GENERIC_READ|syscall.GENERIC_WRITE|winio.WRITE_DAC|winio.WRITE_OWNER|winio.ACCESS_SYSTEM_SECURITY,
		syscall.FILE_SHARE_READ,
		_FILE_CREATE,
		extraFlags)
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
func cloneTree(srcRoot *os.File, destRoot *os.File, subPath string, mutatedFiles map[string]bool) error {
	var di []dirInfo
	err := ensureNotReparsePointRelative(subPath, srcRoot)
	if err != nil {
		return err
	}
	err = filepath.Walk(filepath.Join(srcRoot.Name(), subPath), func(srcFilePath string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(srcRoot.Name(), srcFilePath)
		if err != nil {
			return err
		}

		fileAttributes := info.Sys().(*syscall.Win32FileAttributeData).FileAttributes
		// Directories, reparse points, and files that will be mutated during
		// utility VM import must be copied. All other files can be hard linked.
		isReparsePoint := fileAttributes&syscall.FILE_ATTRIBUTE_REPARSE_POINT != 0
		// In go1.9, FileInfo.IsDir() returns false if the directory is also a symlink.
		// See: https://github.com/golang/go/commit/1989921aef60c83e6f9127a8448fb5ede10e9acc
		// Fixes the problem by checking syscall.FILE_ATTRIBUTE_DIRECTORY directly
		isDir := fileAttributes&syscall.FILE_ATTRIBUTE_DIRECTORY != 0

		if isDir || isReparsePoint || mutatedFiles[relPath] {
			fi, err := copyFileWithMetadata(srcRoot, destRoot, relPath, isDir)
			if err != nil {
				return err
			}
			if isDir && !isReparsePoint {
				di = append(di, dirInfo{path: relPath, fileInfo: *fi})
			}
		} else {
			err = linkRelative(relPath, srcRoot, relPath, destRoot)
			if err != nil {
				return err
			}
		}

		// Don't recurse on reparse points in go1.8 and older. Filepath.Walk
		// handles this in go1.9 and newer.
		if isDir && isReparsePoint && shouldSkipDirectoryReparse {
			return filepath.SkipDir
		}

		return nil
	})
	if err != nil {
		return err
	}

	return reapplyDirectoryTimes(destRoot, di)
}

func (w *legacyLayerWriter) Add(name string, fileInfo *winio.FileBasicInfo) error {
	if err := w.reset(); err != nil {
		return err
	}

	if name == utilityVMPath {
		return w.initUtilityVM()
	}

	name = filepath.Clean(name)
	if hasPathPrefix(name, utilityVMPath) {
		if !w.HasUtilityVM {
			return errors.New("missing UtilityVM directory")
		}
		if !hasPathPrefix(name, utilityVMFilesPath) && name != utilityVMFilesPath {
			return errors.New("invalid UtilityVM layer")
		}
		createDisposition := uint32(_FILE_OPEN)
		if (fileInfo.FileAttributes & syscall.FILE_ATTRIBUTE_DIRECTORY) != 0 {
			st, err := lstatRelative(name, w.destRoot)
			if err != nil && !os.IsNotExist(err) {
				return err
			}
			if st != nil {
				// Delete the existing file/directory if it is not the same type as this directory.
				existingAttr := st.Sys().(*syscall.Win32FileAttributeData).FileAttributes
				if (uint32(fileInfo.FileAttributes)^existingAttr)&(syscall.FILE_ATTRIBUTE_DIRECTORY|syscall.FILE_ATTRIBUTE_REPARSE_POINT) != 0 {
					if err = removeAllRelative(name, w.destRoot); err != nil {
						return err
					}
					st = nil
				}
			}
			if st == nil {
				if err = mkdirRelative(name, w.destRoot); err != nil {
					return err
				}
			}
			if fileInfo.FileAttributes&syscall.FILE_ATTRIBUTE_REPARSE_POINT == 0 {
				w.uvmDi = append(w.uvmDi, dirInfo{path: name, fileInfo: *fileInfo})
			}
		} else {
			// Overwrite any existing hard link.
			err := removeRelative(name, w.destRoot)
			if err != nil && !os.IsNotExist(err) {
				return err
			}
			createDisposition = _FILE_CREATE
		}

		f, err := openRelative(
			name,
			w.destRoot,
			syscall.GENERIC_READ|syscall.GENERIC_WRITE|winio.WRITE_DAC|winio.WRITE_OWNER|winio.ACCESS_SYSTEM_SECURITY,
			syscall.FILE_SHARE_READ,
			createDisposition,
			_FILE_OPEN_REPARSE_POINT,
		)
		if err != nil {
			return err
		}
		defer func() {
			if f != nil {
				f.Close()
				removeRelative(name, w.destRoot)
			}
		}()

		err = winio.SetFileBasicInfo(f, fileInfo)
		if err != nil {
			return err
		}

		w.backupWriter = winio.NewBackupFileWriter(f, true)
		w.currentFile = f
		w.currentFileName = name
		w.currentFileRoot = w.destRoot
		w.addedFiles[name] = true
		f = nil
		return nil
	}

	fname := name
	if (fileInfo.FileAttributes & syscall.FILE_ATTRIBUTE_DIRECTORY) != 0 {
		err := mkdirRelative(name, w.root)
		if err != nil {
			return err
		}
		fname += ".$wcidirs$"
		w.currentIsDir = true
	}

	f, err := openRelative(fname, w.root, syscall.GENERIC_READ|syscall.GENERIC_WRITE, syscall.FILE_SHARE_READ, _FILE_CREATE, 0)
	if err != nil {
		return err
	}
	defer func() {
		if f != nil {
			f.Close()
			removeRelative(fname, w.root)
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
	w.currentFileName = name
	w.currentFileRoot = w.root
	w.addedFiles[name] = true
	f = nil
	return nil
}

func (w *legacyLayerWriter) AddLink(name string, target string) error {
	if err := w.reset(); err != nil {
		return err
	}

	target = filepath.Clean(target)
	var roots []*os.File
	if hasPathPrefix(target, filesPath) {
		// Look for cross-layer hard link targets in the parent layers, since
		// nothing is in the destination path yet.
		roots = w.parentRoots
	} else if hasPathPrefix(target, utilityVMFilesPath) {
		// Since the utility VM is fully cloned into the destination path
		// already, look for cross-layer hard link targets directly in the
		// destination path.
		roots = []*os.File{w.destRoot}
	}

	if roots == nil || (!hasPathPrefix(name, filesPath) && !hasPathPrefix(name, utilityVMFilesPath)) {
		return errors.New("invalid hard link in layer")
	}

	// Find to try the target of the link in a previously added file. If that
	// fails, search in parent layers.
	var selectedRoot *os.File
	if _, ok := w.addedFiles[target]; ok {
		selectedRoot = w.destRoot
	} else {
		for _, r := range roots {
			if _, err := lstatRelative(target, r); err != nil {
				if !os.IsNotExist(err) {
					return err
				}
			} else {
				selectedRoot = r
				break
			}
		}
		if selectedRoot == nil {
			return fmt.Errorf("failed to find link target for '%s' -> '%s'", name, target)
		}
	}

	// The link can't be written until after the ImportLayer call.
	w.PendingLinks = append(w.PendingLinks, pendingLink{
		Path:       name,
		Target:     target,
		TargetRoot: selectedRoot,
	})
	w.addedFiles[name] = true
	return nil
}

func (w *legacyLayerWriter) Remove(name string) error {
	name = filepath.Clean(name)
	if hasPathPrefix(name, filesPath) {
		w.Tombstones = append(w.Tombstones, name)
	} else if hasPathPrefix(name, utilityVMFilesPath) {
		err := w.initUtilityVM()
		if err != nil {
			return err
		}
		// Make sure the path exists; os.RemoveAll will not fail if the file is
		// already gone, and this needs to be a fatal error for diagnostics
		// purposes.
		if _, err := lstatRelative(name, w.destRoot); err != nil {
			return err
		}
		err = removeAllRelative(name, w.destRoot)
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
	if err := w.reset(); err != nil {
		return err
	}
	if err := removeRelative("tombstones.txt", w.root); err != nil && !os.IsNotExist(err) {
		return err
	}
	for _, pd := range w.pendingDirs {
		err := mkdirRelative(pd.Path, pd.Root)
		if err != nil {
			return err
		}
	}
	if w.HasUtilityVM {
		err := reapplyDirectoryTimes(w.destRoot, w.uvmDi)
		if err != nil {
			return err
		}
	}
	return nil
}
