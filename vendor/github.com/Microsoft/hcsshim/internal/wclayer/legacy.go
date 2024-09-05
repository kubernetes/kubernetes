//go:build windows

package wclayer

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
	"github.com/Microsoft/hcsshim/internal/longpath"
	"github.com/Microsoft/hcsshim/internal/safefile"
	"github.com/Microsoft/hcsshim/internal/winapi"
)

var errorIterationCanceled = errors.New("")

var mutatedUtilityVMFiles = map[string]bool{
	`EFI\Microsoft\Boot\BCD`:      true,
	`EFI\Microsoft\Boot\BCD.LOG`:  true,
	`EFI\Microsoft\Boot\BCD.LOG1`: true,
	`EFI\Microsoft\Boot\BCD.LOG2`: true,
}

const (
	filesPath           = `Files`
	HivesPath           = `Hives`
	UtilityVMPath       = `UtilityVM`
	UtilityVMFilesPath  = `UtilityVM\Files`
	RegFilesPath        = `Files\Windows\System32\config`
	BcdFilePath         = `UtilityVM\Files\EFI\Microsoft\Boot\BCD`
	BootMgrFilePath     = `UtilityVM\Files\EFI\Microsoft\Boot\bootmgfw.efi`
	ContainerBaseVhd    = `blank-base.vhdx`
	ContainerScratchVhd = `blank.vhdx`
	UtilityVMBaseVhd    = `SystemTemplateBase.vhdx`
	UtilityVMScratchVhd = `SystemTemplate.vhdx`
	LayoutFileName      = `layout`
	UvmBuildFileName    = `uvmbuildversion`
)

func openFileOrDir(path string, mode uint32, createDisposition uint32) (file *os.File, err error) {
	return winio.OpenForBackup(path, mode, syscall.FILE_SHARE_READ, createDisposition)
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
		return nil, errors.New("invalid tombstones file")
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
	root, err := longpath.LongAbs(r.root)
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
	if err == errorIterationCanceled { //nolint:errorlint // explicitly returned
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
			if errors.Is(err, io.EOF) {
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
		if path == HivesPath || path == filesPath {
			// The Hives directory has a non-deterministic file time because of the
			// nature of the import process. Use the times from System_Delta.
			var g *os.File
			g, err = os.Open(filepath.Join(r.root, HivesPath, `System_Delta`))
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
		fileInfo.FileAttributes = attr
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

func (r *legacyLayerReader) LinkInfo() (uint32, *winio.FileIDInfo, error) {
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
	bufWriter       *bufio.Writer
	currentFileName string
	currentFileRoot *os.File
	backupWriter    *winio.BackupFileWriter
	Tombstones      []string
	HasUtilityVM    bool
	changedDi       []dirInfo
	addedFiles      map[string]bool
	PendingLinks    []pendingLink
	pendingDirs     []pendingDir
	currentIsDir    bool
}

// newLegacyLayerWriter returns a LayerWriter that can write the container layer
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
	w.root, err = safefile.OpenRoot(root)
	if err != nil {
		return
	}
	w.destRoot, err = safefile.OpenRoot(destRoot)
	if err != nil {
		return
	}
	for _, r := range parentRoots {
		f, err := safefile.OpenRoot(r)
		if err != nil {
			return w, err
		}
		w.parentRoots = append(w.parentRoots, f)
	}
	w.bufWriter = bufio.NewWriterSize(io.Discard, 65536)
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
		_ = w.parentRoots[i].Close()
	}
	w.parentRoots = nil
}

func (w *legacyLayerWriter) initUtilityVM() error {
	if !w.HasUtilityVM {
		err := safefile.MkdirRelative(UtilityVMPath, w.destRoot)
		if err != nil {
			return err
		}
		// Server 2016 does not support multiple layers for the utility VM, so
		// clone the utility VM from the parent layer into this layer. Use hard
		// links to avoid unnecessary copying, since most of the files are
		// immutable.
		err = cloneTree(w.parentRoots[0], w.destRoot, UtilityVMFilesPath, mutatedUtilityVMFiles)
		if err != nil {
			return fmt.Errorf("cloning the parent utility VM image failed: %w", err)
		}
		w.HasUtilityVM = true
	}
	return nil
}

func (w *legacyLayerWriter) reset() error {
	err := w.bufWriter.Flush()
	if err != nil {
		return err
	}
	w.bufWriter.Reset(io.Discard)
	if w.currentIsDir {
		r := w.currentFile
		br := winio.NewBackupStreamReader(r)
		// Seek to the beginning of the backup stream, skipping the fileattrs
		if _, err := r.Seek(4, io.SeekStart); err != nil {
			return err
		}

		for {
			bhdr, err := br.Next()
			if errors.Is(err, io.EOF) {
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
				if err := safefile.RemoveRelative(w.currentFileName, w.currentFileRoot); err != nil {
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
	src, err := safefile.OpenRelative(
		subPath,
		srcRoot,
		syscall.GENERIC_READ|winio.ACCESS_SYSTEM_SECURITY,
		syscall.FILE_SHARE_READ,
		winapi.FILE_OPEN,
		winapi.FILE_OPEN_REPARSE_POINT)
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
		extraFlags |= winapi.FILE_DIRECTORY_FILE
	}
	dest, err := safefile.OpenRelative(
		subPath,
		destRoot,
		syscall.GENERIC_READ|syscall.GENERIC_WRITE|winio.WRITE_DAC|winio.WRITE_OWNER|winio.ACCESS_SYSTEM_SECURITY,
		syscall.FILE_SHARE_READ,
		winapi.FILE_CREATE,
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
	err := safefile.EnsureNotReparsePointRelative(subPath, srcRoot)
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
			if isDir {
				di = append(di, dirInfo{path: relPath, fileInfo: *fi})
			}
		} else {
			err = safefile.LinkRelative(relPath, srcRoot, relPath, destRoot)
			if err != nil {
				return err
			}
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

	if name == UtilityVMPath {
		return w.initUtilityVM()
	}

	if (fileInfo.FileAttributes & syscall.FILE_ATTRIBUTE_DIRECTORY) != 0 {
		w.changedDi = append(w.changedDi, dirInfo{path: name, fileInfo: *fileInfo})
	}

	name = filepath.Clean(name)
	if hasPathPrefix(name, UtilityVMPath) {
		if !w.HasUtilityVM {
			return errors.New("missing UtilityVM directory")
		}
		if !hasPathPrefix(name, UtilityVMFilesPath) && name != UtilityVMFilesPath {
			return errors.New("invalid UtilityVM layer")
		}
		createDisposition := uint32(winapi.FILE_OPEN)
		if (fileInfo.FileAttributes & syscall.FILE_ATTRIBUTE_DIRECTORY) != 0 {
			st, err := safefile.LstatRelative(name, w.destRoot)
			if err != nil && !os.IsNotExist(err) {
				return err
			}
			if st != nil {
				// Delete the existing file/directory if it is not the same type as this directory.
				existingAttr := st.Sys().(*syscall.Win32FileAttributeData).FileAttributes
				if (uint32(fileInfo.FileAttributes)^existingAttr)&(syscall.FILE_ATTRIBUTE_DIRECTORY|syscall.FILE_ATTRIBUTE_REPARSE_POINT) != 0 {
					if err = safefile.RemoveAllRelative(name, w.destRoot); err != nil {
						return err
					}
					st = nil
				}
			}
			if st == nil {
				if err = safefile.MkdirRelative(name, w.destRoot); err != nil {
					return err
				}
			}
		} else {
			// Overwrite any existing hard link.
			err := safefile.RemoveRelative(name, w.destRoot)
			if err != nil && !os.IsNotExist(err) {
				return err
			}
			createDisposition = winapi.FILE_CREATE
		}

		f, err := safefile.OpenRelative(
			name,
			w.destRoot,
			syscall.GENERIC_READ|syscall.GENERIC_WRITE|winio.WRITE_DAC|winio.WRITE_OWNER|winio.ACCESS_SYSTEM_SECURITY,
			syscall.FILE_SHARE_READ,
			createDisposition,
			winapi.FILE_OPEN_REPARSE_POINT,
		)
		if err != nil {
			return err
		}
		defer func() {
			if f != nil {
				f.Close()
				_ = safefile.RemoveRelative(name, w.destRoot)
			}
		}()

		err = winio.SetFileBasicInfo(f, fileInfo)
		if err != nil {
			return err
		}

		w.backupWriter = winio.NewBackupFileWriter(f, true)
		w.bufWriter.Reset(w.backupWriter)
		w.currentFile = f
		w.currentFileName = name
		w.currentFileRoot = w.destRoot
		w.addedFiles[name] = true
		f = nil
		return nil
	}

	fname := name
	if (fileInfo.FileAttributes & syscall.FILE_ATTRIBUTE_DIRECTORY) != 0 {
		err := safefile.MkdirRelative(name, w.root)
		if err != nil {
			return err
		}
		fname += ".$wcidirs$"
		w.currentIsDir = true
	}

	f, err := safefile.OpenRelative(fname, w.root, syscall.GENERIC_READ|syscall.GENERIC_WRITE, syscall.FILE_SHARE_READ, winapi.FILE_CREATE, 0)
	if err != nil {
		return err
	}
	defer func() {
		if f != nil {
			f.Close()
			_ = safefile.RemoveRelative(fname, w.root)
		}
	}()

	strippedFi := *fileInfo
	strippedFi.FileAttributes = 0
	err = winio.SetFileBasicInfo(f, &strippedFi)
	if err != nil {
		return err
	}

	if hasPathPrefix(name, HivesPath) {
		w.backupWriter = winio.NewBackupFileWriter(f, false)
		w.bufWriter.Reset(w.backupWriter)
	} else {
		w.bufWriter.Reset(f)
		// The file attributes are written before the stream.
		err = binary.Write(w.bufWriter, binary.LittleEndian, uint32(fileInfo.FileAttributes))
		if err != nil {
			w.bufWriter.Reset(io.Discard)
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
	} else if hasPathPrefix(target, UtilityVMFilesPath) {
		// Since the utility VM is fully cloned into the destination path
		// already, look for cross-layer hard link targets directly in the
		// destination path.
		roots = []*os.File{w.destRoot}
	}

	if roots == nil || (!hasPathPrefix(name, filesPath) && !hasPathPrefix(name, UtilityVMFilesPath)) {
		return errors.New("invalid hard link in layer")
	}

	// Try to find the target of the link in a previously added file. If that
	// fails, search in parent layers.
	var selectedRoot *os.File
	if _, ok := w.addedFiles[target]; ok {
		selectedRoot = w.destRoot
	} else {
		for _, r := range roots {
			if _, err := safefile.LstatRelative(target, r); err != nil {
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
	} else if hasPathPrefix(name, UtilityVMFilesPath) {
		err := w.initUtilityVM()
		if err != nil {
			return err
		}
		// Make sure the path exists; os.RemoveAll will not fail if the file is
		// already gone, and this needs to be a fatal error for diagnostics
		// purposes.
		if _, err := safefile.LstatRelative(name, w.destRoot); err != nil {
			return err
		}
		err = safefile.RemoveAllRelative(name, w.destRoot)
		if err != nil {
			return err
		}
	} else {
		return fmt.Errorf("invalid tombstone %s", name)
	}

	return nil
}

func (w *legacyLayerWriter) Write(b []byte) (int, error) {
	if w.backupWriter == nil && w.currentFile == nil {
		return 0, errors.New("closed")
	}
	return w.bufWriter.Write(b)
}

func (w *legacyLayerWriter) Close() error {
	if err := w.reset(); err != nil {
		return err
	}
	if err := safefile.RemoveRelative("tombstones.txt", w.root); err != nil && !os.IsNotExist(err) {
		return err
	}
	for _, pd := range w.pendingDirs {
		err := safefile.MkdirRelative(pd.Path, pd.Root)
		if err != nil {
			return err
		}
	}
	return nil
}
