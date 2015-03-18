package archive

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/docker/docker/vendor/src/code.google.com/p/go/src/pkg/archive/tar"

	log "github.com/Sirupsen/logrus"
	"github.com/docker/docker/pkg/pools"
	"github.com/docker/docker/pkg/system"
)

type ChangeType int

const (
	ChangeModify = iota
	ChangeAdd
	ChangeDelete
)

type Change struct {
	Path string
	Kind ChangeType
}

func (change *Change) String() string {
	var kind string
	switch change.Kind {
	case ChangeModify:
		kind = "C"
	case ChangeAdd:
		kind = "A"
	case ChangeDelete:
		kind = "D"
	}
	return fmt.Sprintf("%s %s", kind, change.Path)
}

// for sort.Sort
type changesByPath []Change

func (c changesByPath) Less(i, j int) bool { return c[i].Path < c[j].Path }
func (c changesByPath) Len() int           { return len(c) }
func (c changesByPath) Swap(i, j int)      { c[j], c[i] = c[i], c[j] }

// Gnu tar and the go tar writer don't have sub-second mtime
// precision, which is problematic when we apply changes via tar
// files, we handle this by comparing for exact times, *or* same
// second count and either a or b having exactly 0 nanoseconds
func sameFsTime(a, b time.Time) bool {
	return a == b ||
		(a.Unix() == b.Unix() &&
			(a.Nanosecond() == 0 || b.Nanosecond() == 0))
}

func sameFsTimeSpec(a, b syscall.Timespec) bool {
	return a.Sec == b.Sec &&
		(a.Nsec == b.Nsec || a.Nsec == 0 || b.Nsec == 0)
}

// Changes walks the path rw and determines changes for the files in the path,
// with respect to the parent layers
func Changes(layers []string, rw string) ([]Change, error) {
	var changes []Change
	err := filepath.Walk(rw, func(path string, f os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Rebase path
		path, err = filepath.Rel(rw, path)
		if err != nil {
			return err
		}
		path = filepath.Join("/", path)

		// Skip root
		if path == "/" {
			return nil
		}

		// Skip AUFS metadata
		if matched, err := filepath.Match("/.wh..wh.*", path); err != nil || matched {
			return err
		}

		change := Change{
			Path: path,
		}

		// Find out what kind of modification happened
		file := filepath.Base(path)
		// If there is a whiteout, then the file was removed
		if strings.HasPrefix(file, ".wh.") {
			originalFile := file[len(".wh."):]
			change.Path = filepath.Join(filepath.Dir(path), originalFile)
			change.Kind = ChangeDelete
		} else {
			// Otherwise, the file was added
			change.Kind = ChangeAdd

			// ...Unless it already existed in a top layer, in which case, it's a modification
			for _, layer := range layers {
				stat, err := os.Stat(filepath.Join(layer, path))
				if err != nil && !os.IsNotExist(err) {
					return err
				}
				if err == nil {
					// The file existed in the top layer, so that's a modification

					// However, if it's a directory, maybe it wasn't actually modified.
					// If you modify /foo/bar/baz, then /foo will be part of the changed files only because it's the parent of bar
					if stat.IsDir() && f.IsDir() {
						if f.Size() == stat.Size() && f.Mode() == stat.Mode() && sameFsTime(f.ModTime(), stat.ModTime()) {
							// Both directories are the same, don't record the change
							return nil
						}
					}
					change.Kind = ChangeModify
					break
				}
			}
		}

		// Record change
		changes = append(changes, change)
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	return changes, nil
}

type FileInfo struct {
	parent     *FileInfo
	name       string
	stat       *system.Stat_t
	children   map[string]*FileInfo
	capability []byte
	added      bool
}

func (root *FileInfo) LookUp(path string) *FileInfo {
	parent := root
	if path == "/" {
		return root
	}

	pathElements := strings.Split(path, "/")
	for _, elem := range pathElements {
		if elem != "" {
			child := parent.children[elem]
			if child == nil {
				return nil
			}
			parent = child
		}
	}
	return parent
}

func (info *FileInfo) path() string {
	if info.parent == nil {
		return "/"
	}
	return filepath.Join(info.parent.path(), info.name)
}

func (info *FileInfo) isDir() bool {
	return info.parent == nil || info.stat.Mode()&syscall.S_IFDIR == syscall.S_IFDIR
}

func (info *FileInfo) addChanges(oldInfo *FileInfo, changes *[]Change) {

	sizeAtEntry := len(*changes)

	if oldInfo == nil {
		// add
		change := Change{
			Path: info.path(),
			Kind: ChangeAdd,
		}
		*changes = append(*changes, change)
		info.added = true
	}

	// We make a copy so we can modify it to detect additions
	// also, we only recurse on the old dir if the new info is a directory
	// otherwise any previous delete/change is considered recursive
	oldChildren := make(map[string]*FileInfo)
	if oldInfo != nil && info.isDir() {
		for k, v := range oldInfo.children {
			oldChildren[k] = v
		}
	}

	for name, newChild := range info.children {
		oldChild, _ := oldChildren[name]
		if oldChild != nil {
			// change?
			oldStat := oldChild.stat
			newStat := newChild.stat
			// Note: We can't compare inode or ctime or blocksize here, because these change
			// when copying a file into a container. However, that is not generally a problem
			// because any content change will change mtime, and any status change should
			// be visible when actually comparing the stat fields. The only time this
			// breaks down is if some code intentionally hides a change by setting
			// back mtime
			if oldStat.Mode() != newStat.Mode() ||
				oldStat.Uid() != newStat.Uid() ||
				oldStat.Gid() != newStat.Gid() ||
				oldStat.Rdev() != newStat.Rdev() ||
				// Don't look at size for dirs, its not a good measure of change
				(oldStat.Size() != newStat.Size() && oldStat.Mode()&syscall.S_IFDIR != syscall.S_IFDIR) ||
				!sameFsTimeSpec(oldStat.Mtim(), newStat.Mtim()) ||
				bytes.Compare(oldChild.capability, newChild.capability) != 0 {
				change := Change{
					Path: newChild.path(),
					Kind: ChangeModify,
				}
				*changes = append(*changes, change)
				newChild.added = true
			}

			// Remove from copy so we can detect deletions
			delete(oldChildren, name)
		}

		newChild.addChanges(oldChild, changes)
	}
	for _, oldChild := range oldChildren {
		// delete
		change := Change{
			Path: oldChild.path(),
			Kind: ChangeDelete,
		}
		*changes = append(*changes, change)
	}

	// If there were changes inside this directory, we need to add it, even if the directory
	// itself wasn't changed. This is needed to properly save and restore filesystem permissions.
	if len(*changes) > sizeAtEntry && info.isDir() && !info.added && info.path() != "/" {
		change := Change{
			Path: info.path(),
			Kind: ChangeModify,
		}
		// Let's insert the directory entry before the recently added entries located inside this dir
		*changes = append(*changes, change) // just to resize the slice, will be overwritten
		copy((*changes)[sizeAtEntry+1:], (*changes)[sizeAtEntry:])
		(*changes)[sizeAtEntry] = change
	}

}

func (info *FileInfo) Changes(oldInfo *FileInfo) []Change {
	var changes []Change

	info.addChanges(oldInfo, &changes)

	return changes
}

func newRootFileInfo() *FileInfo {
	root := &FileInfo{
		name:     "/",
		children: make(map[string]*FileInfo),
	}
	return root
}

func collectFileInfo(sourceDir string) (*FileInfo, error) {
	root := newRootFileInfo()

	err := filepath.Walk(sourceDir, func(path string, f os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Rebase path
		relPath, err := filepath.Rel(sourceDir, path)
		if err != nil {
			return err
		}
		relPath = filepath.Join("/", relPath)

		if relPath == "/" {
			return nil
		}

		parent := root.LookUp(filepath.Dir(relPath))
		if parent == nil {
			return fmt.Errorf("collectFileInfo: Unexpectedly no parent for %s", relPath)
		}

		info := &FileInfo{
			name:     filepath.Base(relPath),
			children: make(map[string]*FileInfo),
			parent:   parent,
		}

		s, err := system.Lstat(path)
		if err != nil {
			return err
		}
		info.stat = s

		info.capability, _ = system.Lgetxattr(path, "security.capability")

		parent.children[info.name] = info

		return nil
	})
	if err != nil {
		return nil, err
	}
	return root, nil
}

// ChangesDirs compares two directories and generates an array of Change objects describing the changes.
// If oldDir is "", then all files in newDir will be Add-Changes.
func ChangesDirs(newDir, oldDir string) ([]Change, error) {
	var (
		oldRoot, newRoot *FileInfo
		err1, err2       error
		errs             = make(chan error, 2)
	)
	go func() {
		if oldDir != "" {
			oldRoot, err1 = collectFileInfo(oldDir)
		}
		errs <- err1
	}()
	go func() {
		newRoot, err2 = collectFileInfo(newDir)
		errs <- err2
	}()

	// block until both routines have returned
	for i := 0; i < 2; i++ {
		if err := <-errs; err != nil {
			return nil, err
		}
	}

	return newRoot.Changes(oldRoot), nil
}

// ChangesSize calculates the size in bytes of the provided changes, based on newDir.
func ChangesSize(newDir string, changes []Change) int64 {
	var size int64
	for _, change := range changes {
		if change.Kind == ChangeModify || change.Kind == ChangeAdd {
			file := filepath.Join(newDir, change.Path)
			fileInfo, _ := os.Lstat(file)
			if fileInfo != nil && !fileInfo.IsDir() {
				size += fileInfo.Size()
			}
		}
	}
	return size
}

// ExportChanges produces an Archive from the provided changes, relative to dir.
func ExportChanges(dir string, changes []Change) (Archive, error) {
	reader, writer := io.Pipe()
	go func() {
		ta := &tarAppender{
			TarWriter: tar.NewWriter(writer),
			Buffer:    pools.BufioWriter32KPool.Get(nil),
			SeenFiles: make(map[uint64]string),
		}
		// this buffer is needed for the duration of this piped stream
		defer pools.BufioWriter32KPool.Put(ta.Buffer)

		sort.Sort(changesByPath(changes))

		// In general we log errors here but ignore them because
		// during e.g. a diff operation the container can continue
		// mutating the filesystem and we can see transient errors
		// from this
		for _, change := range changes {
			if change.Kind == ChangeDelete {
				whiteOutDir := filepath.Dir(change.Path)
				whiteOutBase := filepath.Base(change.Path)
				whiteOut := filepath.Join(whiteOutDir, ".wh."+whiteOutBase)
				timestamp := time.Now()
				hdr := &tar.Header{
					Name:       whiteOut[1:],
					Size:       0,
					ModTime:    timestamp,
					AccessTime: timestamp,
					ChangeTime: timestamp,
				}
				if err := ta.TarWriter.WriteHeader(hdr); err != nil {
					log.Debugf("Can't write whiteout header: %s", err)
				}
			} else {
				path := filepath.Join(dir, change.Path)
				if err := ta.addTarFile(path, change.Path[1:]); err != nil {
					log.Debugf("Can't add file %s to tar: %s", path, err)
				}
			}
		}

		// Make sure to check the error on Close.
		if err := ta.TarWriter.Close(); err != nil {
			log.Debugf("Can't close layer: %s", err)
		}
		if err := writer.Close(); err != nil {
			log.Debugf("failed close Changes writer: %s", err)
		}
	}()
	return reader, nil
}
