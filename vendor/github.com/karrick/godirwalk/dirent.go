package godirwalk

import (
	"os"
	"path/filepath"
)

// Dirent stores the name and file system mode type of discovered file system
// entries.
type Dirent struct {
	name     string      // base name of the file system entry.
	path     string      // path name of the file system entry.
	modeType os.FileMode // modeType is the type of file system entry.
}

// NewDirent returns a newly initialized Dirent structure, or an error.  This
// function does not follow symbolic links.
//
// This function is rarely used, as Dirent structures are provided by other
// functions in this library that read and walk directories, but is provided,
// however, for the occasion when a program needs to create a Dirent.
func NewDirent(osPathname string) (*Dirent, error) {
	modeType, err := modeType(osPathname)
	if err != nil {
		return nil, err
	}
	return &Dirent{
		name:     filepath.Base(osPathname),
		path:     filepath.Dir(osPathname),
		modeType: modeType,
	}, nil
}

// IsDir returns true if and only if the Dirent represents a file system
// directory.  Note that on some operating systems, more than one file mode bit
// may be set for a node.  For instance, on Windows, a symbolic link that points
// to a directory will have both the directory and the symbolic link bits set.
func (de Dirent) IsDir() bool { return de.modeType&os.ModeDir != 0 }

// IsDirOrSymlinkToDir returns true if and only if the Dirent represents a file
// system directory, or a symbolic link to a directory. Note that if the Dirent
// is not a directory but is a symbolic link, this method will resolve by
// sending a request to the operating system to follow the symbolic link.
func (de Dirent) IsDirOrSymlinkToDir() (bool, error) {
	if de.IsDir() {
		return true, nil
	}
	if !de.IsSymlink() {
		return false, nil
	}
	// Does this symlink point to a directory?
	info, err := os.Stat(filepath.Join(de.path, de.name))
	if err != nil {
		return false, err
	}
	return info.IsDir(), nil
}

// IsRegular returns true if and only if the Dirent represents a regular file.
// That is, it ensures that no mode type bits are set.
func (de Dirent) IsRegular() bool { return de.modeType&os.ModeType == 0 }

// IsSymlink returns true if and only if the Dirent represents a file system
// symbolic link.  Note that on some operating systems, more than one file mode
// bit may be set for a node.  For instance, on Windows, a symbolic link that
// points to a directory will have both the directory and the symbolic link bits
// set.
func (de Dirent) IsSymlink() bool { return de.modeType&os.ModeSymlink != 0 }

// IsDevice returns true if and only if the Dirent represents a device file.
func (de Dirent) IsDevice() bool { return de.modeType&os.ModeDevice != 0 }

// ModeType returns the mode bits that specify the file system node type.  We
// could make our own enum-like data type for encoding the file type, but Go's
// runtime already gives us architecture independent file modes, as discussed in
// `os/types.go`:
//
//    Go's runtime FileMode type has same definition on all systems, so that
//    information about files can be moved from one system to another portably.
func (de Dirent) ModeType() os.FileMode { return de.modeType }

// Name returns the base name of the file system entry.
func (de Dirent) Name() string { return de.name }

// reset releases memory held by entry err and name, and resets mode type to 0.
func (de *Dirent) reset() {
	de.name = ""
	de.path = ""
	de.modeType = 0
}

// Dirents represents a slice of Dirent pointers, which are sortable by base
// name. This type satisfies the `sort.Interface` interface.
type Dirents []*Dirent

// Len returns the count of Dirent structures in the slice.
func (l Dirents) Len() int { return len(l) }

// Less returns true if and only if the base name of the element specified by
// the first index is lexicographically less than that of the second index.
func (l Dirents) Less(i, j int) bool { return l[i].name < l[j].name }

// Swap exchanges the two Dirent entries specified by the two provided indexes.
func (l Dirents) Swap(i, j int) { l[i], l[j] = l[j], l[i] }
