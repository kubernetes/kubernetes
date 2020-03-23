package godirwalk

import (
	"os"
	"path/filepath"

	"github.com/pkg/errors"
)

// Dirent stores the name and file system mode type of discovered file system
// entries.
type Dirent struct {
	name     string
	modeType os.FileMode
}

// NewDirent returns a newly initialized Dirent structure, or an error. This
// function does not follow symbolic links.
//
// This function is rarely used, as Dirent structures are provided by other
// functions in this library that read and walk directories.
func NewDirent(osPathname string) (*Dirent, error) {
	fi, err := os.Lstat(osPathname)
	if err != nil {
		return nil, errors.Wrap(err, "cannot lstat")
	}
	return &Dirent{
		name:     filepath.Base(osPathname),
		modeType: fi.Mode() & os.ModeType,
	}, nil
}

// Name returns the basename of the file system entry.
func (de Dirent) Name() string { return de.name }

// ModeType returns the mode bits that specify the file system node type. We
// could make our own enum-like data type for encoding the file type, but Go's
// runtime already gives us architecture independent file modes, as discussed in
// `os/types.go`:
//
//    Go's runtime FileMode type has same definition on all systems, so that
//    information about files can be moved from one system to another portably.
func (de Dirent) ModeType() os.FileMode { return de.modeType }

// IsDir returns true if and only if the Dirent represents a file system
// directory. Note that on some operating systems, more than one file mode bit
// may be set for a node. For instance, on Windows, a symbolic link that points
// to a directory will have both the directory and the symbolic link bits set.
func (de Dirent) IsDir() bool { return de.modeType&os.ModeDir != 0 }

// IsRegular returns true if and only if the Dirent represents a regular
// file. That is, it ensures that no mode type bits are set.
func (de Dirent) IsRegular() bool { return de.modeType&os.ModeType == 0 }

// IsSymlink returns true if and only if the Dirent represents a file system
// symbolic link. Note that on some operating systems, more than one file mode
// bit may be set for a node. For instance, on Windows, a symbolic link that
// points to a directory will have both the directory and the symbolic link bits
// set.
func (de Dirent) IsSymlink() bool { return de.modeType&os.ModeSymlink != 0 }

// Dirents represents a slice of Dirent pointers, which are sortable by
// name. This type satisfies the `sort.Interface` interface.
type Dirents []*Dirent

// Len returns the count of Dirent structures in the slice.
func (l Dirents) Len() int { return len(l) }

// Less returns true if and only if the Name of the element specified by the
// first index is lexicographically less than that of the second index.
func (l Dirents) Less(i, j int) bool { return l[i].name < l[j].name }

// Swap exchanges the two Dirent entries specified by the two provided indexes.
func (l Dirents) Swap(i, j int) { l[i], l[j] = l[j], l[i] }
