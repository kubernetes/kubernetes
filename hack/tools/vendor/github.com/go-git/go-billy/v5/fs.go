package billy

import (
	"errors"
	"io"
	"os"
	"time"
)

var (
	ErrReadOnly        = errors.New("read-only filesystem")
	ErrNotSupported    = errors.New("feature not supported")
	ErrCrossedBoundary = errors.New("chroot boundary crossed")
)

// Capability holds the supported features of a billy filesystem. This does
// not mean that the capability has to be supported by the underlying storage.
// For example, a billy filesystem may support WriteCapability but the
// storage be mounted in read only mode.
type Capability uint64

const (
	// WriteCapability means that the fs is writable.
	WriteCapability Capability = 1 << iota
	// ReadCapability means that the fs is readable.
	ReadCapability
	// ReadAndWriteCapability is the ability to open a file in read and write mode.
	ReadAndWriteCapability
	// SeekCapability means it is able to move position inside the file.
	SeekCapability
	// TruncateCapability means that a file can be truncated.
	TruncateCapability
	// LockCapability is the ability to lock a file.
	LockCapability

	// DefaultCapabilities lists all capable features supported by filesystems
	// without Capability interface. This list should not be changed until a
	// major version is released.
	DefaultCapabilities Capability = WriteCapability | ReadCapability |
		ReadAndWriteCapability | SeekCapability | TruncateCapability |
		LockCapability

	// AllCapabilities lists all capable features.
	AllCapabilities Capability = WriteCapability | ReadCapability |
		ReadAndWriteCapability | SeekCapability | TruncateCapability |
		LockCapability
)

// Filesystem abstract the operations in a storage-agnostic interface.
// Each method implementation mimics the behavior of the equivalent functions
// at the os package from the standard library.
type Filesystem interface {
	Basic
	TempFile
	Dir
	Symlink
	Chroot
}

// Basic abstract the basic operations in a storage-agnostic interface as
// an extension to the Basic interface.
type Basic interface {
	// Create creates the named file with mode 0666 (before umask), truncating
	// it if it already exists. If successful, methods on the returned File can
	// be used for I/O; the associated file descriptor has mode O_RDWR.
	Create(filename string) (File, error)
	// Open opens the named file for reading. If successful, methods on the
	// returned file can be used for reading; the associated file descriptor has
	// mode O_RDONLY.
	Open(filename string) (File, error)
	// OpenFile is the generalized open call; most users will use Open or Create
	// instead. It opens the named file with specified flag (O_RDONLY etc.) and
	// perm, (0666 etc.) if applicable. If successful, methods on the returned
	// File can be used for I/O.
	OpenFile(filename string, flag int, perm os.FileMode) (File, error)
	// Stat returns a FileInfo describing the named file.
	Stat(filename string) (os.FileInfo, error)
	// Rename renames (moves) oldpath to newpath. If newpath already exists and
	// is not a directory, Rename replaces it. OS-specific restrictions may
	// apply when oldpath and newpath are in different directories.
	Rename(oldpath, newpath string) error
	// Remove removes the named file or directory.
	Remove(filename string) error
	// Join joins any number of path elements into a single path, adding a
	// Separator if necessary. Join calls filepath.Clean on the result; in
	// particular, all empty strings are ignored. On Windows, the result is a
	// UNC path if and only if the first path element is a UNC path.
	Join(elem ...string) string
}

type TempFile interface {
	// TempFile creates a new temporary file in the directory dir with a name
	// beginning with prefix, opens the file for reading and writing, and
	// returns the resulting *os.File. If dir is the empty string, TempFile
	// uses the default directory for temporary files (see os.TempDir).
	// Multiple programs calling TempFile simultaneously will not choose the
	// same file. The caller can use f.Name() to find the pathname of the file.
	// It is the caller's responsibility to remove the file when no longer
	// needed.
	TempFile(dir, prefix string) (File, error)
}

// Dir abstract the dir related operations in a storage-agnostic interface as
// an extension to the Basic interface.
type Dir interface {
	// ReadDir reads the directory named by dirname and returns a list of
	// directory entries sorted by filename.
	ReadDir(path string) ([]os.FileInfo, error)
	// MkdirAll creates a directory named path, along with any necessary
	// parents, and returns nil, or else returns an error. The permission bits
	// perm are used for all directories that MkdirAll creates. If path is/
	// already a directory, MkdirAll does nothing and returns nil.
	MkdirAll(filename string, perm os.FileMode) error
}

// Symlink abstract the symlink related operations in a storage-agnostic
// interface as an extension to the Basic interface.
type Symlink interface {
	// Lstat returns a FileInfo describing the named file. If the file is a
	// symbolic link, the returned FileInfo describes the symbolic link. Lstat
	// makes no attempt to follow the link.
	Lstat(filename string) (os.FileInfo, error)
	// Symlink creates a symbolic-link from link to target. target may be an
	// absolute or relative path, and need not refer to an existing node.
	// Parent directories of link are created as necessary.
	Symlink(target, link string) error
	// Readlink returns the target path of link.
	Readlink(link string) (string, error)
}

// Change abstract the FileInfo change related operations in a storage-agnostic
// interface as an extension to the Basic interface
type Change interface {
	// Chmod changes the mode of the named file to mode. If the file is a
	// symbolic link, it changes the mode of the link's target.
	Chmod(name string, mode os.FileMode) error
	// Lchown changes the numeric uid and gid of the named file. If the file is
	// a symbolic link, it changes the uid and gid of the link itself.
	Lchown(name string, uid, gid int) error
	// Chown changes the numeric uid and gid of the named file. If the file is a
	// symbolic link, it changes the uid and gid of the link's target.
	Chown(name string, uid, gid int) error
	// Chtimes changes the access and modification times of the named file,
	// similar to the Unix utime() or utimes() functions.
	//
	// The underlying filesystem may truncate or round the values to a less
	// precise time unit.
	Chtimes(name string, atime time.Time, mtime time.Time) error
}

// Chroot abstract the chroot related operations in a storage-agnostic interface
// as an extension to the Basic interface.
type Chroot interface {
	// Chroot returns a new filesystem from the same type where the new root is
	// the given path. Files outside of the designated directory tree cannot be
	// accessed.
	Chroot(path string) (Filesystem, error)
	// Root returns the root path of the filesystem.
	Root() string
}

// File represent a file, being a subset of the os.File
type File interface {
	// Name returns the name of the file as presented to Open.
	Name() string
	io.Writer
	io.Reader
	io.ReaderAt
	io.Seeker
	io.Closer
	// Lock locks the file like e.g. flock. It protects against access from
	// other processes.
	Lock() error
	// Unlock unlocks the file.
	Unlock() error
	// Truncate the file.
	Truncate(size int64) error
}

// Capable interface can return the available features of a filesystem.
type Capable interface {
	// Capabilities returns the capabilities of a filesystem in bit flags.
	Capabilities() Capability
}

// Capabilities returns the features supported by a filesystem. If the FS
// does not implement Capable interface it returns all features.
func Capabilities(fs Basic) Capability {
	capable, ok := fs.(Capable)
	if !ok {
		return DefaultCapabilities
	}

	return capable.Capabilities()
}

// CapabilityCheck tests the filesystem for the provided capabilities and
// returns true in case it supports all of them.
func CapabilityCheck(fs Basic, capabilities Capability) bool {
	fsCaps := Capabilities(fs)
	return fsCaps&capabilities == capabilities
}
