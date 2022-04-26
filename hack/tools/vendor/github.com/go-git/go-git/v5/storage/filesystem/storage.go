// Package filesystem is a storage backend base on filesystems
package filesystem

import (
	"github.com/go-git/go-git/v5/plumbing/cache"
	"github.com/go-git/go-git/v5/storage/filesystem/dotgit"

	"github.com/go-git/go-billy/v5"
)

// Storage is an implementation of git.Storer that stores data on disk in the
// standard git format (this is, the .git directory). Zero values of this type
// are not safe to use, see the NewStorage function below.
type Storage struct {
	fs  billy.Filesystem
	dir *dotgit.DotGit

	ObjectStorage
	ReferenceStorage
	IndexStorage
	ShallowStorage
	ConfigStorage
	ModuleStorage
}

// Options holds configuration for the storage.
type Options struct {
	// ExclusiveAccess means that the filesystem is not modified externally
	// while the repo is open.
	ExclusiveAccess bool
	// KeepDescriptors makes the file descriptors to be reused but they will
	// need to be manually closed calling Close().
	KeepDescriptors bool
	// MaxOpenDescriptors is the max number of file descriptors to keep
	// open. If KeepDescriptors is true, all file descriptors will remain open.
	MaxOpenDescriptors int
}

// NewStorage returns a new Storage backed by a given `fs.Filesystem` and cache.
func NewStorage(fs billy.Filesystem, cache cache.Object) *Storage {
	return NewStorageWithOptions(fs, cache, Options{})
}

// NewStorageWithOptions returns a new Storage with extra options,
// backed by a given `fs.Filesystem` and cache.
func NewStorageWithOptions(fs billy.Filesystem, cache cache.Object, ops Options) *Storage {
	dirOps := dotgit.Options{
		ExclusiveAccess: ops.ExclusiveAccess,
	}
	dir := dotgit.NewWithOptions(fs, dirOps)

	return &Storage{
		fs:  fs,
		dir: dir,

		ObjectStorage:    *NewObjectStorageWithOptions(dir, cache, ops),
		ReferenceStorage: ReferenceStorage{dir: dir},
		IndexStorage:     IndexStorage{dir: dir},
		ShallowStorage:   ShallowStorage{dir: dir},
		ConfigStorage:    ConfigStorage{dir: dir},
		ModuleStorage:    ModuleStorage{dir: dir},
	}
}

// Filesystem returns the underlying filesystem
func (s *Storage) Filesystem() billy.Filesystem {
	return s.fs
}

// Init initializes .git directory
func (s *Storage) Init() error {
	return s.dir.Initialize()
}
