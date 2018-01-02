package fs

import (
	"context"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sync/errgroup"

	"github.com/sirupsen/logrus"
)

// ChangeKind is the type of modification that
// a change is making.
type ChangeKind int

const (
	// ChangeKindUnmodified represents an unmodified
	// file
	ChangeKindUnmodified = iota

	// ChangeKindAdd represents an addition of
	// a file
	ChangeKindAdd

	// ChangeKindModify represents a change to
	// an existing file
	ChangeKindModify

	// ChangeKindDelete represents a delete of
	// a file
	ChangeKindDelete
)

func (k ChangeKind) String() string {
	switch k {
	case ChangeKindUnmodified:
		return "unmodified"
	case ChangeKindAdd:
		return "add"
	case ChangeKindModify:
		return "modify"
	case ChangeKindDelete:
		return "delete"
	default:
		return ""
	}
}

// Change represents single change between a diff and its parent.
type Change struct {
	Kind ChangeKind
	Path string
}

// ChangeFunc is the type of function called for each change
// computed during a directory changes calculation.
type ChangeFunc func(ChangeKind, string, os.FileInfo, error) error

// Changes computes changes between two directories calling the
// given change function for each computed change. The first
// directory is intended to the base directory and second
// directory the changed directory.
//
// The change callback is called by the order of path names and
// should be appliable in that order.
//  Due to this apply ordering, the following is true
//  - Removed directory trees only create a single change for the root
//    directory removed. Remaining changes are implied.
//  - A directory which is modified to become a file will not have
//    delete entries for sub-path items, their removal is implied
//    by the removal of the parent directory.
//
// Opaque directories will not be treated specially and each file
// removed from the base directory will show up as a removal.
//
// File content comparisons will be done on files which have timestamps
// which may have been truncated. If either of the files being compared
// has a zero value nanosecond value, each byte will be compared for
// differences. If 2 files have the same seconds value but different
// nanosecond values where one of those values is zero, the files will
// be considered unchanged if the content is the same. This behavior
// is to account for timestamp truncation during archiving.
func Changes(ctx context.Context, a, b string, changeFn ChangeFunc) error {
	if a == "" {
		logrus.Debugf("Using single walk diff for %s", b)
		return addDirChanges(ctx, changeFn, b)
	} else if diffOptions := detectDirDiff(b, a); diffOptions != nil {
		logrus.Debugf("Using single walk diff for %s from %s", diffOptions.diffDir, a)
		return diffDirChanges(ctx, changeFn, a, diffOptions)
	}

	logrus.Debugf("Using double walk diff for %s from %s", b, a)
	return doubleWalkDiff(ctx, changeFn, a, b)
}

func addDirChanges(ctx context.Context, changeFn ChangeFunc, root string) error {
	return filepath.Walk(root, func(path string, f os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Rebase path
		path, err = filepath.Rel(root, path)
		if err != nil {
			return err
		}

		path = filepath.Join(string(os.PathSeparator), path)

		// Skip root
		if path == string(os.PathSeparator) {
			return nil
		}

		return changeFn(ChangeKindAdd, path, f, nil)
	})
}

// diffDirOptions is used when the diff can be directly calculated from
// a diff directory to its base, without walking both trees.
type diffDirOptions struct {
	diffDir      string
	skipChange   func(string) (bool, error)
	deleteChange func(string, string, os.FileInfo) (string, error)
}

// diffDirChanges walks the diff directory and compares changes against the base.
func diffDirChanges(ctx context.Context, changeFn ChangeFunc, base string, o *diffDirOptions) error {
	changedDirs := make(map[string]struct{})
	return filepath.Walk(o.diffDir, func(path string, f os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Rebase path
		path, err = filepath.Rel(o.diffDir, path)
		if err != nil {
			return err
		}

		path = filepath.Join(string(os.PathSeparator), path)

		// Skip root
		if path == string(os.PathSeparator) {
			return nil
		}

		// TODO: handle opaqueness, start new double walker at this
		// location to get deletes, and skip tree in single walker

		if o.skipChange != nil {
			if skip, err := o.skipChange(path); skip {
				return err
			}
		}

		var kind ChangeKind

		deletedFile, err := o.deleteChange(o.diffDir, path, f)
		if err != nil {
			return err
		}

		// Find out what kind of modification happened
		if deletedFile != "" {
			path = deletedFile
			kind = ChangeKindDelete
			f = nil
		} else {
			// Otherwise, the file was added
			kind = ChangeKindAdd

			// ...Unless it already existed in a base, in which case, it's a modification
			stat, err := os.Stat(filepath.Join(base, path))
			if err != nil && !os.IsNotExist(err) {
				return err
			}
			if err == nil {
				// The file existed in the base, so that's a modification

				// However, if it's a directory, maybe it wasn't actually modified.
				// If you modify /foo/bar/baz, then /foo will be part of the changed files only because it's the parent of bar
				if stat.IsDir() && f.IsDir() {
					if f.Size() == stat.Size() && f.Mode() == stat.Mode() && sameFsTime(f.ModTime(), stat.ModTime()) {
						// Both directories are the same, don't record the change
						return nil
					}
				}
				kind = ChangeKindModify
			}
		}

		// If /foo/bar/file.txt is modified, then /foo/bar must be part of the changed files.
		// This block is here to ensure the change is recorded even if the
		// modify time, mode and size of the parent directory in the rw and ro layers are all equal.
		// Check https://github.com/docker/docker/pull/13590 for details.
		if f.IsDir() {
			changedDirs[path] = struct{}{}
		}
		if kind == ChangeKindAdd || kind == ChangeKindDelete {
			parent := filepath.Dir(path)
			if _, ok := changedDirs[parent]; !ok && parent != "/" {
				pi, err := os.Stat(filepath.Join(o.diffDir, parent))
				if err := changeFn(ChangeKindModify, parent, pi, err); err != nil {
					return err
				}
				changedDirs[parent] = struct{}{}
			}
		}

		return changeFn(kind, path, f, nil)
	})
}

// doubleWalkDiff walks both directories to create a diff
func doubleWalkDiff(ctx context.Context, changeFn ChangeFunc, a, b string) (err error) {
	g, ctx := errgroup.WithContext(ctx)

	var (
		c1 = make(chan *currentPath)
		c2 = make(chan *currentPath)

		f1, f2 *currentPath
		rmdir  string
	)
	g.Go(func() error {
		defer close(c1)
		return pathWalk(ctx, a, c1)
	})
	g.Go(func() error {
		defer close(c2)
		return pathWalk(ctx, b, c2)
	})
	g.Go(func() error {
		for c1 != nil || c2 != nil {
			if f1 == nil && c1 != nil {
				f1, err = nextPath(ctx, c1)
				if err != nil {
					return err
				}
				if f1 == nil {
					c1 = nil
				}
			}

			if f2 == nil && c2 != nil {
				f2, err = nextPath(ctx, c2)
				if err != nil {
					return err
				}
				if f2 == nil {
					c2 = nil
				}
			}
			if f1 == nil && f2 == nil {
				continue
			}

			var f os.FileInfo
			k, p := pathChange(f1, f2)
			switch k {
			case ChangeKindAdd:
				if rmdir != "" {
					rmdir = ""
				}
				f = f2.f
				f2 = nil
			case ChangeKindDelete:
				// Check if this file is already removed by being
				// under of a removed directory
				if rmdir != "" && strings.HasPrefix(f1.path, rmdir) {
					f1 = nil
					continue
				} else if rmdir == "" && f1.f.IsDir() {
					rmdir = f1.path + string(os.PathSeparator)
				} else if rmdir != "" {
					rmdir = ""
				}
				f1 = nil
			case ChangeKindModify:
				same, err := sameFile(f1, f2)
				if err != nil {
					return err
				}
				if f1.f.IsDir() && !f2.f.IsDir() {
					rmdir = f1.path + string(os.PathSeparator)
				} else if rmdir != "" {
					rmdir = ""
				}
				f = f2.f
				f1 = nil
				f2 = nil
				if same {
					if !isLinked(f) {
						continue
					}
					k = ChangeKindUnmodified
				}
			}
			if err := changeFn(k, p, f, nil); err != nil {
				return err
			}
		}
		return nil
	})

	return g.Wait()
}
