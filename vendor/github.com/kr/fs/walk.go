// Package fs provides filesystem-related functions.
package fs

import (
	"os"
)

// Walker provides a convenient interface for iterating over the
// descendants of a filesystem path.
// Successive calls to the Step method will step through each
// file or directory in the tree, including the root. The files
// are walked in lexical order, which makes the output deterministic
// but means that for very large directories Walker can be inefficient.
// Walker does not follow symbolic links.
type Walker struct {
	fs      FileSystem
	cur     item
	stack   []item
	descend bool
}

type item struct {
	path string
	info os.FileInfo
	err  error
}

// Walk returns a new Walker rooted at root.
func Walk(root string) *Walker {
	return WalkFS(root, new(fs))
}

// WalkFS returns a new Walker rooted at root on the FileSystem fs.
func WalkFS(root string, fs FileSystem) *Walker {
	info, err := fs.Lstat(root)
	return &Walker{
		fs:    fs,
		stack: []item{{root, info, err}},
	}
}

// Step advances the Walker to the next file or directory,
// which will then be available through the Path, Stat,
// and Err methods.
// It returns false when the walk stops at the end of the tree.
func (w *Walker) Step() bool {
	if w.descend && w.cur.err == nil && w.cur.info.IsDir() {
		list, err := w.fs.ReadDir(w.cur.path)
		if err != nil {
			w.cur.err = err
			w.stack = append(w.stack, w.cur)
		} else {
			for i := len(list) - 1; i >= 0; i-- {
				path := w.fs.Join(w.cur.path, list[i].Name())
				w.stack = append(w.stack, item{path, list[i], nil})
			}
		}
	}

	if len(w.stack) == 0 {
		return false
	}
	i := len(w.stack) - 1
	w.cur = w.stack[i]
	w.stack = w.stack[:i]
	w.descend = true
	return true
}

// Path returns the path to the most recent file or directory
// visited by a call to Step. It contains the argument to Walk
// as a prefix; that is, if Walk is called with "dir", which is
// a directory containing the file "a", Path will return "dir/a".
func (w *Walker) Path() string {
	return w.cur.path
}

// Stat returns info for the most recent file or directory
// visited by a call to Step.
func (w *Walker) Stat() os.FileInfo {
	return w.cur.info
}

// Err returns the error, if any, for the most recent attempt
// by Step to visit a file or directory. If a directory has
// an error, w will not descend into that directory.
func (w *Walker) Err() error {
	return w.cur.err
}

// SkipDir causes the currently visited directory to be skipped.
// If w is not on a directory, SkipDir has no effect.
func (w *Walker) SkipDir() {
	w.descend = false
}
