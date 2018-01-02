package git

import (
	"bytes"
	"errors"
	"io"
	"os"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/filemode"
	"gopkg.in/src-d/go-git.v4/plumbing/format/gitignore"
	"gopkg.in/src-d/go-git.v4/plumbing/format/index"
	"gopkg.in/src-d/go-git.v4/plumbing/object"
	"gopkg.in/src-d/go-git.v4/utils/ioutil"
	"gopkg.in/src-d/go-git.v4/utils/merkletrie"
	"gopkg.in/src-d/go-git.v4/utils/merkletrie/filesystem"
	mindex "gopkg.in/src-d/go-git.v4/utils/merkletrie/index"
	"gopkg.in/src-d/go-git.v4/utils/merkletrie/noder"
)

// ErrDestinationExists in an Move operation means that the target exists on
// the worktree.
var ErrDestinationExists = errors.New("destination exists")

// Status returns the working tree status.
func (w *Worktree) Status() (Status, error) {
	var hash plumbing.Hash

	ref, err := w.r.Head()
	if err != nil && err != plumbing.ErrReferenceNotFound {
		return nil, err
	}

	if err == nil {
		hash = ref.Hash()
	}

	return w.status(hash)
}

func (w *Worktree) status(commit plumbing.Hash) (Status, error) {
	s := make(Status, 0)

	left, err := w.diffCommitWithStaging(commit, false)
	if err != nil {
		return nil, err
	}

	for _, ch := range left {
		a, err := ch.Action()
		if err != nil {
			return nil, err
		}

		fs := s.File(nameFromAction(&ch))
		fs.Worktree = Unmodified

		switch a {
		case merkletrie.Delete:
			s.File(ch.From.String()).Staging = Deleted
		case merkletrie.Insert:
			s.File(ch.To.String()).Staging = Added
		case merkletrie.Modify:
			s.File(ch.To.String()).Staging = Modified
		}
	}

	right, err := w.diffStagingWithWorktree(false)
	if err != nil {
		return nil, err
	}

	for _, ch := range right {
		a, err := ch.Action()
		if err != nil {
			return nil, err
		}

		fs := s.File(nameFromAction(&ch))
		if fs.Staging == Untracked {
			fs.Staging = Unmodified
		}

		switch a {
		case merkletrie.Delete:
			fs.Worktree = Deleted
		case merkletrie.Insert:
			fs.Worktree = Untracked
			fs.Staging = Untracked
		case merkletrie.Modify:
			fs.Worktree = Modified
		}
	}

	return s, nil
}

func nameFromAction(ch *merkletrie.Change) string {
	name := ch.To.String()
	if name == "" {
		return ch.From.String()
	}

	return name
}

func (w *Worktree) diffStagingWithWorktree(reverse bool) (merkletrie.Changes, error) {
	idx, err := w.r.Storer.Index()
	if err != nil {
		return nil, err
	}

	from := mindex.NewRootNode(idx)
	submodules, err := w.getSubmodulesStatus()
	if err != nil {
		return nil, err
	}

	to := filesystem.NewRootNode(w.Filesystem, submodules)

	var c merkletrie.Changes
	if reverse {
		c, err = merkletrie.DiffTree(to, from, diffTreeIsEquals)
	} else {
		c, err = merkletrie.DiffTree(from, to, diffTreeIsEquals)
	}

	if err != nil {
		return nil, err
	}

	return w.excludeIgnoredChanges(c), nil
}

func (w *Worktree) excludeIgnoredChanges(changes merkletrie.Changes) merkletrie.Changes {
	patterns, err := gitignore.ReadPatterns(w.Filesystem, nil)
	if err != nil || len(patterns) == 0 {
		return changes
	}
	m := gitignore.NewMatcher(patterns)

	var res merkletrie.Changes
	for _, ch := range changes {
		var path []string
		for _, n := range ch.To {
			path = append(path, n.Name())
		}
		if len(path) == 0 {
			for _, n := range ch.From {
				path = append(path, n.Name())
			}
		}
		if len(path) != 0 {
			isDir := (len(ch.To) > 0 && ch.To.IsDir()) || (len(ch.From) > 0 && ch.From.IsDir())
			if m.Match(path, isDir) {
				continue
			}
		}
		res = append(res, ch)
	}
	return res
}

func (w *Worktree) getSubmodulesStatus() (map[string]plumbing.Hash, error) {
	o := map[string]plumbing.Hash{}

	sub, err := w.Submodules()
	if err != nil {
		return nil, err
	}

	status, err := sub.Status()
	if err != nil {
		return nil, err
	}

	for _, s := range status {
		if s.Current.IsZero() {
			o[s.Path] = s.Expected
			continue
		}

		o[s.Path] = s.Current
	}

	return o, nil
}

func (w *Worktree) diffCommitWithStaging(commit plumbing.Hash, reverse bool) (merkletrie.Changes, error) {
	var t *object.Tree
	if !commit.IsZero() {
		c, err := w.r.CommitObject(commit)
		if err != nil {
			return nil, err
		}

		t, err = c.Tree()
		if err != nil {
			return nil, err
		}
	}

	return w.diffTreeWithStaging(t, reverse)
}

func (w *Worktree) diffTreeWithStaging(t *object.Tree, reverse bool) (merkletrie.Changes, error) {
	var from noder.Noder
	if t != nil {
		from = object.NewTreeRootNode(t)
	}

	idx, err := w.r.Storer.Index()
	if err != nil {
		return nil, err
	}

	to := mindex.NewRootNode(idx)

	if reverse {
		return merkletrie.DiffTree(to, from, diffTreeIsEquals)
	}

	return merkletrie.DiffTree(from, to, diffTreeIsEquals)
}

var emptyNoderHash = make([]byte, 24)

// diffTreeIsEquals is a implementation of noder.Equals, used to compare
// noder.Noder, it compare the content and the length of the hashes.
//
// Since some of the noder.Noder implementations doesn't compute a hash for
// some directories, if any of the hashes is a 24-byte slice of zero values
// the comparison is not done and the hashes are take as different.
func diffTreeIsEquals(a, b noder.Hasher) bool {
	hashA := a.Hash()
	hashB := b.Hash()

	if bytes.Equal(hashA, emptyNoderHash) || bytes.Equal(hashB, emptyNoderHash) {
		return false
	}

	return bytes.Equal(hashA, hashB)
}

// Add adds the file contents of a file in the worktree to the index. if the
// file is already stagged in the index no error is returned.
func (w *Worktree) Add(path string) (plumbing.Hash, error) {
	s, err := w.Status()
	if err != nil {
		return plumbing.ZeroHash, err
	}

	h, err := w.copyFileToStorage(path)
	if err != nil {
		return h, err
	}

	if s.File(path).Worktree == Unmodified {
		return h, nil
	}

	if err := w.addOrUpdateFileToIndex(path, h); err != nil {
		return h, err
	}

	return h, err
}

func (w *Worktree) copyFileToStorage(path string) (hash plumbing.Hash, err error) {
	fi, err := w.Filesystem.Lstat(path)
	if err != nil {
		return plumbing.ZeroHash, err
	}

	obj := w.r.Storer.NewEncodedObject()
	obj.SetType(plumbing.BlobObject)
	obj.SetSize(fi.Size())

	writer, err := obj.Writer()
	if err != nil {
		return plumbing.ZeroHash, err
	}

	defer ioutil.CheckClose(writer, &err)

	if fi.Mode()&os.ModeSymlink != 0 {
		err = w.fillEncodedObjectFromSymlink(writer, path, fi)
	} else {
		err = w.fillEncodedObjectFromFile(writer, path, fi)
	}

	if err != nil {
		return plumbing.ZeroHash, err
	}

	return w.r.Storer.SetEncodedObject(obj)
}

func (w *Worktree) fillEncodedObjectFromFile(dst io.Writer, path string, fi os.FileInfo) (err error) {
	src, err := w.Filesystem.Open(path)
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(src, &err)

	if _, err := io.Copy(dst, src); err != nil {
		return err
	}

	return err
}

func (w *Worktree) fillEncodedObjectFromSymlink(dst io.Writer, path string, fi os.FileInfo) error {
	target, err := w.Filesystem.Readlink(path)
	if err != nil {
		return err
	}

	_, err = dst.Write([]byte(target))
	return err
}

func (w *Worktree) addOrUpdateFileToIndex(filename string, h plumbing.Hash) error {
	idx, err := w.r.Storer.Index()
	if err != nil {
		return err
	}

	e, err := idx.Entry(filename)
	if err != nil && err != index.ErrEntryNotFound {
		return err
	}

	if err == index.ErrEntryNotFound {
		if err := w.doAddFileToIndex(idx, filename, h); err != nil {
			return err
		}
	} else {
		if err := w.doUpdateFileToIndex(e, filename, h); err != nil {
			return err
		}
	}

	return w.r.Storer.SetIndex(idx)
}

func (w *Worktree) doAddFileToIndex(idx *index.Index, filename string, h plumbing.Hash) error {
	e := &index.Entry{Name: filename}
	idx.Entries = append(idx.Entries, e)

	return w.doUpdateFileToIndex(e, filename, h)
}

func (w *Worktree) doUpdateFileToIndex(e *index.Entry, filename string, h plumbing.Hash) error {
	info, err := w.Filesystem.Lstat(filename)
	if err != nil {
		return err
	}

	e.Hash = h
	e.ModifiedAt = info.ModTime()
	e.Mode, err = filemode.NewFromOSFileMode(info.Mode())
	if err != nil {
		return err
	}

	if e.Mode.IsRegular() {
		e.Size = uint32(info.Size())
	}

	fillSystemInfo(e, info.Sys())
	return nil
}

// Remove removes files from the working tree and from the index.
func (w *Worktree) Remove(path string) (plumbing.Hash, error) {
	hash, err := w.deleteFromIndex(path)
	if err != nil {
		return plumbing.ZeroHash, err
	}

	return hash, w.deleteFromFilesystem(path)
}

func (w *Worktree) deleteFromIndex(path string) (plumbing.Hash, error) {
	idx, err := w.r.Storer.Index()
	if err != nil {
		return plumbing.ZeroHash, err
	}

	e, err := idx.Remove(path)
	if err != nil {
		return plumbing.ZeroHash, err
	}

	return e.Hash, w.r.Storer.SetIndex(idx)
}

func (w *Worktree) deleteFromFilesystem(path string) error {
	err := w.Filesystem.Remove(path)
	if os.IsNotExist(err) {
		return nil
	}

	return err
}

// Move moves or rename a file in the worktree and the index, directories are
// not supported.
func (w *Worktree) Move(from, to string) (plumbing.Hash, error) {
	if _, err := w.Filesystem.Lstat(from); err != nil {
		return plumbing.ZeroHash, err
	}

	if _, err := w.Filesystem.Lstat(to); err == nil {
		return plumbing.ZeroHash, ErrDestinationExists
	}

	hash, err := w.deleteFromIndex(from)
	if err != nil {
		return plumbing.ZeroHash, err
	}

	if err := w.Filesystem.Rename(from, to); err != nil {
		return hash, err
	}

	return hash, w.addOrUpdateFileToIndex(to, hash)
}
