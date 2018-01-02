package git

import (
	"path"
	"strings"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/filemode"
	"gopkg.in/src-d/go-git.v4/plumbing/format/index"
	"gopkg.in/src-d/go-git.v4/plumbing/object"
	"gopkg.in/src-d/go-git.v4/storage"

	"gopkg.in/src-d/go-billy.v3"
)

// Commit stores the current contents of the index in a new commit along with
// a log message from the user describing the changes.
func (w *Worktree) Commit(msg string, opts *CommitOptions) (plumbing.Hash, error) {
	if err := opts.Validate(w.r); err != nil {
		return plumbing.ZeroHash, err
	}

	if opts.All {
		if err := w.autoAddModifiedAndDeleted(); err != nil {
			return plumbing.ZeroHash, err
		}
	}

	idx, err := w.r.Storer.Index()
	if err != nil {
		return plumbing.ZeroHash, err
	}

	h := &buildTreeHelper{
		fs: w.Filesystem,
		s:  w.r.Storer,
	}

	tree, err := h.BuildTree(idx)
	if err != nil {
		return plumbing.ZeroHash, err
	}

	commit, err := w.buildCommitObject(msg, opts, tree)
	if err != nil {
		return plumbing.ZeroHash, err
	}

	return commit, w.updateHEAD(commit)
}

func (w *Worktree) autoAddModifiedAndDeleted() error {
	s, err := w.Status()
	if err != nil {
		return err
	}

	for path, fs := range s {
		if fs.Worktree != Modified && fs.Worktree != Deleted {
			continue
		}

		if _, err := w.Add(path); err != nil {
			return err
		}

	}

	return nil
}

func (w *Worktree) updateHEAD(commit plumbing.Hash) error {
	head, err := w.r.Storer.Reference(plumbing.HEAD)
	if err != nil {
		return err
	}

	name := plumbing.HEAD
	if head.Type() != plumbing.HashReference {
		name = head.Target()
	}

	ref := plumbing.NewHashReference(name, commit)
	return w.r.Storer.SetReference(ref)
}

func (w *Worktree) buildCommitObject(msg string, opts *CommitOptions, tree plumbing.Hash) (plumbing.Hash, error) {
	commit := &object.Commit{
		Author:       *opts.Author,
		Committer:    *opts.Committer,
		Message:      msg,
		TreeHash:     tree,
		ParentHashes: opts.Parents,
	}

	obj := w.r.Storer.NewEncodedObject()
	if err := commit.Encode(obj); err != nil {
		return plumbing.ZeroHash, err
	}
	return w.r.Storer.SetEncodedObject(obj)
}

// buildTreeHelper converts a given index.Index file into multiple git objects
// reading the blobs from the given filesystem and creating the trees from the
// index structure. The created objects are pushed to a given Storer.
type buildTreeHelper struct {
	fs billy.Filesystem
	s  storage.Storer

	trees   map[string]*object.Tree
	entries map[string]*object.TreeEntry
}

// BuildTree builds the tree objects and push its to the storer, the hash
// of the root tree is returned.
func (h *buildTreeHelper) BuildTree(idx *index.Index) (plumbing.Hash, error) {
	const rootNode = ""
	h.trees = map[string]*object.Tree{rootNode: {}}
	h.entries = map[string]*object.TreeEntry{}

	for _, e := range idx.Entries {
		if err := h.commitIndexEntry(e); err != nil {
			return plumbing.ZeroHash, err
		}
	}

	return h.copyTreeToStorageRecursive(rootNode, h.trees[rootNode])
}

func (h *buildTreeHelper) commitIndexEntry(e *index.Entry) error {
	parts := strings.Split(e.Name, "/")

	var fullpath string
	for _, part := range parts {
		parent := fullpath
		fullpath = path.Join(fullpath, part)

		h.doBuildTree(e, parent, fullpath)
	}

	return nil
}

func (h *buildTreeHelper) doBuildTree(e *index.Entry, parent, fullpath string) {
	if _, ok := h.trees[fullpath]; ok {
		return
	}

	if _, ok := h.entries[fullpath]; ok {
		return
	}

	te := object.TreeEntry{Name: path.Base(fullpath)}

	if fullpath == e.Name {
		te.Mode = e.Mode
		te.Hash = e.Hash
	} else {
		te.Mode = filemode.Dir
		h.trees[fullpath] = &object.Tree{}
	}

	h.trees[parent].Entries = append(h.trees[parent].Entries, te)
}

func (h *buildTreeHelper) copyTreeToStorageRecursive(parent string, t *object.Tree) (plumbing.Hash, error) {
	for i, e := range t.Entries {
		if e.Mode != filemode.Dir && !e.Hash.IsZero() {
			continue
		}

		path := path.Join(parent, e.Name)

		var err error
		e.Hash, err = h.copyTreeToStorageRecursive(path, h.trees[path])
		if err != nil {
			return plumbing.ZeroHash, err
		}

		t.Entries[i] = e
	}

	o := h.s.NewEncodedObject()
	if err := t.Encode(o); err != nil {
		return plumbing.ZeroHash, err
	}

	return h.s.SetEncodedObject(o)
}
