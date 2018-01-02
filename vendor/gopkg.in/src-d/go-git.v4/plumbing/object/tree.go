package object

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"path"
	"strings"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/filemode"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/utils/ioutil"
)

const (
	maxTreeDepth      = 1024
	startingStackSize = 8
)

// New errors defined by this package.
var (
	ErrMaxTreeDepth      = errors.New("maximum tree depth exceeded")
	ErrFileNotFound      = errors.New("file not found")
	ErrDirectoryNotFound = errors.New("directory not found")
)

// Tree is basically like a directory - it references a bunch of other trees
// and/or blobs (i.e. files and sub-directories)
type Tree struct {
	Entries []TreeEntry
	Hash    plumbing.Hash

	s storer.EncodedObjectStorer
	m map[string]*TreeEntry
}

// GetTree gets a tree from an object storer and decodes it.
func GetTree(s storer.EncodedObjectStorer, h plumbing.Hash) (*Tree, error) {
	o, err := s.EncodedObject(plumbing.TreeObject, h)
	if err != nil {
		return nil, err
	}

	return DecodeTree(s, o)
}

// DecodeTree decodes an encoded object into a *Tree and associates it to the
// given object storer.
func DecodeTree(s storer.EncodedObjectStorer, o plumbing.EncodedObject) (*Tree, error) {
	t := &Tree{s: s}
	if err := t.Decode(o); err != nil {
		return nil, err
	}

	return t, nil
}

// TreeEntry represents a file
type TreeEntry struct {
	Name string
	Mode filemode.FileMode
	Hash plumbing.Hash
}

// File returns the hash of the file identified by the `path` argument.
// The path is interpreted as relative to the tree receiver.
func (t *Tree) File(path string) (*File, error) {
	e, err := t.FindEntry(path)
	if err != nil {
		return nil, ErrFileNotFound
	}

	blob, err := GetBlob(t.s, e.Hash)
	if err != nil {
		if err == plumbing.ErrObjectNotFound {
			return nil, ErrFileNotFound
		}
		return nil, err
	}

	return NewFile(path, e.Mode, blob), nil
}

// Tree returns the tree identified by the `path` argument.
// The path is interpreted as relative to the tree receiver.
func (t *Tree) Tree(path string) (*Tree, error) {
	e, err := t.FindEntry(path)
	if err != nil {
		return nil, ErrDirectoryNotFound
	}

	tree, err := GetTree(t.s, e.Hash)
	if err == plumbing.ErrObjectNotFound {
		return nil, ErrDirectoryNotFound
	}

	return tree, err
}

// TreeEntryFile returns the *File for a given *TreeEntry.
func (t *Tree) TreeEntryFile(e *TreeEntry) (*File, error) {
	blob, err := GetBlob(t.s, e.Hash)
	if err != nil {
		return nil, err
	}

	return NewFile(e.Name, e.Mode, blob), nil
}

// FindEntry search a TreeEntry in this tree or any subtree.
func (t *Tree) FindEntry(path string) (*TreeEntry, error) {
	pathParts := strings.Split(path, "/")

	var tree *Tree
	var err error
	for tree = t; len(pathParts) > 1; pathParts = pathParts[1:] {
		if tree, err = tree.dir(pathParts[0]); err != nil {
			return nil, err
		}
	}

	return tree.entry(pathParts[0])
}

func (t *Tree) dir(baseName string) (*Tree, error) {
	entry, err := t.entry(baseName)
	if err != nil {
		return nil, ErrDirectoryNotFound
	}

	obj, err := t.s.EncodedObject(plumbing.TreeObject, entry.Hash)
	if err != nil {
		return nil, err
	}

	tree := &Tree{s: t.s}
	tree.Decode(obj)

	return tree, nil
}

var errEntryNotFound = errors.New("entry not found")

func (t *Tree) entry(baseName string) (*TreeEntry, error) {
	if t.m == nil {
		t.buildMap()
	}

	entry, ok := t.m[baseName]
	if !ok {
		return nil, errEntryNotFound
	}

	return entry, nil
}

// Files returns a FileIter allowing to iterate over the Tree
func (t *Tree) Files() *FileIter {
	return NewFileIter(t.s, t)
}

// ID returns the object ID of the tree. The returned value will always match
// the current value of Tree.Hash.
//
// ID is present to fulfill the Object interface.
func (t *Tree) ID() plumbing.Hash {
	return t.Hash
}

// Type returns the type of object. It always returns plumbing.TreeObject.
func (t *Tree) Type() plumbing.ObjectType {
	return plumbing.TreeObject
}

// Decode transform an plumbing.EncodedObject into a Tree struct
func (t *Tree) Decode(o plumbing.EncodedObject) (err error) {
	if o.Type() != plumbing.TreeObject {
		return ErrUnsupportedObject
	}

	t.Hash = o.Hash()
	if o.Size() == 0 {
		return nil
	}

	t.Entries = nil
	t.m = nil

	reader, err := o.Reader()
	if err != nil {
		return err
	}
	defer ioutil.CheckClose(reader, &err)

	r := bufio.NewReader(reader)
	for {
		str, err := r.ReadString(' ')
		if err != nil {
			if err == io.EOF {
				break
			}

			return err
		}
		str = str[:len(str)-1] // strip last byte (' ')

		mode, err := filemode.New(str)
		if err != nil {
			return err
		}

		name, err := r.ReadString(0)
		if err != nil && err != io.EOF {
			return err
		}

		var hash plumbing.Hash
		if _, err = io.ReadFull(r, hash[:]); err != nil {
			return err
		}

		baseName := name[:len(name)-1]
		t.Entries = append(t.Entries, TreeEntry{
			Hash: hash,
			Mode: mode,
			Name: baseName,
		})
	}

	return nil
}

// Encode transforms a Tree into a plumbing.EncodedObject.
func (t *Tree) Encode(o plumbing.EncodedObject) error {
	o.SetType(plumbing.TreeObject)
	w, err := o.Writer()
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(w, &err)
	for _, entry := range t.Entries {
		if _, err := fmt.Fprintf(w, "%o %s", entry.Mode, entry.Name); err != nil {
			return err
		}

		if _, err = w.Write([]byte{0x00}); err != nil {
			return err
		}

		if _, err = w.Write([]byte(entry.Hash[:])); err != nil {
			return err
		}
	}

	return err
}

func (t *Tree) buildMap() {
	t.m = make(map[string]*TreeEntry)
	for i := 0; i < len(t.Entries); i++ {
		t.m[t.Entries[i].Name] = &t.Entries[i]
	}
}

// Diff returns a list of changes between this tree and the provided one
func (from *Tree) Diff(to *Tree) (Changes, error) {
	return DiffTree(from, to)
}

// Patch returns a slice of Patch objects with all the changes between trees
// in chunks. This representation can be used to create several diff outputs.
func (from *Tree) Patch(to *Tree) (*Patch, error) {
	changes, err := DiffTree(from, to)
	if err != nil {
		return nil, err
	}

	return changes.Patch()
}

// treeEntryIter facilitates iterating through the TreeEntry objects in a Tree.
type treeEntryIter struct {
	t   *Tree
	pos int
}

func (iter *treeEntryIter) Next() (TreeEntry, error) {
	if iter.pos >= len(iter.t.Entries) {
		return TreeEntry{}, io.EOF
	}
	iter.pos++
	return iter.t.Entries[iter.pos-1], nil
}

// TreeWalker provides a means of walking through all of the entries in a Tree.
type TreeWalker struct {
	stack     []*treeEntryIter
	base      string
	recursive bool
	seen      map[plumbing.Hash]bool

	s storer.EncodedObjectStorer
	t *Tree
}

// NewTreeWalker returns a new TreeWalker for the given tree.
//
// It is the caller's responsibility to call Close() when finished with the
// tree walker.
func NewTreeWalker(t *Tree, recursive bool, seen map[plumbing.Hash]bool) *TreeWalker {
	stack := make([]*treeEntryIter, 0, startingStackSize)
	stack = append(stack, &treeEntryIter{t, 0})

	return &TreeWalker{
		stack:     stack,
		recursive: recursive,
		seen:      seen,

		s: t.s,
		t: t,
	}
}

// Next returns the next object from the tree. Objects are returned in order
// and subtrees are included. After the last object has been returned further
// calls to Next() will return io.EOF.
//
// In the current implementation any objects which cannot be found in the
// underlying repository will be skipped automatically. It is possible that this
// may change in future versions.
func (w *TreeWalker) Next() (name string, entry TreeEntry, err error) {
	var obj Object
	for {
		current := len(w.stack) - 1
		if current < 0 {
			// Nothing left on the stack so we're finished
			err = io.EOF
			return
		}

		if current > maxTreeDepth {
			// We're probably following bad data or some self-referencing tree
			err = ErrMaxTreeDepth
			return
		}

		entry, err = w.stack[current].Next()
		if err == io.EOF {
			// Finished with the current tree, move back up to the parent
			w.stack = w.stack[:current]
			w.base, _ = path.Split(w.base)
			w.base = path.Clean(w.base) // Remove trailing slash
			continue
		}

		if err != nil {
			return
		}

		if w.seen[entry.Hash] {
			continue
		}

		if entry.Mode == filemode.Dir {
			obj, err = GetTree(w.s, entry.Hash)
		}

		name = path.Join(w.base, entry.Name)

		if err != nil {
			err = io.EOF
			return
		}

		break
	}

	if !w.recursive {
		return
	}

	if t, ok := obj.(*Tree); ok {
		w.stack = append(w.stack, &treeEntryIter{t, 0})
		w.base = path.Join(w.base, entry.Name)
	}

	return
}

// Tree returns the tree that the tree walker most recently operated on.
func (w *TreeWalker) Tree() *Tree {
	current := len(w.stack) - 1
	if w.stack[current].pos == 0 {
		current--
	}

	if current < 0 {
		return nil
	}

	return w.stack[current].t
}

// Close releases any resources used by the TreeWalker.
func (w *TreeWalker) Close() {
	w.stack = nil
}

// TreeIter provides an iterator for a set of trees.
type TreeIter struct {
	storer.EncodedObjectIter
	s storer.EncodedObjectStorer
}

// NewTreeIter takes a storer.EncodedObjectStorer and a
// storer.EncodedObjectIter and returns a *TreeIter that iterates over all
// tree contained in the storer.EncodedObjectIter.
//
// Any non-tree object returned by the storer.EncodedObjectIter is skipped.
func NewTreeIter(s storer.EncodedObjectStorer, iter storer.EncodedObjectIter) *TreeIter {
	return &TreeIter{iter, s}
}

// Next moves the iterator to the next tree and returns a pointer to it. If
// there are no more trees, it returns io.EOF.
func (iter *TreeIter) Next() (*Tree, error) {
	for {
		obj, err := iter.EncodedObjectIter.Next()
		if err != nil {
			return nil, err
		}

		if obj.Type() != plumbing.TreeObject {
			continue
		}

		return DecodeTree(iter.s, obj)
	}
}

// ForEach call the cb function for each tree contained on this iter until
// an error happens or the end of the iter is reached. If ErrStop is sent
// the iteration is stop but no error is returned. The iterator is closed.
func (iter *TreeIter) ForEach(cb func(*Tree) error) error {
	return iter.EncodedObjectIter.ForEach(func(obj plumbing.EncodedObject) error {
		if obj.Type() != plumbing.TreeObject {
			return nil
		}

		t, err := DecodeTree(iter.s, obj)
		if err != nil {
			return err
		}

		return cb(t)
	})
}
