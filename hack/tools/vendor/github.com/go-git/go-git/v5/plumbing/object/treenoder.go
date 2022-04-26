package object

import (
	"io"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/filemode"
	"github.com/go-git/go-git/v5/utils/merkletrie/noder"
)

// A treenoder is a helper type that wraps git trees into merkletrie
// noders.
//
// As a merkletrie noder doesn't understand the concept of modes (e.g.
// file permissions), the treenoder includes the mode of the git tree in
// the hash, so changes in the modes will be detected as modifications
// to the file contents by the merkletrie difftree algorithm.  This is
// consistent with how the "git diff-tree" command works.
type treeNoder struct {
	parent   *Tree  // the root node is its own parent
	name     string // empty string for the root node
	mode     filemode.FileMode
	hash     plumbing.Hash
	children []noder.Noder // memoized
}

// NewTreeRootNode returns the root node of a Tree
func NewTreeRootNode(t *Tree) noder.Noder {
	if t == nil {
		return &treeNoder{}
	}

	return &treeNoder{
		parent: t,
		name:   "",
		mode:   filemode.Dir,
		hash:   t.Hash,
	}
}

func (t *treeNoder) isRoot() bool {
	return t.name == ""
}

func (t *treeNoder) String() string {
	return "treeNoder <" + t.name + ">"
}

func (t *treeNoder) Hash() []byte {
	if t.mode == filemode.Deprecated {
		return append(t.hash[:], filemode.Regular.Bytes()...)
	}
	return append(t.hash[:], t.mode.Bytes()...)
}

func (t *treeNoder) Name() string {
	return t.name
}

func (t *treeNoder) IsDir() bool {
	return t.mode == filemode.Dir
}

// Children will return the children of a treenoder as treenoders,
// building them from the children of the wrapped git tree.
func (t *treeNoder) Children() ([]noder.Noder, error) {
	if t.mode != filemode.Dir {
		return noder.NoChildren, nil
	}

	// children are memoized for efficiency
	if t.children != nil {
		return t.children, nil
	}

	// the parent of the returned children will be ourself as a tree if
	// we are a not the root treenoder.  The root is special as it
	// is is own parent.
	parent := t.parent
	if !t.isRoot() {
		var err error
		if parent, err = t.parent.Tree(t.name); err != nil {
			return nil, err
		}
	}

	return transformChildren(parent)
}

// Returns the children of a tree as treenoders.
// Efficiency is key here.
func transformChildren(t *Tree) ([]noder.Noder, error) {
	var err error
	var e TreeEntry

	// there will be more tree entries than children in the tree,
	// due to submodules and empty directories, but I think it is still
	// worth it to pre-allocate the whole array now, even if sometimes
	// is bigger than needed.
	ret := make([]noder.Noder, 0, len(t.Entries))

	walker := NewTreeWalker(t, false, nil) // don't recurse
	// don't defer walker.Close() for efficiency reasons.
	for {
		_, e, err = walker.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			walker.Close()
			return nil, err
		}

		ret = append(ret, &treeNoder{
			parent: t,
			name:   e.Name,
			mode:   e.Mode,
			hash:   e.Hash,
		})
	}
	walker.Close()

	return ret, nil
}

// len(t.tree.Entries) != the number of elements walked by treewalker
// for some reason because of empty directories, submodules, etc, so we
// have to walk here.
func (t *treeNoder) NumChildren() (int, error) {
	children, err := t.Children()
	if err != nil {
		return 0, err
	}

	return len(children), nil
}
