package index

import (
	"path"
	"strings"

	"github.com/go-git/go-git/v5/plumbing/format/index"
	"github.com/go-git/go-git/v5/utils/merkletrie/noder"
)

// The node represents a index.Entry or a directory inferred from the path
// of all entries. It implements the interface noder.Noder of merkletrie
// package.
//
// This implementation implements a "standard" hash method being able to be
// compared with any other noder.Noder implementation inside of go-git
type node struct {
	path     string
	entry    *index.Entry
	children []noder.Noder
	isDir    bool
}

// NewRootNode returns the root node of a computed tree from a index.Index,
func NewRootNode(idx *index.Index) noder.Noder {
	const rootNode = ""

	m := map[string]*node{rootNode: {isDir: true}}

	for _, e := range idx.Entries {
		parts := strings.Split(e.Name, string("/"))

		var fullpath string
		for _, part := range parts {
			parent := fullpath
			fullpath = path.Join(fullpath, part)

			if _, ok := m[fullpath]; ok {
				continue
			}

			n := &node{path: fullpath}
			if fullpath == e.Name {
				n.entry = e
			} else {
				n.isDir = true
			}

			m[n.path] = n
			m[parent].children = append(m[parent].children, n)
		}
	}

	return m[rootNode]
}

func (n *node) String() string {
	return n.path
}

// Hash the hash of a filesystem is a 24-byte slice, is the result of
// concatenating the computed plumbing.Hash of the file as a Blob and its
// plumbing.FileMode; that way the difftree algorithm will detect changes in the
// contents of files and also in their mode.
//
// If the node is computed and not based on a index.Entry the hash is equals
// to a 24-bytes slices of zero values.
func (n *node) Hash() []byte {
	if n.entry == nil {
		return make([]byte, 24)
	}

	return append(n.entry.Hash[:], n.entry.Mode.Bytes()...)
}

func (n *node) Name() string {
	return path.Base(n.path)
}

func (n *node) IsDir() bool {
	return n.isDir
}

func (n *node) Children() ([]noder.Noder, error) {
	return n.children, nil
}

func (n *node) NumChildren() (int, error) {
	return len(n.children), nil
}
