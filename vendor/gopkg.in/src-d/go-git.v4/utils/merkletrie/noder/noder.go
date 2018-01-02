// Package noder provide an interface for defining nodes in a
// merkletrie, their hashes and their paths (a noders and its
// ancestors).
//
// The hasher interface is easy to implement naively by elements that
// already have a hash, like git blobs and trees.  More sophisticated
// implementations can implement the Equal function in exotic ways
// though: for instance, comparing the modification time of directories
// in a filesystem.
package noder

import "fmt"

// Hasher interface is implemented by types that can tell you
// their hash.
type Hasher interface {
	Hash() []byte
}

// Equal functions take two hashers and return if they are equal.
//
// These functions are expected to be faster than reflect.Equal or
// reflect.DeepEqual because they can compare just the hash of the
// objects, instead of their contents, so they are expected to be O(1).
type Equal func(a, b Hasher) bool

// The Noder interface is implemented by the elements of a Merkle Trie.
//
// There are two types of elements in a Merkle Trie:
//
// - file-like nodes: they cannot have children.
//
// - directory-like nodes: they can have 0 or more children and their
// hash is calculated by combining their children hashes.
type Noder interface {
	Hasher
	fmt.Stringer // for testing purposes
	// Name returns the name of an element (relative, not its full
	// path).
	Name() string
	// IsDir returns true if the element is a directory-like node or
	// false if it is a file-like node.
	IsDir() bool
	// Children returns the children of the element.  Note that empty
	// directory-like noders and file-like noders will both return
	// NoChildren.
	Children() ([]Noder, error)
	// NumChildren returns the number of children this element has.
	//
	// This method is an optimization: the number of children is easily
	// calculated as the length of the value returned by the Children
	// method (above); yet, some implementations will be able to
	// implement NumChildren in O(1) while Children is usually more
	// complex.
	NumChildren() (int, error)
}

// NoChildren represents the children of a noder without children.
var NoChildren = []Noder{}
