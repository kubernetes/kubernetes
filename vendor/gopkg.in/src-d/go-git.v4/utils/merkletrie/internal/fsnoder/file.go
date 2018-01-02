package fsnoder

import (
	"bytes"
	"fmt"
	"hash/fnv"

	"gopkg.in/src-d/go-git.v4/utils/merkletrie/noder"
)

// file values represent file-like noders in a merkle trie.
type file struct {
	name     string // relative
	contents string
	hash     []byte // memoized
}

// newFile returns a noder representing a file with the given contents.
func newFile(name, contents string) (*file, error) {
	if name == "" {
		return nil, fmt.Errorf("files cannot have empty names")
	}

	return &file{
		name:     name,
		contents: contents,
	}, nil
}

// The hash of a file is just its contents.
// Empty files will have the fnv64 basis offset as its hash.
func (f *file) Hash() []byte {
	if f.hash == nil {
		h := fnv.New64a()
		h.Write([]byte(f.contents)) // it nevers returns an error.
		f.hash = h.Sum(nil)
	}

	return f.hash
}

func (f *file) Name() string {
	return f.name
}

func (f *file) IsDir() bool {
	return false
}

func (f *file) Children() ([]noder.Noder, error) {
	return noder.NoChildren, nil
}

func (f *file) NumChildren() (int, error) {
	return 0, nil
}

const (
	fileStartMark = '<'
	fileEndMark   = '>'
)

// String returns a string formated as: name<contents>.
func (f *file) String() string {
	var buf bytes.Buffer
	buf.WriteString(f.name)
	buf.WriteRune(fileStartMark)
	buf.WriteString(f.contents)
	buf.WriteRune(fileEndMark)

	return buf.String()
}
