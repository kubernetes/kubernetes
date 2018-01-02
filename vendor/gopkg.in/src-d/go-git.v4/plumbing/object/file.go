package object

import (
	"bytes"
	"io"
	"strings"

	"gopkg.in/src-d/go-git.v4/plumbing/filemode"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/utils/binary"
	"gopkg.in/src-d/go-git.v4/utils/ioutil"
)

// File represents git file objects.
type File struct {
	// Name is the path of the file. It might be relative to a tree,
	// depending of the function that generates it.
	Name string
	// Mode is the file mode.
	Mode filemode.FileMode
	// Blob with the contents of the file.
	Blob
}

// NewFile returns a File based on the given blob object
func NewFile(name string, m filemode.FileMode, b *Blob) *File {
	return &File{Name: name, Mode: m, Blob: *b}
}

// Contents returns the contents of a file as a string.
func (f *File) Contents() (content string, err error) {
	reader, err := f.Reader()
	if err != nil {
		return "", err
	}
	defer ioutil.CheckClose(reader, &err)

	buf := new(bytes.Buffer)
	if _, err := buf.ReadFrom(reader); err != nil {
		return "", err
	}

	return buf.String(), nil
}

// IsBinary returns if the file is binary or not
func (f *File) IsBinary() (bool, error) {
	reader, err := f.Reader()
	if err != nil {
		return false, err
	}
	defer ioutil.CheckClose(reader, &err)

	return binary.IsBinary(reader)
}

// Lines returns a slice of lines from the contents of a file, stripping
// all end of line characters. If the last line is empty (does not end
// in an end of line), it is also stripped.
func (f *File) Lines() ([]string, error) {
	content, err := f.Contents()
	if err != nil {
		return nil, err
	}

	splits := strings.Split(content, "\n")
	// remove the last line if it is empty
	if splits[len(splits)-1] == "" {
		return splits[:len(splits)-1], nil
	}

	return splits, nil
}

// FileIter provides an iterator for the files in a tree.
type FileIter struct {
	s storer.EncodedObjectStorer
	w TreeWalker
}

// NewFileIter takes a storer.EncodedObjectStorer and a Tree and returns a
// *FileIter that iterates over all files contained in the tree, recursively.
func NewFileIter(s storer.EncodedObjectStorer, t *Tree) *FileIter {
	return &FileIter{s: s, w: *NewTreeWalker(t, true, nil)}
}

// Next moves the iterator to the next file and returns a pointer to it. If
// there are no more files, it returns io.EOF.
func (iter *FileIter) Next() (*File, error) {
	for {
		name, entry, err := iter.w.Next()
		if err != nil {
			return nil, err
		}

		if entry.Mode == filemode.Dir || entry.Mode == filemode.Submodule {
			continue
		}

		blob, err := GetBlob(iter.s, entry.Hash)
		if err != nil {
			return nil, err
		}

		return NewFile(name, entry.Mode, blob), nil
	}
}

// ForEach call the cb function for each file contained in this iter until
// an error happens or the end of the iter is reached. If plumbing.ErrStop is sent
// the iteration is stop but no error is returned. The iterator is closed.
func (iter *FileIter) ForEach(cb func(*File) error) error {
	defer iter.Close()

	for {
		f, err := iter.Next()
		if err != nil {
			if err == io.EOF {
				return nil
			}

			return err
		}

		if err := cb(f); err != nil {
			if err == storer.ErrStop {
				return nil
			}

			return err
		}
	}
}

func (iter *FileIter) Close() {
	iter.w.Close()
}
