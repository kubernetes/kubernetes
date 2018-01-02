package object

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"strings"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/utils/ioutil"
)

// Hash represents the hash of an object
type Hash plumbing.Hash

// Commit points to a single tree, marking it as what the project looked like
// at a certain point in time. It contains meta-information about that point
// in time, such as a timestamp, the author of the changes since the last
// commit, a pointer to the previous commit(s), etc.
// http://schacon.github.io/gitbook/1_the_git_object_model.html
type Commit struct {
	// Hash of the commit object.
	Hash plumbing.Hash
	// Author is the original author of the commit.
	Author Signature
	// Committer is the one performing the commit, might be different from
	// Author.
	Committer Signature
	// Message is the commit message, contains arbitrary text.
	Message string
	// TreeHash is the hash of the root tree of the commit.
	TreeHash plumbing.Hash
	// ParentHashes are the hashes of the parent commits of the commit.
	ParentHashes []plumbing.Hash

	s storer.EncodedObjectStorer
}

// GetCommit gets a commit from an object storer and decodes it.
func GetCommit(s storer.EncodedObjectStorer, h plumbing.Hash) (*Commit, error) {
	o, err := s.EncodedObject(plumbing.CommitObject, h)
	if err != nil {
		return nil, err
	}

	return DecodeCommit(s, o)
}

// DecodeCommit decodes an encoded object into a *Commit and associates it to
// the given object storer.
func DecodeCommit(s storer.EncodedObjectStorer, o plumbing.EncodedObject) (*Commit, error) {
	c := &Commit{s: s}
	if err := c.Decode(o); err != nil {
		return nil, err
	}

	return c, nil
}

// Tree returns the Tree from the commit.
func (c *Commit) Tree() (*Tree, error) {
	return GetTree(c.s, c.TreeHash)
}

// Patch returns the Patch between the actual commit and the provided one.
func (c *Commit) Patch(to *Commit) (*Patch, error) {
	fromTree, err := c.Tree()
	if err != nil {
		return nil, err
	}

	toTree, err := to.Tree()
	if err != nil {
		return nil, err
	}

	return fromTree.Patch(toTree)
}

// Parents return a CommitIter to the parent Commits.
func (c *Commit) Parents() CommitIter {
	return NewCommitIter(c.s,
		storer.NewEncodedObjectLookupIter(c.s, plumbing.CommitObject, c.ParentHashes),
	)
}

// NumParents returns the number of parents in a commit.
func (c *Commit) NumParents() int {
	return len(c.ParentHashes)
}

// File returns the file with the specified "path" in the commit and a
// nil error if the file exists. If the file does not exist, it returns
// a nil file and the ErrFileNotFound error.
func (c *Commit) File(path string) (*File, error) {
	tree, err := c.Tree()
	if err != nil {
		return nil, err
	}

	return tree.File(path)
}

// Files returns a FileIter allowing to iterate over the Tree
func (c *Commit) Files() (*FileIter, error) {
	tree, err := c.Tree()
	if err != nil {
		return nil, err
	}

	return tree.Files(), nil
}

// ID returns the object ID of the commit. The returned value will always match
// the current value of Commit.Hash.
//
// ID is present to fulfill the Object interface.
func (c *Commit) ID() plumbing.Hash {
	return c.Hash
}

// Type returns the type of object. It always returns plumbing.CommitObject.
//
// Type is present to fulfill the Object interface.
func (c *Commit) Type() plumbing.ObjectType {
	return plumbing.CommitObject
}

// Decode transforms a plumbing.EncodedObject into a Commit struct.
func (c *Commit) Decode(o plumbing.EncodedObject) (err error) {
	if o.Type() != plumbing.CommitObject {
		return ErrUnsupportedObject
	}

	c.Hash = o.Hash()

	reader, err := o.Reader()
	if err != nil {
		return err
	}
	defer ioutil.CheckClose(reader, &err)

	r := bufio.NewReader(reader)

	var message bool
	for {
		line, err := r.ReadBytes('\n')
		if err != nil && err != io.EOF {
			return err
		}

		if !message {
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				message = true
				continue
			}

			split := bytes.SplitN(line, []byte{' '}, 2)
			switch string(split[0]) {
			case "tree":
				c.TreeHash = plumbing.NewHash(string(split[1]))
			case "parent":
				c.ParentHashes = append(c.ParentHashes, plumbing.NewHash(string(split[1])))
			case "author":
				c.Author.Decode(split[1])
			case "committer":
				c.Committer.Decode(split[1])
			}
		} else {
			c.Message += string(line)
		}

		if err == io.EOF {
			return nil
		}
	}
}

// Encode transforms a Commit into a plumbing.EncodedObject.
func (b *Commit) Encode(o plumbing.EncodedObject) error {
	o.SetType(plumbing.CommitObject)
	w, err := o.Writer()
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(w, &err)

	if _, err = fmt.Fprintf(w, "tree %s\n", b.TreeHash.String()); err != nil {
		return err
	}

	for _, parent := range b.ParentHashes {
		if _, err = fmt.Fprintf(w, "parent %s\n", parent.String()); err != nil {
			return err
		}
	}

	if _, err = fmt.Fprint(w, "author "); err != nil {
		return err
	}

	if err = b.Author.Encode(w); err != nil {
		return err
	}

	if _, err = fmt.Fprint(w, "\ncommitter "); err != nil {
		return err
	}

	if err = b.Committer.Encode(w); err != nil {
		return err
	}

	if _, err = fmt.Fprintf(w, "\n\n%s", b.Message); err != nil {
		return err
	}

	return err
}

func (c *Commit) String() string {
	return fmt.Sprintf(
		"%s %s\nAuthor: %s\nDate:   %s\n\n%s\n",
		plumbing.CommitObject, c.Hash, c.Author.String(),
		c.Author.When.Format(DateFormat), indent(c.Message),
	)
}

func indent(t string) string {
	var output []string
	for _, line := range strings.Split(t, "\n") {
		if len(line) != 0 {
			line = "    " + line
		}

		output = append(output, line)
	}

	return strings.Join(output, "\n")
}

// CommitIter is a generic closable interface for iterating over commits.
type CommitIter interface {
	Next() (*Commit, error)
	ForEach(func(*Commit) error) error
	Close()
}

// storerCommitIter provides an iterator from commits in an EncodedObjectStorer.
type storerCommitIter struct {
	storer.EncodedObjectIter
	s storer.EncodedObjectStorer
}

// NewCommitIter takes a storer.EncodedObjectStorer and a
// storer.EncodedObjectIter and returns a CommitIter that iterates over all
// commits contained in the storer.EncodedObjectIter.
//
// Any non-commit object returned by the storer.EncodedObjectIter is skipped.
func NewCommitIter(s storer.EncodedObjectStorer, iter storer.EncodedObjectIter) CommitIter {
	return &storerCommitIter{iter, s}
}

// Next moves the iterator to the next commit and returns a pointer to it. If
// there are no more commits, it returns io.EOF.
func (iter *storerCommitIter) Next() (*Commit, error) {
	obj, err := iter.EncodedObjectIter.Next()
	if err != nil {
		return nil, err
	}

	return DecodeCommit(iter.s, obj)
}

// ForEach call the cb function for each commit contained on this iter until
// an error appends or the end of the iter is reached. If ErrStop is sent
// the iteration is stopped but no error is returned. The iterator is closed.
func (iter *storerCommitIter) ForEach(cb func(*Commit) error) error {
	return iter.EncodedObjectIter.ForEach(func(obj plumbing.EncodedObject) error {
		c, err := DecodeCommit(iter.s, obj)
		if err != nil {
			return err
		}

		return cb(c)
	})
}

func (iter *storerCommitIter) Close() {
	iter.EncodedObjectIter.Close()
}
