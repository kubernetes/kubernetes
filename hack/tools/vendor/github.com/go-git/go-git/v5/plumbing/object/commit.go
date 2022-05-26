package object

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"golang.org/x/crypto/openpgp"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/storer"
	"github.com/go-git/go-git/v5/utils/ioutil"
)

const (
	beginpgp  string = "-----BEGIN PGP SIGNATURE-----"
	endpgp    string = "-----END PGP SIGNATURE-----"
	headerpgp string = "gpgsig"
)

// Hash represents the hash of an object
type Hash plumbing.Hash

// Commit points to a single tree, marking it as what the project looked like
// at a certain point in time. It contains meta-information about that point
// in time, such as a timestamp, the author of the changes since the last
// commit, a pointer to the previous commit(s), etc.
// http://shafiulazam.com/gitbook/1_the_git_object_model.html
type Commit struct {
	// Hash of the commit object.
	Hash plumbing.Hash
	// Author is the original author of the commit.
	Author Signature
	// Committer is the one performing the commit, might be different from
	// Author.
	Committer Signature
	// PGPSignature is the PGP signature of the commit.
	PGPSignature string
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

// PatchContext returns the Patch between the actual commit and the provided one.
// Error will be return if context expires. Provided context must be non-nil.
//
// NOTE: Since version 5.1.0 the renames are correctly handled, the settings
// used are the recommended options DefaultDiffTreeOptions.
func (c *Commit) PatchContext(ctx context.Context, to *Commit) (*Patch, error) {
	fromTree, err := c.Tree()
	if err != nil {
		return nil, err
	}

	var toTree *Tree
	if to != nil {
		toTree, err = to.Tree()
		if err != nil {
			return nil, err
		}
	}

	return fromTree.PatchContext(ctx, toTree)
}

// Patch returns the Patch between the actual commit and the provided one.
//
// NOTE: Since version 5.1.0 the renames are correctly handled, the settings
// used are the recommended options DefaultDiffTreeOptions.
func (c *Commit) Patch(to *Commit) (*Patch, error) {
	return c.PatchContext(context.Background(), to)
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

var ErrParentNotFound = errors.New("commit parent not found")

// Parent returns the ith parent of a commit.
func (c *Commit) Parent(i int) (*Commit, error) {
	if len(c.ParentHashes) == 0 || i > len(c.ParentHashes)-1 {
		return nil, ErrParentNotFound
	}

	return GetCommit(c.s, c.ParentHashes[i])
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

	r := bufPool.Get().(*bufio.Reader)
	defer bufPool.Put(r)
	r.Reset(reader)

	var message bool
	var pgpsig bool
	var msgbuf bytes.Buffer
	for {
		line, err := r.ReadBytes('\n')
		if err != nil && err != io.EOF {
			return err
		}

		if pgpsig {
			if len(line) > 0 && line[0] == ' ' {
				line = bytes.TrimLeft(line, " ")
				c.PGPSignature += string(line)
				continue
			} else {
				pgpsig = false
			}
		}

		if !message {
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				message = true
				continue
			}

			split := bytes.SplitN(line, []byte{' '}, 2)

			var data []byte
			if len(split) == 2 {
				data = split[1]
			}

			switch string(split[0]) {
			case "tree":
				c.TreeHash = plumbing.NewHash(string(data))
			case "parent":
				c.ParentHashes = append(c.ParentHashes, plumbing.NewHash(string(data)))
			case "author":
				c.Author.Decode(data)
			case "committer":
				c.Committer.Decode(data)
			case headerpgp:
				c.PGPSignature += string(data) + "\n"
				pgpsig = true
			}
		} else {
			msgbuf.Write(line)
		}

		if err == io.EOF {
			break
		}
	}
	c.Message = msgbuf.String()
	return nil
}

// Encode transforms a Commit into a plumbing.EncodedObject.
func (c *Commit) Encode(o plumbing.EncodedObject) error {
	return c.encode(o, true)
}

// EncodeWithoutSignature export a Commit into a plumbing.EncodedObject without the signature (correspond to the payload of the PGP signature).
func (c *Commit) EncodeWithoutSignature(o plumbing.EncodedObject) error {
	return c.encode(o, false)
}

func (c *Commit) encode(o plumbing.EncodedObject, includeSig bool) (err error) {
	o.SetType(plumbing.CommitObject)
	w, err := o.Writer()
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(w, &err)

	if _, err = fmt.Fprintf(w, "tree %s\n", c.TreeHash.String()); err != nil {
		return err
	}

	for _, parent := range c.ParentHashes {
		if _, err = fmt.Fprintf(w, "parent %s\n", parent.String()); err != nil {
			return err
		}
	}

	if _, err = fmt.Fprint(w, "author "); err != nil {
		return err
	}

	if err = c.Author.Encode(w); err != nil {
		return err
	}

	if _, err = fmt.Fprint(w, "\ncommitter "); err != nil {
		return err
	}

	if err = c.Committer.Encode(w); err != nil {
		return err
	}

	if c.PGPSignature != "" && includeSig {
		if _, err = fmt.Fprint(w, "\n"+headerpgp+" "); err != nil {
			return err
		}

		// Split all the signature lines and re-write with a left padding and
		// newline. Use join for this so it's clear that a newline should not be
		// added after this section, as it will be added when the message is
		// printed.
		signature := strings.TrimSuffix(c.PGPSignature, "\n")
		lines := strings.Split(signature, "\n")
		if _, err = fmt.Fprint(w, strings.Join(lines, "\n ")); err != nil {
			return err
		}
	}

	if _, err = fmt.Fprintf(w, "\n\n%s", c.Message); err != nil {
		return err
	}

	return err
}

// Stats returns the stats of a commit.
func (c *Commit) Stats() (FileStats, error) {
	return c.StatsContext(context.Background())
}

// StatsContext returns the stats of a commit. Error will be return if context
// expires. Provided context must be non-nil.
func (c *Commit) StatsContext(ctx context.Context) (FileStats, error) {
	fromTree, err := c.Tree()
	if err != nil {
		return nil, err
	}

	toTree := &Tree{}
	if c.NumParents() != 0 {
		firstParent, err := c.Parents().Next()
		if err != nil {
			return nil, err
		}

		toTree, err = firstParent.Tree()
		if err != nil {
			return nil, err
		}
	}

	patch, err := toTree.PatchContext(ctx, fromTree)
	if err != nil {
		return nil, err
	}

	return getFileStatsFromFilePatches(patch.FilePatches()), nil
}

func (c *Commit) String() string {
	return fmt.Sprintf(
		"%s %s\nAuthor: %s\nDate:   %s\n\n%s\n",
		plumbing.CommitObject, c.Hash, c.Author.String(),
		c.Author.When.Format(DateFormat), indent(c.Message),
	)
}

// Verify performs PGP verification of the commit with a provided armored
// keyring and returns openpgp.Entity associated with verifying key on success.
func (c *Commit) Verify(armoredKeyRing string) (*openpgp.Entity, error) {
	keyRingReader := strings.NewReader(armoredKeyRing)
	keyring, err := openpgp.ReadArmoredKeyRing(keyRingReader)
	if err != nil {
		return nil, err
	}

	// Extract signature.
	signature := strings.NewReader(c.PGPSignature)

	encoded := &plumbing.MemoryObject{}
	// Encode commit components, excluding signature and get a reader object.
	if err := c.EncodeWithoutSignature(encoded); err != nil {
		return nil, err
	}
	er, err := encoded.Reader()
	if err != nil {
		return nil, err
	}

	return openpgp.CheckArmoredDetachedSignature(keyring, er, signature)
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
