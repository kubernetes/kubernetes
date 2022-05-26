package object

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	stdioutil "io/ioutil"
	"strings"

	"golang.org/x/crypto/openpgp"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/storer"
	"github.com/go-git/go-git/v5/utils/ioutil"
)

// Tag represents an annotated tag object. It points to a single git object of
// any type, but tags typically are applied to commit or blob objects. It
// provides a reference that associates the target with a tag name. It also
// contains meta-information about the tag, including the tagger, tag date and
// message.
//
// Note that this is not used for lightweight tags.
//
// https://git-scm.com/book/en/v2/Git-Internals-Git-References#Tags
type Tag struct {
	// Hash of the tag.
	Hash plumbing.Hash
	// Name of the tag.
	Name string
	// Tagger is the one who created the tag.
	Tagger Signature
	// Message is an arbitrary text message.
	Message string
	// PGPSignature is the PGP signature of the tag.
	PGPSignature string
	// TargetType is the object type of the target.
	TargetType plumbing.ObjectType
	// Target is the hash of the target object.
	Target plumbing.Hash

	s storer.EncodedObjectStorer
}

// GetTag gets a tag from an object storer and decodes it.
func GetTag(s storer.EncodedObjectStorer, h plumbing.Hash) (*Tag, error) {
	o, err := s.EncodedObject(plumbing.TagObject, h)
	if err != nil {
		return nil, err
	}

	return DecodeTag(s, o)
}

// DecodeTag decodes an encoded object into a *Commit and associates it to the
// given object storer.
func DecodeTag(s storer.EncodedObjectStorer, o plumbing.EncodedObject) (*Tag, error) {
	t := &Tag{s: s}
	if err := t.Decode(o); err != nil {
		return nil, err
	}

	return t, nil
}

// ID returns the object ID of the tag, not the object that the tag references.
// The returned value will always match the current value of Tag.Hash.
//
// ID is present to fulfill the Object interface.
func (t *Tag) ID() plumbing.Hash {
	return t.Hash
}

// Type returns the type of object. It always returns plumbing.TagObject.
//
// Type is present to fulfill the Object interface.
func (t *Tag) Type() plumbing.ObjectType {
	return plumbing.TagObject
}

// Decode transforms a plumbing.EncodedObject into a Tag struct.
func (t *Tag) Decode(o plumbing.EncodedObject) (err error) {
	if o.Type() != plumbing.TagObject {
		return ErrUnsupportedObject
	}

	t.Hash = o.Hash()

	reader, err := o.Reader()
	if err != nil {
		return err
	}
	defer ioutil.CheckClose(reader, &err)

	r := bufPool.Get().(*bufio.Reader)
	defer bufPool.Put(r)
	r.Reset(reader)
	for {
		var line []byte
		line, err = r.ReadBytes('\n')
		if err != nil && err != io.EOF {
			return err
		}

		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			break // Start of message
		}

		split := bytes.SplitN(line, []byte{' '}, 2)
		switch string(split[0]) {
		case "object":
			t.Target = plumbing.NewHash(string(split[1]))
		case "type":
			t.TargetType, err = plumbing.ParseObjectType(string(split[1]))
			if err != nil {
				return err
			}
		case "tag":
			t.Name = string(split[1])
		case "tagger":
			t.Tagger.Decode(split[1])
		}

		if err == io.EOF {
			return nil
		}
	}

	data, err := stdioutil.ReadAll(r)
	if err != nil {
		return err
	}

	var pgpsig bool
	// Check if data contains PGP signature.
	if bytes.Contains(data, []byte(beginpgp)) {
		// Split the lines at newline.
		messageAndSig := bytes.Split(data, []byte("\n"))

		for _, l := range messageAndSig {
			if pgpsig {
				if bytes.Contains(l, []byte(endpgp)) {
					t.PGPSignature += endpgp + "\n"
					break
				} else {
					t.PGPSignature += string(l) + "\n"
				}
				continue
			}

			// Check if it's the beginning of a PGP signature.
			if bytes.Contains(l, []byte(beginpgp)) {
				t.PGPSignature += beginpgp + "\n"
				pgpsig = true
				continue
			}

			t.Message += string(l) + "\n"
		}
	} else {
		t.Message = string(data)
	}

	return nil
}

// Encode transforms a Tag into a plumbing.EncodedObject.
func (t *Tag) Encode(o plumbing.EncodedObject) error {
	return t.encode(o, true)
}

// EncodeWithoutSignature export a Tag into a plumbing.EncodedObject without the signature (correspond to the payload of the PGP signature).
func (t *Tag) EncodeWithoutSignature(o plumbing.EncodedObject) error {
	return t.encode(o, false)
}

func (t *Tag) encode(o plumbing.EncodedObject, includeSig bool) (err error) {
	o.SetType(plumbing.TagObject)
	w, err := o.Writer()
	if err != nil {
		return err
	}
	defer ioutil.CheckClose(w, &err)

	if _, err = fmt.Fprintf(w,
		"object %s\ntype %s\ntag %s\ntagger ",
		t.Target.String(), t.TargetType.Bytes(), t.Name); err != nil {
		return err
	}

	if err = t.Tagger.Encode(w); err != nil {
		return err
	}

	if _, err = fmt.Fprint(w, "\n\n"); err != nil {
		return err
	}

	if _, err = fmt.Fprint(w, t.Message); err != nil {
		return err
	}

	// Note that this is highly sensitive to what it sent along in the message.
	// Message *always* needs to end with a newline, or else the message and the
	// signature will be concatenated into a corrupt object. Since this is a
	// lower-level method, we assume you know what you are doing and have already
	// done the needful on the message in the caller.
	if includeSig {
		if _, err = fmt.Fprint(w, t.PGPSignature); err != nil {
			return err
		}
	}

	return err
}

// Commit returns the commit pointed to by the tag. If the tag points to a
// different type of object ErrUnsupportedObject will be returned.
func (t *Tag) Commit() (*Commit, error) {
	if t.TargetType != plumbing.CommitObject {
		return nil, ErrUnsupportedObject
	}

	o, err := t.s.EncodedObject(plumbing.CommitObject, t.Target)
	if err != nil {
		return nil, err
	}

	return DecodeCommit(t.s, o)
}

// Tree returns the tree pointed to by the tag. If the tag points to a commit
// object the tree of that commit will be returned. If the tag does not point
// to a commit or tree object ErrUnsupportedObject will be returned.
func (t *Tag) Tree() (*Tree, error) {
	switch t.TargetType {
	case plumbing.CommitObject:
		c, err := t.Commit()
		if err != nil {
			return nil, err
		}

		return c.Tree()
	case plumbing.TreeObject:
		return GetTree(t.s, t.Target)
	default:
		return nil, ErrUnsupportedObject
	}
}

// Blob returns the blob pointed to by the tag. If the tag points to a
// different type of object ErrUnsupportedObject will be returned.
func (t *Tag) Blob() (*Blob, error) {
	if t.TargetType != plumbing.BlobObject {
		return nil, ErrUnsupportedObject
	}

	return GetBlob(t.s, t.Target)
}

// Object returns the object pointed to by the tag.
func (t *Tag) Object() (Object, error) {
	o, err := t.s.EncodedObject(t.TargetType, t.Target)
	if err != nil {
		return nil, err
	}

	return DecodeObject(t.s, o)
}

// String returns the meta information contained in the tag as a formatted
// string.
func (t *Tag) String() string {
	obj, _ := t.Object()

	return fmt.Sprintf(
		"%s %s\nTagger: %s\nDate:   %s\n\n%s\n%s",
		plumbing.TagObject, t.Name, t.Tagger.String(), t.Tagger.When.Format(DateFormat),
		t.Message, objectAsString(obj),
	)
}

// Verify performs PGP verification of the tag with a provided armored
// keyring and returns openpgp.Entity associated with verifying key on success.
func (t *Tag) Verify(armoredKeyRing string) (*openpgp.Entity, error) {
	keyRingReader := strings.NewReader(armoredKeyRing)
	keyring, err := openpgp.ReadArmoredKeyRing(keyRingReader)
	if err != nil {
		return nil, err
	}

	// Extract signature.
	signature := strings.NewReader(t.PGPSignature)

	encoded := &plumbing.MemoryObject{}
	// Encode tag components, excluding signature and get a reader object.
	if err := t.EncodeWithoutSignature(encoded); err != nil {
		return nil, err
	}
	er, err := encoded.Reader()
	if err != nil {
		return nil, err
	}

	return openpgp.CheckArmoredDetachedSignature(keyring, er, signature)
}

// TagIter provides an iterator for a set of tags.
type TagIter struct {
	storer.EncodedObjectIter
	s storer.EncodedObjectStorer
}

// NewTagIter takes a storer.EncodedObjectStorer and a
// storer.EncodedObjectIter and returns a *TagIter that iterates over all
// tags contained in the storer.EncodedObjectIter.
//
// Any non-tag object returned by the storer.EncodedObjectIter is skipped.
func NewTagIter(s storer.EncodedObjectStorer, iter storer.EncodedObjectIter) *TagIter {
	return &TagIter{iter, s}
}

// Next moves the iterator to the next tag and returns a pointer to it. If
// there are no more tags, it returns io.EOF.
func (iter *TagIter) Next() (*Tag, error) {
	obj, err := iter.EncodedObjectIter.Next()
	if err != nil {
		return nil, err
	}

	return DecodeTag(iter.s, obj)
}

// ForEach call the cb function for each tag contained on this iter until
// an error happens or the end of the iter is reached. If ErrStop is sent
// the iteration is stop but no error is returned. The iterator is closed.
func (iter *TagIter) ForEach(cb func(*Tag) error) error {
	return iter.EncodedObjectIter.ForEach(func(obj plumbing.EncodedObject) error {
		t, err := DecodeTag(iter.s, obj)
		if err != nil {
			return err
		}

		return cb(t)
	})
}

func objectAsString(obj Object) string {
	switch o := obj.(type) {
	case *Commit:
		return o.String()
	default:
		return ""
	}
}
