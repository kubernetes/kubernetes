package packfile

import (
	"fmt"
	"io"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/object"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
)

// ObjectWalker walks a hash and makes a callback with each object it walks.  This
// will yield duplicates.
type ObjectWalker struct {
	objs storer.EncodedObjectStorer
}

// NewObjectWalker instantiates a new object walker with the given store
func NewObjectWalker(objs storer.EncodedObjectStorer) *ObjectWalker {
	return &ObjectWalker{objs: objs}
}

// Walk an object to the beginning of time
func (ow *ObjectWalker) Walk(hash plumbing.Hash, cb func(plumbing.EncodedObject) error) error {
	obj, err := ow.objs.EncodedObject(plumbing.AnyObject, hash)
	if err != nil {
		return err
	}

	if err = cb(obj); err != nil {
		return err
	}

	// Further walk the following object types
	switch obj.Type() {
	case plumbing.CommitObject:
		err = ow.walkCommit(obj, cb)

	case plumbing.TreeObject:
		err = ow.walkTree(obj, cb)

	}

	return err

}

func (ow *ObjectWalker) walkCommit(obj plumbing.EncodedObject, cb func(plumbing.EncodedObject) error) error {
	commit, err := object.GetCommit(ow.objs, obj.Hash())
	if err != nil {
		return err
	}

	var tobj *object.Tree
	if tobj, err = commit.Tree(); err != nil {
		return err
	}

	if err = ow.Walk(tobj.Hash, cb); err != nil {
		return err
	}

	iter := commit.Parents()
	for {
		cmt, e1 := iter.Next()
		if e1 != nil {
			if e1 != io.EOF {
				err = e1
			}
			break
		}

		if e1 = ow.Walk(cmt.Hash, cb); e1 != nil {
			err = e1
			break
		}
	}
	return err
}

func (ow *ObjectWalker) walkTree(obj plumbing.EncodedObject, cb func(plumbing.EncodedObject) error) error {
	t := &object.Tree{}
	err := t.Decode(obj)
	if err != nil {
		return err
	}

	for _, entry := range t.Entries {
		err = mergeErrors(err, ow.Walk(entry.Hash, cb))
	}
	return err
}

func mergeErrors(err1, err2 error) error {
	if err1 == nil {
		return err2
	} else if err2 == nil {
		return err1
	} else {
		return fmt.Errorf("%s\n%s", err1, err2)
	}
}
