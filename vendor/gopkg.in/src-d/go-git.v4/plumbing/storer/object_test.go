package storer

import (
	"fmt"
	"testing"

	. "gopkg.in/check.v1"
	"gopkg.in/src-d/go-git.v4/plumbing"
)

func Test(t *testing.T) { TestingT(t) }

type ObjectSuite struct {
	Objects []plumbing.EncodedObject
	Hash    []plumbing.Hash
}

var _ = Suite(&ObjectSuite{})

func (s *ObjectSuite) SetUpSuite(c *C) {
	s.Objects = []plumbing.EncodedObject{
		s.buildObject([]byte("foo")),
		s.buildObject([]byte("bar")),
	}

	for _, o := range s.Objects {
		s.Hash = append(s.Hash, o.Hash())
	}
}

func (s *ObjectSuite) TestMultiObjectIterNext(c *C) {
	expected := []plumbing.EncodedObject{
		&plumbing.MemoryObject{},
		&plumbing.MemoryObject{},
		&plumbing.MemoryObject{},
		&plumbing.MemoryObject{},
		&plumbing.MemoryObject{},
		&plumbing.MemoryObject{},
	}

	iter := NewMultiEncodedObjectIter([]EncodedObjectIter{
		NewEncodedObjectSliceIter(expected[0:2]),
		NewEncodedObjectSliceIter(expected[2:4]),
		NewEncodedObjectSliceIter(expected[4:5]),
	})

	var i int
	iter.ForEach(func(o plumbing.EncodedObject) error {
		c.Assert(o, Equals, expected[i])
		i++
		return nil
	})

	iter.Close()
}

func (s *ObjectSuite) buildObject(content []byte) plumbing.EncodedObject {
	o := &plumbing.MemoryObject{}
	o.Write(content)

	return o
}

func (s *ObjectSuite) TestObjectLookupIter(c *C) {
	var count int

	storage := &MockObjectStorage{s.Objects}
	i := NewEncodedObjectLookupIter(storage, plumbing.CommitObject, s.Hash)
	err := i.ForEach(func(o plumbing.EncodedObject) error {
		c.Assert(o, NotNil)
		c.Assert(o.Hash().String(), Equals, s.Hash[count].String())
		count++
		return nil
	})

	c.Assert(err, IsNil)
	i.Close()
}

func (s *ObjectSuite) TestObjectSliceIter(c *C) {
	var count int

	i := NewEncodedObjectSliceIter(s.Objects)
	err := i.ForEach(func(o plumbing.EncodedObject) error {
		c.Assert(o, NotNil)
		c.Assert(o.Hash().String(), Equals, s.Hash[count].String())
		count++
		return nil
	})

	c.Assert(count, Equals, 2)
	c.Assert(err, IsNil)
	c.Assert(i.series, HasLen, 0)
}

func (s *ObjectSuite) TestObjectSliceIterStop(c *C) {
	i := NewEncodedObjectSliceIter(s.Objects)

	var count = 0
	err := i.ForEach(func(o plumbing.EncodedObject) error {
		c.Assert(o, NotNil)
		c.Assert(o.Hash().String(), Equals, s.Hash[count].String())
		count++
		return ErrStop
	})

	c.Assert(count, Equals, 1)
	c.Assert(err, IsNil)
}

func (s *ObjectSuite) TestObjectSliceIterError(c *C) {
	i := NewEncodedObjectSliceIter([]plumbing.EncodedObject{
		s.buildObject([]byte("foo")),
	})

	err := i.ForEach(func(plumbing.EncodedObject) error {
		return fmt.Errorf("a random error")
	})

	c.Assert(err, NotNil)
}

type MockObjectStorage struct {
	db []plumbing.EncodedObject
}

func (o *MockObjectStorage) NewEncodedObject() plumbing.EncodedObject {
	return nil
}

func (o *MockObjectStorage) SetEncodedObject(obj plumbing.EncodedObject) (plumbing.Hash, error) {
	return plumbing.ZeroHash, nil
}

func (o *MockObjectStorage) EncodedObject(t plumbing.ObjectType, h plumbing.Hash) (plumbing.EncodedObject, error) {
	for _, o := range o.db {
		if o.Hash() == h {
			return o, nil
		}
	}
	return nil, plumbing.ErrObjectNotFound
}

func (o *MockObjectStorage) IterEncodedObjects(t plumbing.ObjectType) (EncodedObjectIter, error) {
	return nil, nil
}

func (o *MockObjectStorage) Begin() Transaction {
	return nil
}
