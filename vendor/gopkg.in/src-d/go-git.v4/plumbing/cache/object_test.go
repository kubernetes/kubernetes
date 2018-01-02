package cache

import (
	"fmt"
	"io"
	"sync"
	"testing"

	"gopkg.in/src-d/go-git.v4/plumbing"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type ObjectSuite struct {
	c       Object
	aObject plumbing.EncodedObject
	bObject plumbing.EncodedObject
	cObject plumbing.EncodedObject
	dObject plumbing.EncodedObject
}

var _ = Suite(&ObjectSuite{})

func (s *ObjectSuite) SetUpTest(c *C) {
	s.aObject = newObject("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 1*Byte)
	s.bObject = newObject("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", 3*Byte)
	s.cObject = newObject("cccccccccccccccccccccccccccccccccccccccc", 1*Byte)
	s.dObject = newObject("dddddddddddddddddddddddddddddddddddddddd", 1*Byte)

	s.c = NewObjectLRU(2 * Byte)
}

func (s *ObjectSuite) TestPutSameObject(c *C) {
	s.c.Put(s.aObject)
	s.c.Put(s.aObject)
	_, ok := s.c.Get(s.aObject.Hash())
	c.Assert(ok, Equals, true)
}

func (s *ObjectSuite) TestPutBigObject(c *C) {
	s.c.Put(s.bObject)
	_, ok := s.c.Get(s.aObject.Hash())
	c.Assert(ok, Equals, false)
}

func (s *ObjectSuite) TestPutCacheOverflow(c *C) {
	s.c.Put(s.aObject)
	s.c.Put(s.cObject)
	s.c.Put(s.dObject)

	obj, ok := s.c.Get(s.aObject.Hash())
	c.Assert(ok, Equals, false)
	c.Assert(obj, IsNil)
	obj, ok = s.c.Get(s.cObject.Hash())
	c.Assert(ok, Equals, true)
	c.Assert(obj, NotNil)
	obj, ok = s.c.Get(s.dObject.Hash())
	c.Assert(ok, Equals, true)
	c.Assert(obj, NotNil)
}

func (s *ObjectSuite) TestClear(c *C) {
	s.c.Put(s.aObject)
	s.c.Clear()
	obj, ok := s.c.Get(s.aObject.Hash())
	c.Assert(ok, Equals, false)
	c.Assert(obj, IsNil)
}

func (s *ObjectSuite) TestConcurrentAccess(c *C) {
	var wg sync.WaitGroup

	for i := 0; i < 1000; i++ {
		wg.Add(3)
		go func(i int) {
			s.c.Put(newObject(fmt.Sprint(i), FileSize(i)))
			wg.Done()
		}(i)

		go func(i int) {
			if i%30 == 0 {
				s.c.Clear()
			}
			wg.Done()
		}(i)

		go func(i int) {
			s.c.Get(plumbing.NewHash(fmt.Sprint(i)))
			wg.Done()
		}(i)
	}

	wg.Wait()
}

type dummyObject struct {
	hash plumbing.Hash
	size FileSize
}

func newObject(hash string, size FileSize) plumbing.EncodedObject {
	return &dummyObject{
		hash: plumbing.NewHash(hash),
		size: size,
	}
}

func (d *dummyObject) Hash() plumbing.Hash           { return d.hash }
func (*dummyObject) Type() plumbing.ObjectType       { return plumbing.InvalidObject }
func (*dummyObject) SetType(plumbing.ObjectType)     {}
func (d *dummyObject) Size() int64                   { return int64(d.size) }
func (*dummyObject) SetSize(s int64)                 {}
func (*dummyObject) Reader() (io.ReadCloser, error)  { return nil, nil }
func (*dummyObject) Writer() (io.WriteCloser, error) { return nil, nil }
