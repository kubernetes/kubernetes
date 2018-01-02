package test

import (
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/ioutil"

	"gopkg.in/src-d/go-git.v4/config"
	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/format/index"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/storage"

	"github.com/src-d/go-git-fixtures"
	. "gopkg.in/check.v1"
)

type Storer interface {
	storer.EncodedObjectStorer
	storer.ReferenceStorer
	storer.ShallowStorer
	storer.IndexStorer
	config.ConfigStorer
	storage.ModuleStorer
}

type TestObject struct {
	Object plumbing.EncodedObject
	Hash   string
	Type   plumbing.ObjectType
}

type BaseStorageSuite struct {
	Storer Storer

	validTypes  []plumbing.ObjectType
	testObjects map[plumbing.ObjectType]TestObject
}

func NewBaseStorageSuite(s Storer) BaseStorageSuite {
	commit := &plumbing.MemoryObject{}
	commit.SetType(plumbing.CommitObject)
	tree := &plumbing.MemoryObject{}
	tree.SetType(plumbing.TreeObject)
	blob := &plumbing.MemoryObject{}
	blob.SetType(plumbing.BlobObject)
	tag := &plumbing.MemoryObject{}
	tag.SetType(plumbing.TagObject)

	return BaseStorageSuite{
		Storer: s,
		validTypes: []plumbing.ObjectType{
			plumbing.CommitObject,
			plumbing.BlobObject,
			plumbing.TagObject,
			plumbing.TreeObject,
		},
		testObjects: map[plumbing.ObjectType]TestObject{
			plumbing.CommitObject: {commit, "dcf5b16e76cce7425d0beaef62d79a7d10fce1f5", plumbing.CommitObject},
			plumbing.TreeObject:   {tree, "4b825dc642cb6eb9a060e54bf8d69288fbee4904", plumbing.TreeObject},
			plumbing.BlobObject:   {blob, "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391", plumbing.BlobObject},
			plumbing.TagObject:    {tag, "d994c6bb648123a17e8f70a966857c546b2a6f94", plumbing.TagObject},
		}}
}

func (s *BaseStorageSuite) SetUpTest(c *C) {
	c.Assert(fixtures.Init(), IsNil)
}

func (s *BaseStorageSuite) TearDownTest(c *C) {
	c.Assert(fixtures.Clean(), IsNil)
}

func (s *BaseStorageSuite) TestSetEncodedObjectAndEncodedObject(c *C) {
	for _, to := range s.testObjects {
		comment := Commentf("failed for type %s", to.Type.String())

		h, err := s.Storer.SetEncodedObject(to.Object)
		c.Assert(err, IsNil)
		c.Assert(h.String(), Equals, to.Hash, comment)

		o, err := s.Storer.EncodedObject(to.Type, h)
		c.Assert(err, IsNil)
		c.Assert(objectEquals(o, to.Object), IsNil)

		o, err = s.Storer.EncodedObject(plumbing.AnyObject, h)
		c.Assert(err, IsNil)
		c.Assert(objectEquals(o, to.Object), IsNil)

		for _, t := range s.validTypes {
			if t == to.Type {
				continue
			}

			o, err = s.Storer.EncodedObject(t, h)
			c.Assert(o, IsNil)
			c.Assert(err, Equals, plumbing.ErrObjectNotFound)
		}
	}
}

func (s *BaseStorageSuite) TestSetEncodedObjectInvalid(c *C) {
	o := s.Storer.NewEncodedObject()
	o.SetType(plumbing.REFDeltaObject)

	_, err := s.Storer.SetEncodedObject(o)
	c.Assert(err, NotNil)
}

func (s *BaseStorageSuite) TestIterEncodedObjects(c *C) {
	for _, o := range s.testObjects {
		h, err := s.Storer.SetEncodedObject(o.Object)
		c.Assert(err, IsNil)
		c.Assert(h, Equals, o.Object.Hash())
	}

	for _, t := range s.validTypes {
		comment := Commentf("failed for type %s)", t.String())
		i, err := s.Storer.IterEncodedObjects(t)
		c.Assert(err, IsNil, comment)

		o, err := i.Next()
		c.Assert(err, IsNil)
		c.Assert(objectEquals(o, s.testObjects[t].Object), IsNil)

		o, err = i.Next()
		c.Assert(o, IsNil)
		c.Assert(err, Equals, io.EOF, comment)
	}

	i, err := s.Storer.IterEncodedObjects(plumbing.AnyObject)
	c.Assert(err, IsNil)

	foundObjects := []plumbing.EncodedObject{}
	i.ForEach(func(o plumbing.EncodedObject) error {
		foundObjects = append(foundObjects, o)
		return nil
	})

	c.Assert(foundObjects, HasLen, len(s.testObjects))
	for _, to := range s.testObjects {
		found := false
		for _, o := range foundObjects {
			if to.Object.Hash() == o.Hash() {
				found = true
				break
			}
		}
		c.Assert(found, Equals, true, Commentf("Object of type %s not found", to.Type.String()))
	}
}

func (s *BaseStorageSuite) TestPackfileWriter(c *C) {
	pwr, ok := s.Storer.(storer.PackfileWriter)
	if !ok {
		c.Skip("not a storer.PackWriter")
	}

	pw, err := pwr.PackfileWriter()
	c.Assert(err, IsNil)

	f := fixtures.Basic().One()
	_, err = io.Copy(pw, f.Packfile())
	c.Assert(err, IsNil)

	err = pw.Close()
	c.Assert(err, IsNil)

	iter, err := s.Storer.IterEncodedObjects(plumbing.AnyObject)
	c.Assert(err, IsNil)
	objects := 0
	err = iter.ForEach(func(plumbing.EncodedObject) error {
		objects++
		return nil
	})
	c.Assert(err, IsNil)
	c.Assert(objects, Equals, 31)
}

func (s *BaseStorageSuite) TestObjectStorerTxSetEncodedObjectAndCommit(c *C) {
	storer, ok := s.Storer.(storer.Transactioner)
	if !ok {
		c.Skip("not a plumbing.ObjectStorerTx")
	}

	tx := storer.Begin()
	for _, o := range s.testObjects {
		h, err := tx.SetEncodedObject(o.Object)
		c.Assert(err, IsNil)
		c.Assert(h.String(), Equals, o.Hash)
	}

	iter, err := s.Storer.IterEncodedObjects(plumbing.AnyObject)
	c.Assert(err, IsNil)
	_, err = iter.Next()
	c.Assert(err, Equals, io.EOF)

	err = tx.Commit()
	c.Assert(err, IsNil)

	iter, err = s.Storer.IterEncodedObjects(plumbing.AnyObject)
	c.Assert(err, IsNil)

	var count int
	iter.ForEach(func(o plumbing.EncodedObject) error {
		count++
		return nil
	})

	c.Assert(count, Equals, 4)
}

func (s *BaseStorageSuite) TestObjectStorerTxSetObjectAndGetObject(c *C) {
	storer, ok := s.Storer.(storer.Transactioner)
	if !ok {
		c.Skip("not a plumbing.ObjectStorerTx")
	}

	tx := storer.Begin()
	for _, expected := range s.testObjects {
		h, err := tx.SetEncodedObject(expected.Object)
		c.Assert(err, IsNil)
		c.Assert(h.String(), Equals, expected.Hash)

		o, err := tx.EncodedObject(expected.Type, plumbing.NewHash(expected.Hash))
		c.Assert(err, IsNil)
		c.Assert(o.Hash().String(), DeepEquals, expected.Hash)
	}
}

func (s *BaseStorageSuite) TestObjectStorerTxGetObjectNotFound(c *C) {
	storer, ok := s.Storer.(storer.Transactioner)
	if !ok {
		c.Skip("not a plumbing.ObjectStorerTx")
	}

	tx := storer.Begin()
	o, err := tx.EncodedObject(plumbing.AnyObject, plumbing.ZeroHash)
	c.Assert(o, IsNil)
	c.Assert(err, Equals, plumbing.ErrObjectNotFound)
}

func (s *BaseStorageSuite) TestObjectStorerTxSetObjectAndRollback(c *C) {
	storer, ok := s.Storer.(storer.Transactioner)
	if !ok {
		c.Skip("not a plumbing.ObjectStorerTx")
	}

	tx := storer.Begin()
	for _, o := range s.testObjects {
		h, err := tx.SetEncodedObject(o.Object)
		c.Assert(err, IsNil)
		c.Assert(h.String(), Equals, o.Hash)
	}

	err := tx.Rollback()
	c.Assert(err, IsNil)

	iter, err := s.Storer.IterEncodedObjects(plumbing.AnyObject)
	c.Assert(err, IsNil)
	_, err = iter.Next()
	c.Assert(err, Equals, io.EOF)
}

func (s *BaseStorageSuite) TestSetReferenceAndGetReference(c *C) {
	err := s.Storer.SetReference(
		plumbing.NewReferenceFromStrings("foo", "bc9968d75e48de59f0870ffb71f5e160bbbdcf52"),
	)
	c.Assert(err, IsNil)

	err = s.Storer.SetReference(
		plumbing.NewReferenceFromStrings("bar", "482e0eada5de4039e6f216b45b3c9b683b83bfa"),
	)
	c.Assert(err, IsNil)

	e, err := s.Storer.Reference(plumbing.ReferenceName("foo"))
	c.Assert(err, IsNil)
	c.Assert(e.Hash().String(), Equals, "bc9968d75e48de59f0870ffb71f5e160bbbdcf52")
}

func (s *BaseStorageSuite) TestRemoveReference(c *C) {
	err := s.Storer.SetReference(
		plumbing.NewReferenceFromStrings("foo", "bc9968d75e48de59f0870ffb71f5e160bbbdcf52"),
	)
	c.Assert(err, IsNil)

	err = s.Storer.RemoveReference(plumbing.ReferenceName("foo"))
	c.Assert(err, IsNil)

	_, err = s.Storer.Reference(plumbing.ReferenceName("foo"))
	c.Assert(err, Equals, plumbing.ErrReferenceNotFound)
}

func (s *BaseStorageSuite) TestRemoveReferenceNonExistent(c *C) {
	err := s.Storer.SetReference(
		plumbing.NewReferenceFromStrings("foo", "bc9968d75e48de59f0870ffb71f5e160bbbdcf52"),
	)
	c.Assert(err, IsNil)

	err = s.Storer.RemoveReference(plumbing.ReferenceName("nonexistent"))
	c.Assert(err, IsNil)

	e, err := s.Storer.Reference(plumbing.ReferenceName("foo"))
	c.Assert(err, IsNil)
	c.Assert(e.Hash().String(), Equals, "bc9968d75e48de59f0870ffb71f5e160bbbdcf52")
}

func (s *BaseStorageSuite) TestGetReferenceNotFound(c *C) {
	r, err := s.Storer.Reference(plumbing.ReferenceName("bar"))
	c.Assert(err, Equals, plumbing.ErrReferenceNotFound)
	c.Assert(r, IsNil)
}

func (s *BaseStorageSuite) TestIterReferences(c *C) {
	err := s.Storer.SetReference(
		plumbing.NewReferenceFromStrings("refs/foo", "bc9968d75e48de59f0870ffb71f5e160bbbdcf52"),
	)
	c.Assert(err, IsNil)

	i, err := s.Storer.IterReferences()
	c.Assert(err, IsNil)

	e, err := i.Next()
	c.Assert(err, IsNil)
	c.Assert(e.Hash().String(), Equals, "bc9968d75e48de59f0870ffb71f5e160bbbdcf52")

	e, err = i.Next()
	c.Assert(e, IsNil)
	c.Assert(err, Equals, io.EOF)
}

func (s *BaseStorageSuite) TestSetShallowAndShallow(c *C) {
	expected := []plumbing.Hash{
		plumbing.NewHash("b66c08ba28aa1f81eb06a1127aa3936ff77e5e2c"),
		plumbing.NewHash("c3f4688a08fd86f1bf8e055724c84b7a40a09733"),
		plumbing.NewHash("c78874f116be67ecf54df225a613162b84cc6ebf"),
	}

	err := s.Storer.SetShallow(expected)
	c.Assert(err, IsNil)

	result, err := s.Storer.Shallow()
	c.Assert(err, IsNil)
	c.Assert(result, DeepEquals, expected)
}

func (s *BaseStorageSuite) TestSetConfigAndConfig(c *C) {
	expected := config.NewConfig()
	expected.Core.IsBare = true
	expected.Remotes["foo"] = &config.RemoteConfig{
		Name: "foo",
		URLs: []string{"http://foo/bar.git"},
	}

	err := s.Storer.SetConfig(expected)
	c.Assert(err, IsNil)

	cfg, err := s.Storer.Config()
	c.Assert(err, IsNil)

	c.Assert(cfg.Core.IsBare, DeepEquals, expected.Core.IsBare)
	c.Assert(cfg.Remotes, DeepEquals, expected.Remotes)
}

func (s *BaseStorageSuite) TestIndex(c *C) {
	expected := &index.Index{}
	expected.Version = 2

	idx, err := s.Storer.Index()
	c.Assert(err, IsNil)
	c.Assert(idx, DeepEquals, expected)
}

func (s *BaseStorageSuite) TestSetIndexAndIndex(c *C) {
	expected := &index.Index{}
	expected.Version = 2

	err := s.Storer.SetIndex(expected)
	c.Assert(err, IsNil)

	idx, err := s.Storer.Index()
	c.Assert(err, IsNil)
	c.Assert(idx, DeepEquals, expected)
}

func (s *BaseStorageSuite) TestSetConfigInvalid(c *C) {
	cfg := config.NewConfig()
	cfg.Remotes["foo"] = &config.RemoteConfig{}

	err := s.Storer.SetConfig(cfg)
	c.Assert(err, NotNil)
}

func (s *BaseStorageSuite) TestModule(c *C) {
	storer, err := s.Storer.Module("foo")
	c.Assert(err, IsNil)
	c.Assert(storer, NotNil)

	storer, err = s.Storer.Module("foo")
	c.Assert(err, IsNil)
	c.Assert(storer, NotNil)
}

func (s *BaseStorageSuite) TestDeltaObjectStorer(c *C) {
	dos, ok := s.Storer.(storer.DeltaObjectStorer)
	if !ok {
		c.Skip("not an DeltaObjectStorer")
	}

	pwr, ok := s.Storer.(storer.PackfileWriter)
	if !ok {
		c.Skip("not a storer.PackWriter")
	}

	pw, err := pwr.PackfileWriter()
	c.Assert(err, IsNil)

	f := fixtures.Basic().One()
	_, err = io.Copy(pw, f.Packfile())
	c.Assert(err, IsNil)

	err = pw.Close()
	c.Assert(err, IsNil)

	h := plumbing.NewHash("32858aad3c383ed1ff0a0f9bdf231d54a00c9e88")
	obj, err := dos.DeltaObject(plumbing.AnyObject, h)
	c.Assert(err, IsNil)
	c.Assert(obj.Type(), Equals, plumbing.BlobObject)

	h = plumbing.NewHash("aa9b383c260e1d05fbbf6b30a02914555e20c725")
	obj, err = dos.DeltaObject(plumbing.AnyObject, h)
	c.Assert(err, IsNil)
	c.Assert(obj.Type(), Equals, plumbing.OFSDeltaObject)
	_, ok = obj.(plumbing.DeltaObject)
	c.Assert(ok, Equals, true)
}

func objectEquals(a plumbing.EncodedObject, b plumbing.EncodedObject) error {
	ha := a.Hash()
	hb := b.Hash()
	if ha != hb {
		return fmt.Errorf("hashes do not match: %s != %s",
			ha.String(), hb.String())
	}

	ra, err := a.Reader()
	if err != nil {
		return fmt.Errorf("can't get reader on b: %q", err)
	}

	rb, err := b.Reader()
	if err != nil {
		return fmt.Errorf("can't get reader on a: %q", err)
	}

	ca, err := ioutil.ReadAll(ra)
	if err != nil {
		return fmt.Errorf("error reading a: %q", err)
	}

	cb, err := ioutil.ReadAll(rb)
	if err != nil {
		return fmt.Errorf("error reading b: %q", err)
	}

	if hex.EncodeToString(ca) != hex.EncodeToString(cb) {
		return errors.New("content does not match")
	}

	return nil
}
