package filesystem

import (
	"io/ioutil"
	"testing"

	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/storage/test"

	. "gopkg.in/check.v1"
	"gopkg.in/src-d/go-billy.v3/memfs"
	"gopkg.in/src-d/go-billy.v3/osfs"
)

func Test(t *testing.T) { TestingT(t) }

type StorageSuite struct {
	test.BaseStorageSuite
	dir string
}

var _ = Suite(&StorageSuite{})

func (s *StorageSuite) SetUpTest(c *C) {
	s.dir = c.MkDir()
	storage, err := NewStorage(osfs.New(s.dir))
	c.Assert(err, IsNil)

	// ensure that right interfaces are implemented
	var _ storer.EncodedObjectStorer = storage
	var _ storer.IndexStorer = storage
	var _ storer.ReferenceStorer = storage
	var _ storer.ShallowStorer = storage
	var _ storer.DeltaObjectStorer = storage
	var _ storer.PackfileWriter = storage

	s.BaseStorageSuite = test.NewBaseStorageSuite(storage)
	s.BaseStorageSuite.SetUpTest(c)
}

func (s *StorageSuite) TestFilesystem(c *C) {
	fs := memfs.New()
	storage, err := NewStorage(fs)
	c.Assert(err, IsNil)

	c.Assert(storage.Filesystem(), Equals, fs)
}

func (s *StorageSuite) TestNewStorageShouldNotAddAnyContentsToDir(c *C) {
	fis, err := ioutil.ReadDir(s.dir)
	c.Assert(err, IsNil)
	c.Assert(fis, HasLen, 0)
}
