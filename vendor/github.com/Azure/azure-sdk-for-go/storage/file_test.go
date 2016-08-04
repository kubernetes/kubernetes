package storage

import (
	chk "gopkg.in/check.v1"
)

type StorageFileSuite struct{}

var _ = chk.Suite(&StorageFileSuite{})

func getFileClient(c *chk.C) FileServiceClient {
	return getBasicClient(c).GetFileService()
}

func (s *StorageFileSuite) Test_pathForFileShare(c *chk.C) {
	c.Assert(pathForFileShare("foo"), chk.Equals, "/foo")
}

func (s *StorageFileSuite) TestCreateShareDeleteShare(c *chk.C) {
	cli := getFileClient(c)
	name := randShare()
	c.Assert(cli.CreateShare(name), chk.IsNil)
	c.Assert(cli.DeleteShare(name), chk.IsNil)
}

func (s *StorageFileSuite) TestCreateShareIfNotExists(c *chk.C) {
	cli := getFileClient(c)
	name := randShare()
	defer cli.DeleteShare(name)

	// First create
	ok, err := cli.CreateShareIfNotExists(name)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)

	// Second create, should not give errors
	ok, err = cli.CreateShareIfNotExists(name)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)
}

func (s *StorageFileSuite) TestDeleteShareIfNotExists(c *chk.C) {
	cli := getFileClient(c)
	name := randShare()

	// delete non-existing share
	ok, err := cli.DeleteShareIfExists(name)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)

	c.Assert(cli.CreateShare(name), chk.IsNil)

	// delete existing share
	ok, err = cli.DeleteShareIfExists(name)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)
}

func (s *StorageFileSuite) Test_checkForStorageEmulator(c *chk.C) {
	f := getEmulatorClient(c).GetFileService()
	err := f.checkForStorageEmulator()
	c.Assert(err, chk.NotNil)
}

const testSharePrefix = "zzzzztest"

func randShare() string {
	return testSharePrefix + randString(32-len(testSharePrefix))
}
