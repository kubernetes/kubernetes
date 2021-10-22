package storage

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import chk "gopkg.in/check.v1"

type StorageDirSuite struct{}

var _ = chk.Suite(&StorageDirSuite{})

func (s *StorageDirSuite) TestListZeroDirsAndFiles(c *chk.C) {
	// create share
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	share := cli.GetShareReference(shareName(c))
	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)

	// list contents, should be empty
	root := share.GetRootDirectoryReference()
	resp, err := root.ListDirsAndFiles(ListDirsAndFilesParameters{})
	c.Assert(err, chk.IsNil)
	c.Assert(resp.Directories, chk.IsNil)
	c.Assert(resp.Files, chk.IsNil)
}

func (s *StorageDirSuite) TestListDirsAndFiles(c *chk.C) {
	// create share
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	share := cli.GetShareReference(shareName(c))
	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)

	// create a directory and a file
	root := share.GetRootDirectoryReference()
	dir := root.GetDirectoryReference("SomeDirectory")
	file := root.GetFileReference("lol.file")
	c.Assert(dir.Create(nil), chk.IsNil)
	c.Assert(file.Create(512, nil), chk.IsNil)

	// list contents
	resp, err := root.ListDirsAndFiles(ListDirsAndFilesParameters{})
	c.Assert(err, chk.IsNil)
	c.Assert(len(resp.Directories), chk.Equals, 1)
	c.Assert(len(resp.Files), chk.Equals, 1)
	c.Assert(resp.Directories[0].Name, chk.Equals, dir.Name)
	c.Assert(resp.Files[0].Name, chk.Equals, file.Name)

	// delete file
	del, err := file.DeleteIfExists(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(del, chk.Equals, true)

	ok, err := file.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)
}

func (s *StorageDirSuite) TestCreateDirectory(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	share := cli.GetShareReference(shareName(c))
	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)

	root := share.GetRootDirectoryReference()
	dir := root.GetDirectoryReference("dir")
	err := dir.Create(nil)
	c.Assert(err, chk.IsNil)

	// check properties
	c.Assert(dir.Properties.Etag, chk.Not(chk.Equals), "")
	c.Assert(dir.Properties.LastModified, chk.Not(chk.Equals), "")

	// delete directory and verify
	c.Assert(dir.Delete(nil), chk.IsNil)
	exists, err := dir.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(exists, chk.Equals, false)
}

func (s *StorageDirSuite) TestCreateDirectoryIfNotExists(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// create share
	share := cli.GetShareReference(shareName(c))
	share.Create(nil)
	defer share.Delete(nil)

	// create non exisiting directory
	root := share.GetRootDirectoryReference()
	dir := root.GetDirectoryReference("dir")
	exists, err := dir.CreateIfNotExists(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(exists, chk.Equals, true)

	c.Assert(dir.Properties.Etag, chk.Not(chk.Equals), "")
	c.Assert(dir.Properties.LastModified, chk.Not(chk.Equals), "")

	c.Assert(dir.Delete(nil), chk.IsNil)
	exists, err = dir.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(exists, chk.Equals, false)
}

func (s *StorageDirSuite) TestCreateDirectoryIfExists(c *chk.C) {
	// create share
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	share := cli.GetShareReference(shareName(c))
	share.Create(nil)
	defer share.Delete(nil)

	// create directory
	root := share.GetRootDirectoryReference()
	dir := root.GetDirectoryReference("dir")
	dir.Create(nil)

	// try to create directory
	exists, err := dir.CreateIfNotExists(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(exists, chk.Equals, false)

	// check properties
	c.Assert(dir.Properties.Etag, chk.Not(chk.Equals), "")
	c.Assert(dir.Properties.LastModified, chk.Not(chk.Equals), "")

	// delete directory
	c.Assert(dir.Delete(nil), chk.IsNil)
}

func (s *StorageDirSuite) TestDirectoryMetadata(c *chk.C) {
	// create share
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	share := cli.GetShareReference(shareName(c))
	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)
	root := share.GetRootDirectoryReference()

	dir := root.GetDirectoryReference("testdir")
	c.Assert(dir.Create(nil), chk.IsNil)

	// get metadata, shouldn't be any
	c.Assert(dir.Metadata, chk.IsNil)

	// set some custom metadata
	md := map[string]string{
		"something": "somethingvalue",
		"another":   "anothervalue",
	}
	dir.Metadata = md
	c.Assert(dir.SetMetadata(nil), chk.IsNil)

	// retrieve and verify
	c.Assert(dir.FetchAttributes(nil), chk.IsNil)
	c.Assert(dir.Metadata, chk.DeepEquals, md)
}
