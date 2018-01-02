package storage

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import chk "gopkg.in/check.v1"

type StorageShareSuite struct{}

var _ = chk.Suite(&StorageShareSuite{})

func getFileClient(c *chk.C) FileServiceClient {
	return getBasicClient(c).GetFileService()
}

func (s *StorageShareSuite) TestCreateShareDeleteShare(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	share := cli.GetShareReference(shareName(c))
	c.Assert(share.Create(nil), chk.IsNil)
	c.Assert(share.Delete(nil), chk.IsNil)
}

func (s *StorageShareSuite) TestCreateShareIfNotExists(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// Create non existing
	share := cli.GetShareReference(shareName(c, "notexists"))
	ok, err := share.CreateIfNotExists(nil)
	defer share.Delete(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)

}

func (s *StorageShareSuite) TestCreateShareIfExists(c *chk.C) {
	cli := getFileClient(c)
	share := cli.GetShareReference(shareName(c, "exists"))
	share.Create(nil)
	defer share.Delete(nil)

	rec := cli.client.appendRecorder(c)
	share.fsc = &cli
	defer rec.Stop()

	// Try to create exisiting
	ok, err := share.CreateIfNotExists(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)
}

func (s *StorageShareSuite) TestDeleteShareIfNotExists(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// delete non-existing share
	share1 := cli.GetShareReference(shareName(c, "1"))
	ok, err := share1.DeleteIfExists(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)

	// delete existing share
	share2 := cli.GetShareReference(shareName(c, "2"))
	c.Assert(share2.Create(nil), chk.IsNil)
	ok, err = share2.DeleteIfExists(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)
}

func (s *StorageShareSuite) TestListShares(c *chk.C) {
	cli := getFileClient(c)
	cli.deleteAllShares()
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	name := shareName(c)
	share := cli.GetShareReference(name)

	c.Assert(share.Create(nil), chk.IsNil)

	resp, err := cli.ListShares(ListSharesParameters{
		MaxResults: 5,
	})
	c.Assert(err, chk.IsNil)

	c.Check(len(resp.Shares), chk.Equals, 1)
	c.Check(resp.Shares[0].Name, chk.Equals, name)

	// clean up via the retrieved share object
	resp.Shares[0].Delete(nil)
}

func (s *StorageShareSuite) TestShareExists(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// Share does not exist
	share1 := cli.GetShareReference(shareName(c, "1"))
	ok, err := share1.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)

	// Share exists
	share2 := cli.GetShareReference(shareName(c, "2"))
	c.Assert(share2.Create(nil), chk.IsNil)
	defer share1.Delete(nil)
	ok, err = share2.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)
}

func (s *StorageShareSuite) TestGetAndSetShareProperties(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()
	share := cli.GetShareReference(shareName(c))
	quota := 55

	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)
	c.Assert(share.Properties.LastModified, chk.Not(chk.Equals), "")

	share.Properties.Quota = quota
	err := share.SetProperties(nil)
	c.Assert(err, chk.IsNil)

	err = share.FetchAttributes(nil)
	c.Assert(err, chk.IsNil)

	c.Assert(share.Properties.Quota, chk.Equals, quota)
}

func (s *StorageShareSuite) TestGetAndSetShareMetadata(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()
	share1 := cli.GetShareReference(shareName(c, "1"))

	c.Assert(share1.Create(nil), chk.IsNil)
	defer share1.Delete(nil)

	// by default there should be no metadata
	c.Assert(share1.Metadata, chk.IsNil)
	c.Assert(share1.FetchAttributes(nil), chk.IsNil)
	c.Assert(share1.Metadata, chk.IsNil)

	share2 := cli.GetShareReference(shareName(c, "2"))
	c.Assert(share2.Create(nil), chk.IsNil)
	defer share2.Delete(nil)

	c.Assert(share2.Metadata, chk.IsNil)

	mPut := map[string]string{
		"lol":      "rofl",
		"rofl_baz": "waz qux",
	}

	share2.Metadata = mPut
	c.Assert(share2.SetMetadata(nil), chk.IsNil)
	c.Check(share2.Metadata, chk.DeepEquals, mPut)

	c.Assert(share2.FetchAttributes(nil), chk.IsNil)
	c.Check(share2.Metadata, chk.DeepEquals, mPut)
}

func (s *StorageShareSuite) TestMetadataCaseMunging(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()
	share := cli.GetShareReference(shareName(c))

	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)

	mPutUpper := map[string]string{
		"Lol":      "different rofl",
		"rofl_BAZ": "different waz qux",
	}
	mExpectLower := map[string]string{
		"lol":      "different rofl",
		"rofl_baz": "different waz qux",
	}

	share.Metadata = mPutUpper
	c.Assert(share.SetMetadata(nil), chk.IsNil)

	c.Check(share.Metadata, chk.DeepEquals, mPutUpper)
	c.Assert(share.FetchAttributes(nil), chk.IsNil)
	c.Check(share.Metadata, chk.DeepEquals, mExpectLower)
}

func (cli *FileServiceClient) deleteAllShares() {
	resp, _ := cli.ListShares(ListSharesParameters{})
	if resp != nil && len(resp.Shares) > 0 {
		for _, sh := range resp.Shares {
			share := cli.GetShareReference(sh.Name)
			share.Delete(nil)
		}
	}
}

func shareName(c *chk.C, extras ...string) string {
	return nameGenerator(63, "share-", alphanum, c, extras)
}
