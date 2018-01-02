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

import (
	"io/ioutil"
	"net/http"
	"testing"

	chk "gopkg.in/check.v1"
)

type CopyBlobSuite struct{}

var _ = chk.Suite(&CopyBlobSuite{})

func (s *CopyBlobSuite) TestBlobCopy(c *chk.C) {
	if testing.Short() {
		c.Skip("skipping blob copy in short mode, no SLA on async operation")
	}

	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	srcBlob := cnt.GetBlobReference(blobName(c, "src"))
	dstBlob := cnt.GetBlobReference(blobName(c, "dst"))
	body := content(1024)

	c.Assert(srcBlob.putSingleBlockBlob(body), chk.IsNil)
	defer srcBlob.Delete(nil)

	c.Assert(dstBlob.Copy(srcBlob.GetURL(), nil), chk.IsNil)
	defer dstBlob.Delete(nil)

	resp, err := dstBlob.Get(nil)
	c.Assert(err, chk.IsNil)

	b, err := ioutil.ReadAll(resp)
	defer resp.Close()
	c.Assert(err, chk.IsNil)
	c.Assert(b, chk.DeepEquals, body)
}

func (s *CopyBlobSuite) TestStartBlobCopy(c *chk.C) {
	if testing.Short() {
		c.Skip("skipping blob copy in short mode, no SLA on async operation")
	}

	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	srcBlob := cnt.GetBlobReference(blobName(c, "src"))
	dstBlob := cnt.GetBlobReference(blobName(c, "dst"))
	body := content(1024)

	c.Assert(srcBlob.putSingleBlockBlob(body), chk.IsNil)
	defer srcBlob.Delete(nil)

	// given we dont know when it will start, can we even test destination creation?
	// will just test that an error wasn't thrown for now.
	copyID, err := dstBlob.StartCopy(srcBlob.GetURL(), nil)
	c.Assert(copyID, chk.NotNil)
	c.Assert(err, chk.IsNil)
}

// Tests abort of blobcopy. Given the blobcopy is usually over before we can actually trigger an abort
// it is agreed that we perform a copy then try and perform an abort. It should result in a HTTP status of 409.
// So basically we're testing negative scenario (as good as we can do for now)
func (s *CopyBlobSuite) TestAbortBlobCopy(c *chk.C) {
	if testing.Short() {
		c.Skip("skipping blob copy in short mode, no SLA on async operation")
	}

	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	srcBlob := cnt.GetBlobReference(blobName(c, "src"))
	dstBlob := cnt.GetBlobReference(blobName(c, "dst"))
	body := content(1024)

	c.Assert(srcBlob.putSingleBlockBlob(body), chk.IsNil)
	defer srcBlob.Delete(nil)

	// given we dont know when it will start, can we even test destination creation?
	// will just test that an error wasn't thrown for now.
	copyID, err := dstBlob.StartCopy(srcBlob.GetURL(), nil)
	c.Assert(copyID, chk.NotNil)
	c.Assert(err, chk.IsNil)

	err = dstBlob.WaitForCopy(copyID)
	c.Assert(err, chk.IsNil)

	// abort abort abort, but we *know* its already completed.
	err = dstBlob.AbortCopy(copyID, nil)

	// abort should fail (over already)
	c.Assert(err.(AzureStorageServiceError).StatusCode, chk.Equals, http.StatusConflict)
}

func (s *CopyBlobSuite) TestIncrementalCopyBlobNoTimeout(c *chk.C) {
	if testing.Short() {
		c.Skip("skipping blob copy in short mode, no SLA on async operation")
	}

	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	options := CreateContainerOptions{
		Access: ContainerAccessTypeBlob,
	}
	c.Assert(cnt.Create(&options), chk.IsNil)
	defer cnt.Delete(nil)

	b := cnt.GetBlobReference(blobName(c, "src"))
	size := int64(10 * 1024 * 1024)
	b.Properties.ContentLength = size
	c.Assert(b.PutPageBlob(nil), chk.IsNil)

	snapshotTime, err := b.CreateSnapshot(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(snapshotTime, chk.NotNil)

	u := b.GetURL()
	destBlob := cnt.GetBlobReference(blobName(c, "dst"))
	copyID, err := destBlob.IncrementalCopyBlob(u, *snapshotTime, nil)
	c.Assert(copyID, chk.NotNil)
	c.Assert(err, chk.IsNil)
}

func (s *CopyBlobSuite) TestIncrementalCopyBlobWithTimeout(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	options := CreateContainerOptions{
		Access: ContainerAccessTypeBlob,
	}
	c.Assert(cnt.Create(&options), chk.IsNil)
	defer cnt.Delete(nil)

	b := cnt.GetBlobReference(blobName(c, "src"))
	size := int64(10 * 1024 * 1024)
	b.Properties.ContentLength = size
	c.Assert(b.PutPageBlob(nil), chk.IsNil)

	snapshotTime, err := b.CreateSnapshot(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(snapshotTime, chk.NotNil)

	u := b.GetURL()
	destBlob := cnt.GetBlobReference(blobName(c, "dst"))
	copyID, err := destBlob.IncrementalCopyBlob(u, *snapshotTime, &IncrementalCopyOptions{Timeout: 30})
	c.Assert(copyID, chk.NotNil)
	c.Assert(err, chk.IsNil)
}
