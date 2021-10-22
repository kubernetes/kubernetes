package storage

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"io/ioutil"

	chk "gopkg.in/check.v1"
)

type AppendBlobSuite struct{}

var _ = chk.Suite(&AppendBlobSuite{})

func (s *AppendBlobSuite) TestPutAppendBlob(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.PutAppendBlob(nil), chk.IsNil)

	// Verify
	err := b.GetProperties(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(b.Properties.ContentLength, chk.Equals, int64(0))
	c.Assert(b.Properties.BlobType, chk.Equals, BlobTypeAppend)
}

func (s *AppendBlobSuite) TestPutAppendBlobAppendBlocks(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.PutAppendBlob(nil), chk.IsNil)

	chunk1 := content(1024)
	chunk2 := content(512)

	// Append first block
	c.Assert(b.AppendBlock(chunk1, nil), chk.IsNil)

	// Verify contents
	options := GetBlobRangeOptions{
		Range: &BlobRange{
			Start: 0,
			End:   uint64(len(chunk1) - 1),
		},
	}
	out, err := b.GetRange(&options)
	c.Assert(err, chk.IsNil)
	defer out.Close()
	blobContents, err := ioutil.ReadAll(out)
	c.Assert(err, chk.IsNil)
	c.Assert(blobContents, chk.DeepEquals, chunk1)

	// Append second block
	op := AppendBlockOptions{
		ContentMD5: true,
	}
	c.Assert(b.AppendBlock(chunk2, &op), chk.IsNil)

	// Verify contents
	options.Range.End = uint64(len(chunk1) + len(chunk2) - 1)
	out, err = b.GetRange(&options)
	c.Assert(err, chk.IsNil)
	defer out.Close()
	blobContents, err = ioutil.ReadAll(out)
	c.Assert(err, chk.IsNil)
	c.Assert(blobContents, chk.DeepEquals, append(chunk1, chunk2...))
}

func (s *StorageBlobSuite) TestPutAppendBlobSpecialChars(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.PutAppendBlob(nil), chk.IsNil)

	// Verify metadata
	err := b.GetProperties(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(b.Properties.ContentLength, chk.Equals, int64(0))
	c.Assert(b.Properties.BlobType, chk.Equals, BlobTypeAppend)

	chunk1 := content(1024)
	chunk2 := content(512)

	// Append first block
	c.Assert(b.AppendBlock(chunk1, nil), chk.IsNil)

	// Verify contents
	options := GetBlobRangeOptions{
		Range: &BlobRange{
			Start: 0,
			End:   uint64(len(chunk1) - 1),
		},
	}
	out, err := b.GetRange(&options)
	c.Assert(err, chk.IsNil)
	defer out.Close()
	blobContents, err := ioutil.ReadAll(out)
	c.Assert(err, chk.IsNil)
	c.Assert(blobContents, chk.DeepEquals, chunk1)

	// Append second block
	c.Assert(b.AppendBlock(chunk2, nil), chk.IsNil)

	// Verify contents
	options.Range.End = uint64(len(chunk1) + len(chunk2) - 1)
	out, err = b.GetRange(&options)
	c.Assert(err, chk.IsNil)
	defer out.Close()
	blobContents, err = ioutil.ReadAll(out)
	c.Assert(err, chk.IsNil)
	c.Assert(blobContents, chk.DeepEquals, append(chunk1, chunk2...))
}
