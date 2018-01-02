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
	"bytes"
	"io/ioutil"

	chk "gopkg.in/check.v1"
)

type PageBlobSuite struct{}

var _ = chk.Suite(&PageBlobSuite{})

func (s *PageBlobSuite) TestPutPageBlob(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	size := int64(10 * 1024 * 1024)
	b.Properties.ContentLength = size
	c.Assert(b.PutPageBlob(nil), chk.IsNil)

	// Verify
	err := b.GetProperties(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(b.Properties.ContentLength, chk.Equals, size)
	c.Assert(b.Properties.BlobType, chk.Equals, BlobTypePage)
}

func (s *PageBlobSuite) TestPutPagesUpdate(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	size := int64(10 * 1024 * 1024) // larger than we'll use
	b.Properties.ContentLength = size
	c.Assert(b.PutPageBlob(nil), chk.IsNil)

	chunk1 := content(1024)
	chunk2 := content(512)

	// Append chunks
	blobRange := BlobRange{
		End: uint64(len(chunk1) - 1),
	}
	c.Assert(b.WriteRange(blobRange, bytes.NewReader(chunk1), nil), chk.IsNil)
	blobRange.Start = uint64(len(chunk1))
	blobRange.End = uint64(len(chunk1) + len(chunk2) - 1)
	c.Assert(b.WriteRange(blobRange, bytes.NewReader(chunk2), nil), chk.IsNil)

	// Verify contents
	options := GetBlobRangeOptions{
		Range: &BlobRange{
			End: uint64(len(chunk1) + len(chunk2) - 1),
		},
	}
	out, err := b.GetRange(&options)
	c.Assert(err, chk.IsNil)
	defer out.Close()
	blobContents, err := ioutil.ReadAll(out)
	c.Assert(err, chk.IsNil)
	c.Assert(blobContents, chk.DeepEquals, append(chunk1, chunk2...))

	// Overwrite first half of chunk1
	chunk0 := content(512)
	blobRange.Start = 0
	blobRange.End = uint64(len(chunk0) - 1)
	c.Assert(b.WriteRange(blobRange, bytes.NewReader(chunk0), nil), chk.IsNil)

	// Verify contents
	out, err = b.GetRange(&options)
	c.Assert(err, chk.IsNil)
	defer out.Close()
	blobContents, err = ioutil.ReadAll(out)
	c.Assert(err, chk.IsNil)
	c.Assert(blobContents, chk.DeepEquals, append(append(chunk0, chunk1[512:]...), chunk2...))
}

func (s *PageBlobSuite) TestPutPagesClear(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	size := int64(10 * 1024 * 1024) // larger than we'll use
	b.Properties.ContentLength = size
	c.Assert(b.PutPageBlob(nil), chk.IsNil)

	// Put 0-2047
	chunk := content(2048)
	blobRange := BlobRange{
		End: 2047,
	}
	c.Assert(b.WriteRange(blobRange, bytes.NewReader(chunk), nil), chk.IsNil)

	// Clear 512-1023
	blobRange.Start = 512
	blobRange.End = 1023
	c.Assert(b.ClearRange(blobRange, nil), chk.IsNil)

	// Verify contents
	options := GetBlobRangeOptions{
		Range: &BlobRange{
			Start: 0,
			End:   2047,
		},
	}
	out, err := b.GetRange(&options)
	c.Assert(err, chk.IsNil)
	contents, err := ioutil.ReadAll(out)
	c.Assert(err, chk.IsNil)
	defer out.Close()
	c.Assert(contents, chk.DeepEquals, append(append(chunk[:512], make([]byte, 512)...), chunk[1024:]...))
}

func (s *PageBlobSuite) TestGetPageRanges(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	size := int64(10 * 1024) // larger than we'll use

	// Get page ranges on empty blob
	blob1 := cnt.GetBlobReference(blobName(c, "1"))
	blob1.Properties.ContentLength = size
	c.Assert(blob1.PutPageBlob(nil), chk.IsNil)
	out, err := blob1.GetPageRanges(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(len(out.PageList), chk.Equals, 0)

	// Get page ranges with just one range
	blob2 := cnt.GetBlobReference(blobName(c, "2"))
	blob2.Properties.ContentLength = size
	c.Assert(blob2.PutPageBlob(nil), chk.IsNil)
	blobRange := []BlobRange{
		{End: 511},
		{Start: 1024, End: 2047},
	}
	c.Assert(blob2.WriteRange(blobRange[0], bytes.NewReader(content(512)), nil), chk.IsNil)

	out, err = blob2.GetPageRanges(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(len(out.PageList), chk.Equals, 1)
	expected := []PageRange{
		{End: 511},
		{Start: 1024, End: 2047},
	}
	c.Assert(out.PageList[0], chk.Equals, expected[0])

	// Get page ranges with just two range
	blob3 := cnt.GetBlobReference(blobName(c, "3"))
	blob3.Properties.ContentLength = size
	c.Assert(blob3.PutPageBlob(nil), chk.IsNil)
	for _, br := range blobRange {
		c.Assert(blob3.WriteRange(br, bytes.NewReader(content(int(br.End-br.Start+1))), nil), chk.IsNil)
	}
	out, err = blob3.GetPageRanges(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(len(out.PageList), chk.Equals, 2)
	c.Assert(out.PageList, chk.DeepEquals, expected)
}
