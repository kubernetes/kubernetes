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
	"encoding/base64"
	"io"
	"io/ioutil"

	chk "gopkg.in/check.v1"
)

type BlockBlobSuite struct{}

var _ = chk.Suite(&BlockBlobSuite{})

func (s *BlockBlobSuite) TestCreateBlockBlobFromReader(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	length := 8888
	data := content(length)
	err := b.CreateBlockBlobFromReader(bytes.NewReader(data), nil)
	c.Assert(err, chk.IsNil)
	c.Assert(b.Properties.ContentLength, chk.Equals, int64(length))

	resp, err := b.Get(nil)
	c.Assert(err, chk.IsNil)
	gotData, err := ioutil.ReadAll(resp)
	defer resp.Close()

	c.Assert(err, chk.IsNil)
	c.Assert(gotData, chk.DeepEquals, data)
}

func (s *BlockBlobSuite) TestPutBlock(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	chunk := content(1024)
	blockID := base64.StdEncoding.EncodeToString([]byte("lol"))
	c.Assert(b.PutBlock(blockID, chunk, nil), chk.IsNil)
}

func (s *BlockBlobSuite) TestGetBlockList_PutBlockList(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	chunk := content(1024)
	blockID := base64.StdEncoding.EncodeToString([]byte("lol"))

	// Put one block
	c.Assert(b.PutBlock(blockID, chunk, nil), chk.IsNil)
	defer b.Delete(nil)

	// Get committed blocks
	committed, err := b.GetBlockList(BlockListTypeCommitted, nil)
	c.Assert(err, chk.IsNil)

	if len(committed.CommittedBlocks) > 0 {
		c.Fatal("There are committed blocks")
	}

	// Get uncommitted blocks
	uncommitted, err := b.GetBlockList(BlockListTypeUncommitted, nil)
	c.Assert(err, chk.IsNil)

	c.Assert(len(uncommitted.UncommittedBlocks), chk.Equals, 1)
	// Commit block list
	c.Assert(b.PutBlockList([]Block{{blockID, BlockStatusUncommitted}}, nil), chk.IsNil)

	// Get all blocks
	all, err := b.GetBlockList(BlockListTypeAll, nil)
	c.Assert(err, chk.IsNil)
	c.Assert(len(all.CommittedBlocks), chk.Equals, 1)
	c.Assert(len(all.UncommittedBlocks), chk.Equals, 0)

	// Verify the block
	thatBlock := all.CommittedBlocks[0]
	c.Assert(thatBlock.Name, chk.Equals, blockID)
	c.Assert(thatBlock.Size, chk.Equals, int64(len(chunk)))
}

func (s *BlockBlobSuite) TestCreateBlockBlob(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.CreateBlockBlob(nil), chk.IsNil)

	// Verify
	blocks, err := b.GetBlockList(BlockListTypeAll, nil)
	c.Assert(err, chk.IsNil)
	c.Assert(len(blocks.CommittedBlocks), chk.Equals, 0)
	c.Assert(len(blocks.UncommittedBlocks), chk.Equals, 0)
}

func (s *BlockBlobSuite) TestPutEmptyBlockBlob(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	err := b.GetProperties(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(b.Properties.ContentLength, chk.Not(chk.Equals), 0)
}

func (s *BlockBlobSuite) TestPutBlockWithLengthUsingLimitReader(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	length := 512
	data := content(length)

	lr := io.LimitReader(bytes.NewReader(data), 256)
	c.Assert(b.PutBlockWithLength("0000", 256, lr, nil), chk.IsNil)
}
