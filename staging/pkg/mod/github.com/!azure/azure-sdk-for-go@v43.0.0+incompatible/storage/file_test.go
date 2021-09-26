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
	"crypto/md5"
	"encoding/base64"
	"io"

	chk "gopkg.in/check.v1"
)

type StorageFileSuite struct{}

var _ = chk.Suite(&StorageFileSuite{})

func (s *StorageFileSuite) TestCreateFile(c *chk.C) {
	cli := getFileClient(c)
	cli.deleteAllShares()
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// create share
	share := cli.GetShareReference(shareName(c))
	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)
	root := share.GetRootDirectoryReference()

	// create directory structure
	dir1 := root.GetDirectoryReference("one")
	c.Assert(dir1.Create(nil), chk.IsNil)
	dir2 := dir1.GetDirectoryReference("two")
	c.Assert(dir2.Create(nil), chk.IsNil)

	// verify file doesn't exist
	file := dir2.GetFileReference("some.file")
	exists, err := file.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(exists, chk.Equals, false)

	// create file
	c.Assert(file.Create(1024, nil), chk.IsNil)

	// delete file and verify
	c.Assert(file.Delete(nil), chk.IsNil)
	exists, err = file.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(exists, chk.Equals, false)
}

func (s *StorageFileSuite) TestGetFile(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// create share
	share := cli.GetShareReference(shareName(c))
	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)
	root := share.GetRootDirectoryReference()

	// create file
	const size = uint64(1024)
	byteStream, _ := newByteStream(size)
	file := root.GetFileReference("some.file")
	c.Assert(file.Create(size, nil), chk.IsNil)

	// fill file with some data
	c.Assert(file.WriteRange(byteStream, FileRange{End: size - 1}, nil), chk.IsNil)

	// set some metadata
	md := map[string]string{
		"something": "somethingvalue",
		"another":   "anothervalue",
	}
	file.Metadata = md
	c.Assert(file.SetMetadata(nil), chk.IsNil)

	options := GetFileOptions{
		GetContentMD5: false,
	}
	// retrieve full file content and verify
	stream, err := file.DownloadRangeToStream(FileRange{Start: 0, End: size - 1}, &options)
	c.Assert(err, chk.IsNil)
	defer stream.Body.Close()
	var b1 [size]byte
	count, _ := stream.Body.Read(b1[:])
	c.Assert(count, chk.Equals, int(size))
	var c1 [size]byte
	bs, _ := newByteStream(size)
	bs.Read(c1[:])
	c.Assert(b1, chk.DeepEquals, c1)

	// retrieve partial file content and verify
	stream, err = file.DownloadRangeToStream(FileRange{Start: size / 2, End: size - 1}, &options)
	c.Assert(err, chk.IsNil)
	defer stream.Body.Close()
	var b2 [size / 2]byte
	count, _ = stream.Body.Read(b2[:])
	c.Assert(count, chk.Equals, int(size)/2)
	var c2 [size / 2]byte
	bs, _ = newByteStream(size / 2)
	bs.Read(c2[:])
	c.Assert(b2, chk.DeepEquals, c2)
}

func (s *StorageFileSuite) TestFileRanges(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	share := cli.GetShareReference(shareName(c))
	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)
	root := share.GetRootDirectoryReference()

	fileSize := uint64(4096)
	contentBytes := content(int(fileSize))

	// --- File with no valid ranges
	file1 := root.GetFileReference("file1.txt")
	c.Assert(file1.Create(fileSize, nil), chk.IsNil)

	ranges, err := file1.ListRanges(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(ranges.ContentLength, chk.Equals, fileSize)
	c.Assert(ranges.FileRanges, chk.IsNil)

	// --- File after writing a range
	file2 := root.GetFileReference("file2.txt")
	c.Assert(file2.Create(fileSize, nil), chk.IsNil)
	c.Assert(file2.WriteRange(bytes.NewReader(contentBytes), FileRange{End: fileSize - 1}, nil), chk.IsNil)

	ranges, err = file2.ListRanges(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(len(ranges.FileRanges), chk.Equals, 1)
	c.Assert((ranges.FileRanges[0].End-ranges.FileRanges[0].Start)+1, chk.Equals, fileSize)

	// --- File after writing and clearing
	file3 := root.GetFileReference("file3.txt")
	c.Assert(file3.Create(fileSize, nil), chk.IsNil)
	c.Assert(file3.WriteRange(bytes.NewReader(contentBytes), FileRange{End: fileSize - 1}, nil), chk.IsNil)
	c.Assert(file3.ClearRange(FileRange{End: fileSize - 1}, nil), chk.IsNil)

	ranges, err = file3.ListRanges(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(ranges.FileRanges, chk.IsNil)

	// --- File with ranges and subranges
	file4 := root.GetFileReference("file4.txt")
	c.Assert(file4.Create(fileSize, nil), chk.IsNil)
	putRanges := []FileRange{
		{End: 511},
		{Start: 1024, End: 1535},
		{Start: 2048, End: 2559},
		{Start: 3072, End: 3583},
	}

	for _, r := range putRanges {
		err = file4.WriteRange(bytes.NewReader(contentBytes[:512]), r, nil)
		c.Assert(err, chk.IsNil)
	}

	// validate all ranges
	ranges, err = file4.ListRanges(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(ranges.FileRanges, chk.DeepEquals, putRanges)

	options := ListRangesOptions{
		ListRange: &FileRange{
			Start: 1000,
			End:   3000,
		},
	}
	// validate sub-ranges
	ranges, err = file4.ListRanges(&options)
	c.Assert(err, chk.IsNil)
	c.Assert(ranges.FileRanges, chk.DeepEquals, putRanges[1:3])

	// --- clear partial range and validate
	file5 := root.GetFileReference("file5.txt")
	c.Assert(file5.Create(fileSize, nil), chk.IsNil)
	c.Assert(file5.WriteRange(bytes.NewReader(contentBytes), FileRange{End: fileSize - 1}, nil), chk.IsNil)
	c.Assert(file5.ClearRange(putRanges[0], nil), chk.IsNil)
	c.Assert(file5.ClearRange(putRanges[2], nil), chk.IsNil)

	ranges, err = file5.ListRanges(nil)
	c.Assert(err, chk.IsNil)
	expectedtRanges := []FileRange{
		{Start: 512, End: 2047},
		{Start: 2560, End: 4095},
	}
	c.Assert(ranges.FileRanges, chk.HasLen, 2)
	c.Assert(ranges.FileRanges[0], chk.DeepEquals, expectedtRanges[0])
	c.Assert(ranges.FileRanges[1], chk.DeepEquals, expectedtRanges[1])
}

func (s *StorageFileSuite) TestFileProperties(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// create share
	share := cli.GetShareReference(shareName(c))
	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)
	root := share.GetRootDirectoryReference()

	fileSize := uint64(512)
	file := root.GetFileReference("test.dat")
	c.Assert(file.Create(fileSize, nil), chk.IsNil)

	// get initial set of properties
	c.Assert(file.Properties.Length, chk.Equals, fileSize)
	c.Assert(file.Properties.Etag, chk.NotNil)

	// set some file properties
	cc := "cachecontrol"
	ct := "mytype"
	enc := "noencoding"
	lang := "neutral"
	disp := "friendly"
	file.Properties.CacheControl = cc
	file.Properties.Type = ct
	file.Properties.Disposition = disp
	file.Properties.Encoding = enc
	file.Properties.Language = lang
	c.Assert(file.SetProperties(nil), chk.IsNil)

	// retrieve and verify
	c.Assert(file.FetchAttributes(nil), chk.IsNil)
	c.Assert(file.Properties.CacheControl, chk.Equals, cc)
	c.Assert(file.Properties.Type, chk.Equals, ct)
	c.Assert(file.Properties.Disposition, chk.Equals, disp)
	c.Assert(file.Properties.Encoding, chk.Equals, enc)
	c.Assert(file.Properties.Language, chk.Equals, lang)
}

func (s *StorageFileSuite) TestFileMetadata(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// create share
	share := cli.GetShareReference(shareName(c))
	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)
	root := share.GetRootDirectoryReference()

	fileSize := uint64(512)
	file := root.GetFileReference("test.dat")
	c.Assert(file.Create(fileSize, nil), chk.IsNil)

	// get metadata, shouldn't be any
	c.Assert(file.Metadata, chk.HasLen, 0)

	// set some custom metadata
	md := map[string]string{
		"something": "somethingvalue",
		"another":   "anothervalue",
	}
	file.Metadata = md
	c.Assert(file.SetMetadata(nil), chk.IsNil)

	// retrieve and verify
	c.Assert(file.FetchAttributes(nil), chk.IsNil)
	c.Assert(file.Metadata, chk.DeepEquals, md)
}

func (s *StorageFileSuite) TestFileMD5(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// create share
	share := cli.GetShareReference(shareName(c))
	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)
	root := share.GetRootDirectoryReference()

	// create file
	const size = uint64(1024)
	fileSize := uint64(size)
	file := root.GetFileReference("test.dat")
	c.Assert(file.Create(fileSize, nil), chk.IsNil)

	// fill file with some data and MD5 hash
	byteStream, contentMD5 := newByteStream(size)
	options := WriteRangeOptions{
		ContentMD5: contentMD5,
	}
	c.Assert(file.WriteRange(byteStream, FileRange{End: size - 1}, &options), chk.IsNil)

	// download file and verify
	downloadOptions := GetFileOptions{
		GetContentMD5: true,
	}
	stream, err := file.DownloadRangeToStream(FileRange{Start: 0, End: size - 1}, &downloadOptions)
	c.Assert(err, chk.IsNil)
	defer stream.Body.Close()
	c.Assert(stream.ContentMD5, chk.Equals, contentMD5)
}

// returns a byte stream along with a base-64 encoded MD5 hash of its contents
func newByteStream(count uint64) (io.Reader, string) {
	b := make([]uint8, count)
	for i := uint64(0); i < count; i++ {
		b[i] = 0xff
	}

	// create an MD5 hash of the array
	hash := md5.Sum(b)

	return bytes.NewReader(b), base64.StdEncoding.EncodeToString(hash[:])
}

func (s *StorageFileSuite) TestCopyFileSameAccountNoMetaData(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// create share
	share := cli.GetShareReference(shareName(c))
	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)
	root := share.GetRootDirectoryReference()

	// create directory structure
	dir1 := root.GetDirectoryReference("one")
	c.Assert(dir1.Create(nil), chk.IsNil)
	dir2 := dir1.GetDirectoryReference("two")
	c.Assert(dir2.Create(nil), chk.IsNil)

	// create file
	file := dir2.GetFileReference("some.file")
	c.Assert(file.Create(1024, nil), chk.IsNil)
	exists, err := file.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(exists, chk.Equals, true)

	otherFile := dir2.GetFileReference("someother.file")

	// copy the file, no timeout parameter
	err = otherFile.CopyFile(file.URL(), nil)
	c.Assert(err, chk.IsNil)

	// delete files
	c.Assert(file.Delete(nil), chk.IsNil)
	c.Assert(otherFile.Delete(nil), chk.IsNil)
}

func (s *StorageFileSuite) TestCopyFileSameAccountTimeout(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// create share
	share := cli.GetShareReference(shareName(c))
	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)
	root := share.GetRootDirectoryReference()

	// create directory structure
	dir1 := root.GetDirectoryReference("one")
	c.Assert(dir1.Create(nil), chk.IsNil)
	dir2 := dir1.GetDirectoryReference("two")
	c.Assert(dir2.Create(nil), chk.IsNil)

	// create file
	file := dir2.GetFileReference("some.file")
	c.Assert(file.Create(1024, nil), chk.IsNil)

	// copy the file, 60 second timeout.
	otherFile := dir2.GetFileReference("someother.file")
	options := FileRequestOptions{}
	options.Timeout = 60
	c.Assert(otherFile.CopyFile(file.URL(), &options), chk.IsNil)

	// delete files
	c.Assert(file.Delete(nil), chk.IsNil)
	c.Assert(otherFile.Delete(nil), chk.IsNil)
}

func (s *StorageFileSuite) TestCopyFileMissingFile(c *chk.C) {
	cli := getFileClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// create share
	share := cli.GetShareReference(shareName(c))
	c.Assert(share.Create(nil), chk.IsNil)
	defer share.Delete(nil)
	root := share.GetRootDirectoryReference()

	// create directory structure
	dir1 := root.GetDirectoryReference("one")
	c.Assert(dir1.Create(nil), chk.IsNil)

	otherFile := dir1.GetFileReference("someother.file")

	// copy the file, no timeout parameter
	err := otherFile.CopyFile("", nil)
	c.Assert(err, chk.NotNil)
}
