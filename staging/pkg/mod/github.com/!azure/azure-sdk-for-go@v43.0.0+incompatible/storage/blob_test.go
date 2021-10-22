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
	"encoding/xml"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"
	"strings"

	chk "gopkg.in/check.v1"
)

type StorageBlobSuite struct{}

var _ = chk.Suite(&StorageBlobSuite{})

func getBlobClient(c *chk.C) BlobStorageClient {
	return getBasicClient(c).GetBlobService()
}

func (s *StorageBlobSuite) Test_buildPath(c *chk.C) {
	cli := getBlobClient(c)
	cnt := cli.GetContainerReference("lol")
	b := cnt.GetBlobReference("rofl")
	c.Assert(b.buildPath(), chk.Equals, "/lol/rofl")
}

func (s *StorageBlobSuite) Test_pathForResource(c *chk.C) {
	c.Assert(pathForResource("lol", ""), chk.Equals, "/lol")
	c.Assert(pathForResource("lol", "blob"), chk.Equals, "/lol/blob")
}

func (s *StorageBlobSuite) TestBlobExists(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	b := cnt.GetBlobReference(blobName(c))
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)
	defer b.Delete(nil)

	ok, err := b.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)
	b.Name += ".lol"
	ok, err = b.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)

}

func (s *StorageBlobSuite) TestGetBlobURL(c *chk.C) {
	cli, err := NewBasicClient(dummyStorageAccount, dummyMiniStorageKey)
	c.Assert(err, chk.IsNil)
	blobCli := cli.GetBlobService()

	cnt := blobCli.GetContainerReference("c")
	b := cnt.GetBlobReference("nested/blob")
	c.Assert(b.GetURL(), chk.Equals, "https://golangrocksonazure.blob.core.windows.net/c/nested/blob")

	cnt.Name = ""
	c.Assert(b.GetURL(), chk.Equals, "https://golangrocksonazure.blob.core.windows.net/$root/nested/blob")

	b.Name = "blob"
	c.Assert(b.GetURL(), chk.Equals, "https://golangrocksonazure.blob.core.windows.net/$root/blob")

}

func (s *StorageBlobSuite) TestGetBlobContainerURL(c *chk.C) {
	cli, err := NewBasicClient(dummyStorageAccount, dummyMiniStorageKey)
	c.Assert(err, chk.IsNil)
	blobCli := cli.GetBlobService()

	cnt := blobCli.GetContainerReference("c")
	b := cnt.GetBlobReference("")
	c.Assert(b.GetURL(), chk.Equals, "https://golangrocksonazure.blob.core.windows.net/c")

	cnt.Name = ""
	c.Assert(b.GetURL(), chk.Equals, "https://golangrocksonazure.blob.core.windows.net/$root")
}

func (s *StorageBlobSuite) TestDeleteBlobIfExists(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.Delete(nil), chk.NotNil)

	ok, err := b.DeleteIfExists(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)
}

func (s *StorageBlobSuite) TestDeleteBlobWithConditions(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.CreateBlockBlob(nil), chk.IsNil)
	err := b.GetProperties(nil)
	c.Assert(err, chk.IsNil)
	etag := b.Properties.Etag

	// "Delete if matches incorrect or old Etag" should fail without deleting.
	options := DeleteBlobOptions{
		IfMatch: "GolangRocksOnAzure",
	}
	err = b.Delete(&options)
	c.Assert(err, chk.FitsTypeOf, AzureStorageServiceError{})
	c.Assert(err.(AzureStorageServiceError).StatusCode, chk.Equals, http.StatusPreconditionFailed)
	ok, err := b.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)

	// "Delete if matches new Etag" should succeed.
	options.IfMatch = etag
	ok, err = b.DeleteIfExists(&options)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)
}

func (s *StorageBlobSuite) TestGetBlobProperties(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	// try to get properties on a nonexisting blob
	blob1 := cnt.GetBlobReference(blobName(c, "1"))
	err := blob1.GetProperties(nil)
	c.Assert(err, chk.NotNil)

	// Put a blob
	blob2 := cnt.GetBlobReference(blobName(c, "2"))
	contents := content(64)
	c.Assert(blob2.putSingleBlockBlob(contents), chk.IsNil)

	// Get blob properties
	err = blob2.GetProperties(nil)
	c.Assert(err, chk.IsNil)

	c.Assert(blob2.Properties.ContentLength, chk.Equals, int64(len(contents)))
	c.Assert(blob2.Properties.ContentType, chk.Equals, "application/octet-stream")
	c.Assert(blob2.Properties.BlobType, chk.Equals, BlobTypeBlock)
}

// Ensure it's possible to generate a ListBlobs response with
// metadata, e.g., for a stub server.
func (s *StorageBlobSuite) TestMarshalBlobMetadata(c *chk.C) {
	buf, err := xml.Marshal(Blob{
		Name:       blobName(c),
		Properties: BlobProperties{},
		Metadata: map[string]string{
			"lol": "baz < waz",
		},
	})
	c.Assert(err, chk.IsNil)
	c.Assert(string(buf), chk.Matches, `.*<Metadata><Lol>baz &lt; waz</Lol></Metadata>.*`)
}

func (s *StorageBlobSuite) TestGetAndSetBlobMetadata(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	// Get empty metadata
	blob1 := cnt.GetBlobReference(blobName(c, "1"))
	c.Assert(blob1.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	err := blob1.GetMetadata(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(blob1.Metadata, chk.HasLen, 0)

	// Get and set the metadata
	blob2 := cnt.GetBlobReference(blobName(c, "2"))
	c.Assert(blob2.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)
	metaPut := BlobMetadata{
		"lol":      "rofl",
		"rofl_baz": "waz qux",
	}
	blob2.Metadata = metaPut

	err = blob2.SetMetadata(nil)
	c.Assert(err, chk.IsNil)

	err = blob2.GetMetadata(nil)
	c.Assert(err, chk.IsNil)
	c.Check(blob2.Metadata, chk.DeepEquals, metaPut)
}

func (s *StorageBlobSuite) TestMetadataCaseMunging(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	b := cnt.GetBlobReference(blobName(c))
	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	// Case munging
	metaPutUpper := BlobMetadata{
		"Lol":      "different rofl",
		"rofl_BAZ": "different waz qux",
	}
	metaExpectLower := BlobMetadata{
		"lol":      "different rofl",
		"rofl_baz": "different waz qux",
	}

	b.Metadata = metaPutUpper
	err := b.SetMetadata(nil)
	c.Assert(err, chk.IsNil)

	err = b.GetMetadata(nil)
	c.Assert(err, chk.IsNil)
	c.Check(b.Metadata, chk.DeepEquals, metaExpectLower)
}

func (s *StorageBlobSuite) TestSetMetadataWithExtraHeaders(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	meta := BlobMetadata{
		"lol":      "rofl",
		"rofl_baz": "waz qux",
	}
	b.Metadata = meta

	options := SetBlobMetadataOptions{
		IfMatch: "incorrect-etag",
	}

	// Set with incorrect If-Match in extra headers should result in error
	err := b.SetMetadata(&options)
	c.Assert(err, chk.NotNil)

	err = b.GetProperties(nil)
	c.Assert(err, chk.IsNil)

	// Set with matching If-Match in extra headers should succeed
	options.IfMatch = b.Properties.Etag
	b.Metadata = meta
	err = b.SetMetadata(&options)
	c.Assert(err, chk.IsNil)
}

func (s *StorageBlobSuite) TestSetBlobProperties(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	input := BlobProperties{
		CacheControl:    "private, max-age=0, no-cache",
		ContentMD5:      "oBATU+oaDduHWbVZLuzIJw==",
		ContentType:     "application/json",
		ContentEncoding: "gzip",
		ContentLanguage: "de-DE",
	}
	b.Properties = input

	err := b.SetProperties(nil)
	c.Assert(err, chk.IsNil)

	err = b.GetProperties(nil)
	c.Assert(err, chk.IsNil)

	c.Check(b.Properties.CacheControl, chk.Equals, input.CacheControl)
	c.Check(b.Properties.ContentType, chk.Equals, input.ContentType)
	c.Check(b.Properties.ContentMD5, chk.Equals, input.ContentMD5)
	c.Check(b.Properties.ContentEncoding, chk.Equals, input.ContentEncoding)
	c.Check(b.Properties.ContentLanguage, chk.Equals, input.ContentLanguage)
}

func (s *StorageBlobSuite) TestSetPageBlobProperties(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	size := int64(1024)
	b.Properties.ContentLength = size
	c.Assert(b.PutPageBlob(nil), chk.IsNil)

	b.Properties.ContentLength = int64(512)
	options := SetBlobPropertiesOptions{Timeout: 30}
	err := b.SetProperties(&options)
	c.Assert(err, chk.IsNil)
}

func (s *StorageBlobSuite) TestSnapshotBlob(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	snapshotTime, err := b.CreateSnapshot(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(snapshotTime, chk.NotNil)
}

func (s *StorageBlobSuite) TestSnapshotBlobWithTimeout(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	options := SnapshotOptions{
		Timeout: 0,
	}
	snapshotTime, err := b.CreateSnapshot(&options)
	c.Assert(err, chk.IsNil)
	c.Assert(snapshotTime, chk.NotNil)
}

func (s *StorageBlobSuite) TestSnapshotBlobWithValidLease(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	// generate lease.
	currentLeaseID, err := b.AcquireLease(30, "", nil)
	c.Assert(err, chk.IsNil)

	options := SnapshotOptions{
		LeaseID: currentLeaseID,
	}
	snapshotTime, err := b.CreateSnapshot(&options)
	c.Assert(err, chk.IsNil)
	c.Assert(snapshotTime, chk.NotNil)
}

func (s *StorageBlobSuite) TestSnapshotBlobWithInvalidLease(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	c.Assert(b.putSingleBlockBlob([]byte("Hello!")), chk.IsNil)

	// generate lease.
	leaseID, err := b.AcquireLease(30, "", nil)
	c.Assert(err, chk.IsNil)
	c.Assert(leaseID, chk.Not(chk.Equals), "")

	options := SnapshotOptions{
		LeaseID: "GolangRocksOnAzure",
	}
	snapshotTime, err := b.CreateSnapshot(&options)
	c.Assert(err, chk.NotNil)
	c.Assert(snapshotTime, chk.IsNil)
}

func (s *StorageBlobSuite) TestGetBlobRange(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	b := cnt.GetBlobReference(blobName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	body := "0123456789"
	c.Assert(b.putSingleBlockBlob([]byte(body)), chk.IsNil)
	defer b.Delete(nil)

	cases := []struct {
		options  GetBlobRangeOptions
		expected string
	}{
		{
			options: GetBlobRangeOptions{
				Range: &BlobRange{
					Start: 0,
					End:   uint64(len(body)),
				},
			},
			expected: body,
		},
		{
			options: GetBlobRangeOptions{
				Range: &BlobRange{
					Start: 0,
					End:   0,
				},
			},
			expected: body,
		},
		{
			options: GetBlobRangeOptions{
				Range: &BlobRange{
					Start: 1,
					End:   3,
				},
			},
			expected: body[1 : 3+1],
		},
		{
			options: GetBlobRangeOptions{
				Range: &BlobRange{
					Start: 3,
					End:   uint64(len(body)),
				},
			},
			expected: body[3:],
		},
		{
			options: GetBlobRangeOptions{
				Range: &BlobRange{
					Start: 3,
					End:   0,
				},
			},
			expected: body[3:],
		},
	}

	err := b.GetProperties(nil)
	c.Assert(err, chk.IsNil)

	// Read 1-3
	for _, r := range cases {
		resp, err := b.GetRange(&(r.options))
		c.Assert(err, chk.IsNil)
		blobBody, err := ioutil.ReadAll(resp)
		c.Assert(err, chk.IsNil)

		str := string(blobBody)
		c.Assert(str, chk.Equals, r.expected)

		// Was content length left untouched?
		c.Assert(b.Properties.ContentLength, chk.Equals, int64(len(body)))
	}
}

func (b *Blob) putSingleBlockBlob(chunk []byte) error {
	if len(chunk) > MaxBlobBlockSize {
		return fmt.Errorf("storage: provided chunk (%d bytes) cannot fit into single-block blob (max %d bytes)", len(chunk), MaxBlobBlockSize)
	}

	uri := b.Container.bsc.client.getEndpoint(blobServiceName, b.buildPath(), nil)
	headers := b.Container.bsc.client.getStandardHeaders()
	b.Properties.BlobType = BlobTypeBlock
	headers["x-ms-blob-type"] = string(BlobTypeBlock)
	headers["Content-Length"] = strconv.Itoa(len(chunk))

	resp, err := b.Container.bsc.client.exec(http.MethodPut, uri, headers, bytes.NewReader(chunk), b.Container.bsc.auth)
	if err != nil {
		return err
	}
	return checkRespCode(resp, []int{http.StatusCreated})
}

func blobName(c *chk.C, extras ...string) string {
	return nameGenerator(1024, "blob/", alphanum, c, extras)

}

func contentWithSpecialChars(n int) string {
	name := string(content(n)) + "/" + string(content(n)) + "-._~:?#[]@!$&'()*,;+= " + string(content(n))
	return name
}

func nameGenerator(maxLen int, prefix, valid string, c *chk.C, extras []string) string {
	extra := strings.Join(extras, "")
	name := prefix + extra + removeInvalidCharacters(c.TestName(), valid)
	if len(name) > maxLen {
		return name[:maxLen]
	}
	return name
}

func removeInvalidCharacters(unformatted string, valid string) string {
	unformatted = strings.ToLower(unformatted)
	buffer := bytes.NewBufferString(strconv.Itoa((len(unformatted))))
	runes := []rune(unformatted)
	for _, r := range runes {
		if strings.ContainsRune(valid, r) {
			buffer.WriteRune(r)
		}
	}
	return string(buffer.Bytes())
}

func content(n int) []byte {
	buffer := bytes.NewBufferString("")
	rep := (n / len(veryLongString)) + 1
	for i := 0; i < rep; i++ {
		buffer.WriteString(veryLongString)
	}
	return buffer.Bytes()[:n]
}

const (
	alphanum       = "0123456789abcdefghijklmnopqrstuvwxyz"
	alpha          = "abcdefghijklmnopqrstuvwxyz"
	veryLongString = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer feugiat eleifend scelerisque. Phasellus tempor turpis eget magna pretium, et finibus massa convallis. Donec eget lacinia nibh. Ut ut cursus odio. Quisque id justo interdum, maximus ex a, dapibus leo. Nullam mattis arcu nec justo vehicula pretium. Curabitur fermentum quam ac dolor venenatis, vitae scelerisque ex posuere. Donec ut ante porttitor, ultricies ante ac, pulvinar metus. Nunc suscipit elit gravida dolor facilisis sollicitudin. Fusce ac ultrices libero. Donec erat lectus, hendrerit volutpat nisl quis, porta accumsan nibh. Pellentesque hendrerit nisi id mi porttitor maximus. Phasellus vitae venenatis velit. Quisque id felis nec lacus iaculis porttitor. Maecenas egestas tortor et nulla dapibus varius. In hac habitasse platea dictumst."
)
