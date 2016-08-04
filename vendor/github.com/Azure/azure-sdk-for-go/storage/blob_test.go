package storage

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/xml"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"sort"
	"sync"
	"testing"
	"time"

	chk "gopkg.in/check.v1"
)

type StorageBlobSuite struct{}

var _ = chk.Suite(&StorageBlobSuite{})

const testContainerPrefix = "zzzztest-"

func getBlobClient(c *chk.C) BlobStorageClient {
	return getBasicClient(c).GetBlobService()
}

func (s *StorageBlobSuite) Test_pathForContainer(c *chk.C) {
	c.Assert(pathForContainer("foo"), chk.Equals, "/foo")
}

func (s *StorageBlobSuite) Test_pathForBlob(c *chk.C) {
	c.Assert(pathForBlob("foo", "blob"), chk.Equals, "/foo/blob")
}

func (s *StorageBlobSuite) Test_blobSASStringToSign(c *chk.C) {
	_, err := blobSASStringToSign("2012-02-12", "CS", "SE", "SP")
	c.Assert(err, chk.NotNil) // not implemented SAS for versions earlier than 2013-08-15

	out, err := blobSASStringToSign("2013-08-15", "CS", "SE", "SP")
	c.Assert(err, chk.IsNil)
	c.Assert(out, chk.Equals, "SP\n\nSE\nCS\n\n2013-08-15\n\n\n\n\n")
}

func (s *StorageBlobSuite) TestGetBlobSASURI(c *chk.C) {
	api, err := NewClient("foo", "YmFy", DefaultBaseURL, "2013-08-15", true)
	c.Assert(err, chk.IsNil)
	cli := api.GetBlobService()
	expiry := time.Time{}

	expectedParts := url.URL{
		Scheme: "https",
		Host:   "foo.blob.core.windows.net",
		Path:   "container/name",
		RawQuery: url.Values{
			"sv":  {"2013-08-15"},
			"sig": {"/OXG7rWh08jYwtU03GzJM0DHZtidRGpC6g69rSGm3I0="},
			"sr":  {"b"},
			"sp":  {"r"},
			"se":  {"0001-01-01T00:00:00Z"},
		}.Encode()}

	u, err := cli.GetBlobSASURI("container", "name", expiry, "r")
	c.Assert(err, chk.IsNil)
	sasParts, err := url.Parse(u)
	c.Assert(err, chk.IsNil)
	c.Assert(expectedParts.String(), chk.Equals, sasParts.String())
	c.Assert(expectedParts.Query(), chk.DeepEquals, sasParts.Query())
}

func (s *StorageBlobSuite) TestBlobSASURICorrectness(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()
	blob := randNameWithSpecialChars(5)
	body := []byte(randString(100))
	expiry := time.Now().UTC().Add(time.Hour)
	permissions := "r"

	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.DeleteContainer(cnt)

	c.Assert(cli.putSingleBlockBlob(cnt, blob, body), chk.IsNil)

	sasURI, err := cli.GetBlobSASURI(cnt, blob, expiry, permissions)
	c.Assert(err, chk.IsNil)

	resp, err := http.Get(sasURI)
	c.Assert(err, chk.IsNil)

	blobResp, err := ioutil.ReadAll(resp.Body)
	defer resp.Body.Close()
	c.Assert(err, chk.IsNil)

	c.Assert(resp.StatusCode, chk.Equals, http.StatusOK)
	c.Assert(len(blobResp), chk.Equals, len(body))
}

func (s *StorageBlobSuite) TestListContainersPagination(c *chk.C) {
	cli := getBlobClient(c)
	c.Assert(deleteTestContainers(cli), chk.IsNil)

	const n = 5
	const pageSize = 2

	// Create test containers
	created := []string{}
	for i := 0; i < n; i++ {
		name := randContainer()
		c.Assert(cli.CreateContainer(name, ContainerAccessTypePrivate), chk.IsNil)
		created = append(created, name)
	}
	sort.Strings(created)

	// Defer test container deletions
	defer func() {
		var wg sync.WaitGroup
		for _, cnt := range created {
			wg.Add(1)
			go func(name string) {
				c.Assert(cli.DeleteContainer(name), chk.IsNil)
				wg.Done()
			}(cnt)
		}
		wg.Wait()
	}()

	// Paginate results
	seen := []string{}
	marker := ""
	for {
		resp, err := cli.ListContainers(ListContainersParameters{
			Prefix:     testContainerPrefix,
			MaxResults: pageSize,
			Marker:     marker})
		c.Assert(err, chk.IsNil)

		containers := resp.Containers
		if len(containers) > pageSize {
			c.Fatalf("Got a bigger page. Expected: %d, got: %d", pageSize, len(containers))
		}

		for _, c := range containers {
			seen = append(seen, c.Name)
		}

		marker = resp.NextMarker
		if marker == "" || len(containers) == 0 {
			break
		}
	}

	c.Assert(seen, chk.DeepEquals, created)
}

func (s *StorageBlobSuite) TestContainerExists(c *chk.C) {
	cnt := randContainer()
	cli := getBlobClient(c)
	ok, err := cli.ContainerExists(cnt)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)

	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypeBlob), chk.IsNil)
	defer cli.DeleteContainer(cnt)

	ok, err = cli.ContainerExists(cnt)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)
}

func (s *StorageBlobSuite) TestCreateContainerDeleteContainer(c *chk.C) {
	cnt := randContainer()
	cli := getBlobClient(c)
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	c.Assert(cli.DeleteContainer(cnt), chk.IsNil)
}

func (s *StorageBlobSuite) TestCreateContainerIfNotExists(c *chk.C) {
	cnt := randContainer()
	cli := getBlobClient(c)
	defer cli.DeleteContainer(cnt)

	// First create
	ok, err := cli.CreateContainerIfNotExists(cnt, ContainerAccessTypePrivate)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)

	// Second create, should not give errors
	ok, err = cli.CreateContainerIfNotExists(cnt, ContainerAccessTypePrivate)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)
}

func (s *StorageBlobSuite) TestDeleteContainerIfExists(c *chk.C) {
	cnt := randContainer()
	cli := getBlobClient(c)

	// Nonexisting container
	c.Assert(cli.DeleteContainer(cnt), chk.NotNil)

	ok, err := cli.DeleteContainerIfExists(cnt)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)

	// Existing container
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	ok, err = cli.DeleteContainerIfExists(cnt)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)
}

func (s *StorageBlobSuite) TestBlobExists(c *chk.C) {
	cnt := randContainer()
	blob := randName(5)
	cli := getBlobClient(c)

	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypeBlob), chk.IsNil)
	defer cli.DeleteContainer(cnt)
	c.Assert(cli.putSingleBlockBlob(cnt, blob, []byte("Hello!")), chk.IsNil)
	defer cli.DeleteBlob(cnt, blob, nil)

	ok, err := cli.BlobExists(cnt, blob+".foo")
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)

	ok, err = cli.BlobExists(cnt, blob)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)
}

func (s *StorageBlobSuite) TestGetBlobURL(c *chk.C) {
	api, err := NewBasicClient("foo", "YmFy")
	c.Assert(err, chk.IsNil)
	cli := api.GetBlobService()

	c.Assert(cli.GetBlobURL("c", "nested/blob"), chk.Equals, "https://foo.blob.core.windows.net/c/nested/blob")
	c.Assert(cli.GetBlobURL("", "blob"), chk.Equals, "https://foo.blob.core.windows.net/$root/blob")
	c.Assert(cli.GetBlobURL("", "nested/blob"), chk.Equals, "https://foo.blob.core.windows.net/$root/nested/blob")
}

func (s *StorageBlobSuite) TestBlobCopy(c *chk.C) {
	if testing.Short() {
		c.Skip("skipping blob copy in short mode, no SLA on async operation")
	}

	cli := getBlobClient(c)
	cnt := randContainer()
	src := randName(5)
	dst := randName(5)
	body := []byte(randString(1024))

	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	c.Assert(cli.putSingleBlockBlob(cnt, src, body), chk.IsNil)
	defer cli.DeleteBlob(cnt, src, nil)

	c.Assert(cli.CopyBlob(cnt, dst, cli.GetBlobURL(cnt, src)), chk.IsNil)
	defer cli.DeleteBlob(cnt, dst, nil)

	blobBody, err := cli.GetBlob(cnt, dst)
	c.Assert(err, chk.IsNil)

	b, err := ioutil.ReadAll(blobBody)
	defer blobBody.Close()
	c.Assert(err, chk.IsNil)
	c.Assert(b, chk.DeepEquals, body)
}

func (s *StorageBlobSuite) TestDeleteBlobIfExists(c *chk.C) {
	cnt := randContainer()
	blob := randName(5)

	cli := getBlobClient(c)
	c.Assert(cli.DeleteBlob(cnt, blob, nil), chk.NotNil)

	ok, err := cli.DeleteBlobIfExists(cnt, blob, nil)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)
}

func (s *StorageBlobSuite) TestDeleteBlobWithConditions(c *chk.C) {
	cnt := randContainer()
	blob := randName(5)

	cli := getBlobClient(c)

	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	c.Assert(cli.CreateBlockBlob(cnt, blob), chk.IsNil)
	oldProps, err := cli.GetBlobProperties(cnt, blob)
	c.Assert(err, chk.IsNil)

	// Update metadata, so Etag changes
	c.Assert(cli.SetBlobMetadata(cnt, blob, map[string]string{}, nil), chk.IsNil)
	newProps, err := cli.GetBlobProperties(cnt, blob)
	c.Assert(err, chk.IsNil)

	// "Delete if matches old Etag" should fail without deleting.
	err = cli.DeleteBlob(cnt, blob, map[string]string{
		"If-Match": oldProps.Etag,
	})
	c.Assert(err, chk.FitsTypeOf, AzureStorageServiceError{})
	c.Assert(err.(AzureStorageServiceError).StatusCode, chk.Equals, http.StatusPreconditionFailed)
	_, err = cli.GetBlob(cnt, blob)
	c.Assert(err, chk.IsNil)

	// "Delete if matches new Etag" should succeed.
	err = cli.DeleteBlob(cnt, blob, map[string]string{
		"If-Match": newProps.Etag,
	})
	c.Assert(err, chk.IsNil)
	_, err = cli.GetBlob(cnt, blob)
	c.Assert(err, chk.Not(chk.IsNil))
}

func (s *StorageBlobSuite) TestGetBlobProperties(c *chk.C) {
	cnt := randContainer()
	blob := randName(5)
	contents := randString(64)

	cli := getBlobClient(c)
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.DeleteContainer(cnt)

	// Nonexisting blob
	_, err := cli.GetBlobProperties(cnt, blob)
	c.Assert(err, chk.NotNil)

	// Put the blob
	c.Assert(cli.putSingleBlockBlob(cnt, blob, []byte(contents)), chk.IsNil)

	// Get blob properties
	props, err := cli.GetBlobProperties(cnt, blob)
	c.Assert(err, chk.IsNil)

	c.Assert(props.ContentLength, chk.Equals, int64(len(contents)))
	c.Assert(props.ContentType, chk.Equals, "application/octet-stream")
	c.Assert(props.BlobType, chk.Equals, BlobTypeBlock)
}

func (s *StorageBlobSuite) TestListBlobsPagination(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()

	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.DeleteContainer(cnt)

	blobs := []string{}
	const n = 5
	const pageSize = 2
	for i := 0; i < n; i++ {
		name := randName(5)
		c.Assert(cli.putSingleBlockBlob(cnt, name, []byte("Hello, world!")), chk.IsNil)
		blobs = append(blobs, name)
	}
	sort.Strings(blobs)

	// Paginate
	seen := []string{}
	marker := ""
	for {
		resp, err := cli.ListBlobs(cnt, ListBlobsParameters{
			MaxResults: pageSize,
			Marker:     marker})
		c.Assert(err, chk.IsNil)

		for _, v := range resp.Blobs {
			seen = append(seen, v.Name)
		}

		marker = resp.NextMarker
		if marker == "" || len(resp.Blobs) == 0 {
			break
		}
	}

	// Compare
	c.Assert(seen, chk.DeepEquals, blobs)
}

// listBlobsAsFiles is a helper function to list blobs as "folders" and "files".
func listBlobsAsFiles(cli BlobStorageClient, cnt string, parentDir string) (folders []string, files []string, err error) {
	var blobParams ListBlobsParameters
	var blobListResponse BlobListResponse

	// Top level "folders"
	blobParams = ListBlobsParameters{
		Delimiter: "/",
		Prefix:    parentDir,
	}

	blobListResponse, err = cli.ListBlobs(cnt, blobParams)
	if err != nil {
		return nil, nil, err
	}

	// These are treated as "folders" under the parentDir.
	folders = blobListResponse.BlobPrefixes

	// "Files"" are blobs which are under the parentDir.
	files = make([]string, len(blobListResponse.Blobs))
	for i := range blobListResponse.Blobs {
		files[i] = blobListResponse.Blobs[i].Name
	}

	return folders, files, nil
}

// TestListBlobsTraversal tests that we can correctly traverse
// blobs in blob storage as if it were a file system by using
// a combination of Prefix, Delimiter, and BlobPrefixes.
//
// Blob storage is flat, but we can *simulate* the file
// system with folders and files using conventions in naming.
// With the blob namedd "/usr/bin/ls", when we use delimiter '/',
// the "ls" would be a "file"; with "/", /usr" and "/usr/bin" being
// the "folders"
//
// NOTE: The use of delimiter (eg forward slash) is extremely fiddly
// and difficult to get right so some discipline in naming and rules
// when using the API is required to get everything to work as expected.
//
// Assuming our delimiter is a forward slash, the rules are:
//
//  - Do use a leading forward slash in blob names to make things
//    consistent and simpler (see further).
//    Note that doing so will show "<no name>" as the only top-level
//    folder in the container in Azure portal, which may look strange.
//
//  - The "folder names" are returned *with trailing forward slash* as per MSDN.
//
//  - The "folder names" will be "absolue paths", e.g. listing things under "/usr/"
//    will return folder names "/usr/bin/".
//
//  - The "file names" are returned as full blob names, e.g. when listing
//    things under "/usr/bin/", the file names will be "/usr/bin/ls" and
//    "/usr/bin/cat".
//
//  - Everything is returned with case-sensitive order as expected in real file system
//    as per MSDN.
//
//  - To list things under a "folder" always use trailing forward slash.
//
//    Example: to list top level folders we use root folder named "" with
//    trailing forward slash, so we use "/".
//
//    Example: to list folders under "/usr", we again append forward slash and
//    so we use "/usr/".
//
//    Because we use leading forward slash we don't need to have different
//    treatment of "get top-level folders" and "get non-top-level folders"
//    scenarios.
func (s *StorageBlobSuite) TestListBlobsTraversal(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()

	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.DeleteContainer(cnt)

	// Note use of leading forward slash as per naming rules.
	blobsToCreate := []string{
		"/usr/bin/ls",
		"/usr/bin/cat",
		"/usr/lib64/libc.so",
		"/etc/hosts",
		"/etc/init.d/iptables",
	}

	// Create the above blobs
	for _, blobName := range blobsToCreate {
		err := cli.CreateBlockBlob(cnt, blobName)
		c.Assert(err, chk.IsNil)
	}

	var folders []string
	var files []string
	var err error

	// Top level folders and files.
	folders, files, err = listBlobsAsFiles(cli, cnt, "/")
	c.Assert(err, chk.IsNil)
	c.Assert(folders, chk.DeepEquals, []string{"/etc/", "/usr/"})
	c.Assert(files, chk.DeepEquals, []string{})

	// Things under /etc/. Note use of trailing forward slash here as per rules.
	folders, files, err = listBlobsAsFiles(cli, cnt, "/etc/")
	c.Assert(err, chk.IsNil)
	c.Assert(folders, chk.DeepEquals, []string{"/etc/init.d/"})
	c.Assert(files, chk.DeepEquals, []string{"/etc/hosts"})

	// Things under /etc/init.d/
	folders, files, err = listBlobsAsFiles(cli, cnt, "/etc/init.d/")
	c.Assert(err, chk.IsNil)
	c.Assert(folders, chk.DeepEquals, []string(nil))
	c.Assert(files, chk.DeepEquals, []string{"/etc/init.d/iptables"})

	// Things under /usr/
	folders, files, err = listBlobsAsFiles(cli, cnt, "/usr/")
	c.Assert(err, chk.IsNil)
	c.Assert(folders, chk.DeepEquals, []string{"/usr/bin/", "/usr/lib64/"})
	c.Assert(files, chk.DeepEquals, []string{})

	// Things under /usr/bin/
	folders, files, err = listBlobsAsFiles(cli, cnt, "/usr/bin/")
	c.Assert(err, chk.IsNil)
	c.Assert(folders, chk.DeepEquals, []string(nil))
	c.Assert(files, chk.DeepEquals, []string{"/usr/bin/cat", "/usr/bin/ls"})
}

func (s *StorageBlobSuite) TestListBlobsWithMetadata(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()

	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	expectMeta := make(map[string]BlobMetadata)

	// Put 4 blobs with metadata
	for i := 0; i < 4; i++ {
		name := randName(5)
		c.Assert(cli.putSingleBlockBlob(cnt, name, []byte("Hello, world!")), chk.IsNil)
		c.Assert(cli.SetBlobMetadata(cnt, name, map[string]string{
			"Foo":     name,
			"Bar_BAZ": "Waz Qux",
		}, nil), chk.IsNil)
		expectMeta[name] = BlobMetadata{
			"foo":     name,
			"bar_baz": "Waz Qux",
		}
	}

	// Put one more blob with no metadata
	blobWithoutMetadata := randName(5)
	c.Assert(cli.putSingleBlockBlob(cnt, blobWithoutMetadata, []byte("Hello, world!")), chk.IsNil)
	expectMeta[blobWithoutMetadata] = nil

	// Get ListBlobs with include:"metadata"
	resp, err := cli.ListBlobs(cnt, ListBlobsParameters{
		MaxResults: 5,
		Include:    "metadata"})
	c.Assert(err, chk.IsNil)

	respBlobs := make(map[string]Blob)
	for _, v := range resp.Blobs {
		respBlobs[v.Name] = v
	}

	// Verify the metadata is as expected
	for name := range expectMeta {
		c.Check(respBlobs[name].Metadata, chk.DeepEquals, expectMeta[name])
	}
}

// Ensure it's possible to generate a ListBlobs response with
// metadata, e.g., for a stub server.
func (s *StorageBlobSuite) TestMarshalBlobMetadata(c *chk.C) {
	buf, err := xml.Marshal(Blob{
		Name:       randName(5),
		Properties: BlobProperties{},
		Metadata:   BlobMetadata{"foo": "baz < waz"},
	})
	c.Assert(err, chk.IsNil)
	c.Assert(string(buf), chk.Matches, `.*<Metadata><Foo>baz &lt; waz</Foo></Metadata>.*`)
}

func (s *StorageBlobSuite) TestGetAndSetMetadata(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()

	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	blob := randName(5)
	c.Assert(cli.putSingleBlockBlob(cnt, blob, []byte{}), chk.IsNil)

	m, err := cli.GetBlobMetadata(cnt, blob)
	c.Assert(err, chk.IsNil)
	c.Assert(m, chk.Not(chk.Equals), nil)
	c.Assert(len(m), chk.Equals, 0)

	mPut := map[string]string{
		"foo":     "bar",
		"bar_baz": "waz qux",
	}

	err = cli.SetBlobMetadata(cnt, blob, mPut, nil)
	c.Assert(err, chk.IsNil)

	m, err = cli.GetBlobMetadata(cnt, blob)
	c.Assert(err, chk.IsNil)
	c.Check(m, chk.DeepEquals, mPut)

	// Case munging

	mPutUpper := map[string]string{
		"Foo":     "different bar",
		"bar_BAZ": "different waz qux",
	}
	mExpectLower := map[string]string{
		"foo":     "different bar",
		"bar_baz": "different waz qux",
	}

	err = cli.SetBlobMetadata(cnt, blob, mPutUpper, nil)
	c.Assert(err, chk.IsNil)

	m, err = cli.GetBlobMetadata(cnt, blob)
	c.Assert(err, chk.IsNil)
	c.Check(m, chk.DeepEquals, mExpectLower)
}

func (s *StorageBlobSuite) TestSetMetadataWithExtraHeaders(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()

	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	blob := randName(5)
	c.Assert(cli.putSingleBlockBlob(cnt, blob, []byte{}), chk.IsNil)

	mPut := map[string]string{
		"foo":     "bar",
		"bar_baz": "waz qux",
	}

	extraHeaders := map[string]string{
		"If-Match": "incorrect-etag",
	}

	// Set with incorrect If-Match in extra headers should result in error
	err := cli.SetBlobMetadata(cnt, blob, mPut, extraHeaders)
	c.Assert(err, chk.NotNil)

	props, err := cli.GetBlobProperties(cnt, blob)
	extraHeaders = map[string]string{
		"If-Match": props.Etag,
	}

	// Set with matching If-Match in extra headers should succeed
	err = cli.SetBlobMetadata(cnt, blob, mPut, extraHeaders)
	c.Assert(err, chk.IsNil)
}

func (s *StorageBlobSuite) TestPutEmptyBlockBlob(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()

	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	blob := randName(5)
	c.Assert(cli.putSingleBlockBlob(cnt, blob, []byte{}), chk.IsNil)

	props, err := cli.GetBlobProperties(cnt, blob)
	c.Assert(err, chk.IsNil)
	c.Assert(props.ContentLength, chk.Not(chk.Equals), 0)
}

func (s *StorageBlobSuite) TestGetBlobRange(c *chk.C) {
	cnt := randContainer()
	blob := randName(5)
	body := "0123456789"

	cli := getBlobClient(c)
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypeBlob), chk.IsNil)
	defer cli.DeleteContainer(cnt)

	c.Assert(cli.putSingleBlockBlob(cnt, blob, []byte(body)), chk.IsNil)
	defer cli.DeleteBlob(cnt, blob, nil)

	// Read 1-3
	for _, r := range []struct {
		rangeStr string
		expected string
	}{
		{"0-", body},
		{"1-3", body[1 : 3+1]},
		{"3-", body[3:]},
	} {
		resp, err := cli.GetBlobRange(cnt, blob, r.rangeStr, nil)
		c.Assert(err, chk.IsNil)
		blobBody, err := ioutil.ReadAll(resp)
		c.Assert(err, chk.IsNil)

		str := string(blobBody)
		c.Assert(str, chk.Equals, r.expected)
	}
}

func (s *StorageBlobSuite) TestCreateBlockBlobFromReader(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	name := randName(5)
	data := randBytes(8888)
	c.Assert(cli.CreateBlockBlobFromReader(cnt, name, uint64(len(data)), bytes.NewReader(data), nil), chk.IsNil)

	body, err := cli.GetBlob(cnt, name)
	c.Assert(err, chk.IsNil)
	gotData, err := ioutil.ReadAll(body)
	body.Close()

	c.Assert(err, chk.IsNil)
	c.Assert(gotData, chk.DeepEquals, data)
}

func (s *StorageBlobSuite) TestCreateBlockBlobFromReaderWithShortData(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	name := randName(5)
	data := randBytes(8888)
	err := cli.CreateBlockBlobFromReader(cnt, name, 9999, bytes.NewReader(data), nil)
	c.Assert(err, chk.Not(chk.IsNil))

	_, err = cli.GetBlob(cnt, name)
	// Upload was incomplete: blob should not have been created.
	c.Assert(err, chk.Not(chk.IsNil))
}

func (s *StorageBlobSuite) TestPutBlock(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	blob := randName(5)
	chunk := []byte(randString(1024))
	blockID := base64.StdEncoding.EncodeToString([]byte("foo"))
	c.Assert(cli.PutBlock(cnt, blob, blockID, chunk), chk.IsNil)
}

func (s *StorageBlobSuite) TestGetBlockList_PutBlockList(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	blob := randName(5)
	chunk := []byte(randString(1024))
	blockID := base64.StdEncoding.EncodeToString([]byte("foo"))

	// Put one block
	c.Assert(cli.PutBlock(cnt, blob, blockID, chunk), chk.IsNil)
	defer cli.deleteBlob(cnt, blob, nil)

	// Get committed blocks
	committed, err := cli.GetBlockList(cnt, blob, BlockListTypeCommitted)
	c.Assert(err, chk.IsNil)

	if len(committed.CommittedBlocks) > 0 {
		c.Fatal("There are committed blocks")
	}

	// Get uncommitted blocks
	uncommitted, err := cli.GetBlockList(cnt, blob, BlockListTypeUncommitted)
	c.Assert(err, chk.IsNil)

	c.Assert(len(uncommitted.UncommittedBlocks), chk.Equals, 1)
	// Commit block list
	c.Assert(cli.PutBlockList(cnt, blob, []Block{{blockID, BlockStatusUncommitted}}), chk.IsNil)

	// Get all blocks
	all, err := cli.GetBlockList(cnt, blob, BlockListTypeAll)
	c.Assert(err, chk.IsNil)
	c.Assert(len(all.CommittedBlocks), chk.Equals, 1)
	c.Assert(len(all.UncommittedBlocks), chk.Equals, 0)

	// Verify the block
	thatBlock := all.CommittedBlocks[0]
	c.Assert(thatBlock.Name, chk.Equals, blockID)
	c.Assert(thatBlock.Size, chk.Equals, int64(len(chunk)))
}

func (s *StorageBlobSuite) TestCreateBlockBlob(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	blob := randName(5)
	c.Assert(cli.CreateBlockBlob(cnt, blob), chk.IsNil)

	// Verify
	blocks, err := cli.GetBlockList(cnt, blob, BlockListTypeAll)
	c.Assert(err, chk.IsNil)
	c.Assert(len(blocks.CommittedBlocks), chk.Equals, 0)
	c.Assert(len(blocks.UncommittedBlocks), chk.Equals, 0)
}

func (s *StorageBlobSuite) TestPutPageBlob(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	blob := randName(5)
	size := int64(10 * 1024 * 1024)
	c.Assert(cli.PutPageBlob(cnt, blob, size, nil), chk.IsNil)

	// Verify
	props, err := cli.GetBlobProperties(cnt, blob)
	c.Assert(err, chk.IsNil)
	c.Assert(props.ContentLength, chk.Equals, size)
	c.Assert(props.BlobType, chk.Equals, BlobTypePage)
}

func (s *StorageBlobSuite) TestPutPagesUpdate(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	blob := randName(5)
	size := int64(10 * 1024 * 1024) // larger than we'll use
	c.Assert(cli.PutPageBlob(cnt, blob, size, nil), chk.IsNil)

	chunk1 := []byte(randString(1024))
	chunk2 := []byte(randString(512))

	// Append chunks
	c.Assert(cli.PutPage(cnt, blob, 0, int64(len(chunk1)-1), PageWriteTypeUpdate, chunk1, nil), chk.IsNil)
	c.Assert(cli.PutPage(cnt, blob, int64(len(chunk1)), int64(len(chunk1)+len(chunk2)-1), PageWriteTypeUpdate, chunk2, nil), chk.IsNil)

	// Verify contents
	out, err := cli.GetBlobRange(cnt, blob, fmt.Sprintf("%v-%v", 0, len(chunk1)+len(chunk2)-1), nil)
	c.Assert(err, chk.IsNil)
	defer out.Close()
	blobContents, err := ioutil.ReadAll(out)
	c.Assert(err, chk.IsNil)
	c.Assert(blobContents, chk.DeepEquals, append(chunk1, chunk2...))
	out.Close()

	// Overwrite first half of chunk1
	chunk0 := []byte(randString(512))
	c.Assert(cli.PutPage(cnt, blob, 0, int64(len(chunk0)-1), PageWriteTypeUpdate, chunk0, nil), chk.IsNil)

	// Verify contents
	out, err = cli.GetBlobRange(cnt, blob, fmt.Sprintf("%v-%v", 0, len(chunk1)+len(chunk2)-1), nil)
	c.Assert(err, chk.IsNil)
	defer out.Close()
	blobContents, err = ioutil.ReadAll(out)
	c.Assert(err, chk.IsNil)
	c.Assert(blobContents, chk.DeepEquals, append(append(chunk0, chunk1[512:]...), chunk2...))
}

func (s *StorageBlobSuite) TestPutPagesClear(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	blob := randName(5)
	size := int64(10 * 1024 * 1024) // larger than we'll use
	c.Assert(cli.PutPageBlob(cnt, blob, size, nil), chk.IsNil)

	// Put 0-2047
	chunk := []byte(randString(2048))
	c.Assert(cli.PutPage(cnt, blob, 0, 2047, PageWriteTypeUpdate, chunk, nil), chk.IsNil)

	// Clear 512-1023
	c.Assert(cli.PutPage(cnt, blob, 512, 1023, PageWriteTypeClear, nil, nil), chk.IsNil)

	// Verify contents
	out, err := cli.GetBlobRange(cnt, blob, "0-2047", nil)
	c.Assert(err, chk.IsNil)
	contents, err := ioutil.ReadAll(out)
	c.Assert(err, chk.IsNil)
	defer out.Close()
	c.Assert(contents, chk.DeepEquals, append(append(chunk[:512], make([]byte, 512)...), chunk[1024:]...))
}

func (s *StorageBlobSuite) TestGetPageRanges(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	blob := randName(5)
	size := int64(10 * 1024 * 1024) // larger than we'll use
	c.Assert(cli.PutPageBlob(cnt, blob, size, nil), chk.IsNil)

	// Get page ranges on empty blob
	out, err := cli.GetPageRanges(cnt, blob)
	c.Assert(err, chk.IsNil)
	c.Assert(len(out.PageList), chk.Equals, 0)

	// Add 0-512 page
	c.Assert(cli.PutPage(cnt, blob, 0, 511, PageWriteTypeUpdate, []byte(randString(512)), nil), chk.IsNil)

	out, err = cli.GetPageRanges(cnt, blob)
	c.Assert(err, chk.IsNil)
	c.Assert(len(out.PageList), chk.Equals, 1)

	// Add 1024-2048
	c.Assert(cli.PutPage(cnt, blob, 1024, 2047, PageWriteTypeUpdate, []byte(randString(1024)), nil), chk.IsNil)

	out, err = cli.GetPageRanges(cnt, blob)
	c.Assert(err, chk.IsNil)
	c.Assert(len(out.PageList), chk.Equals, 2)
}

func (s *StorageBlobSuite) TestPutAppendBlob(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	blob := randName(5)
	c.Assert(cli.PutAppendBlob(cnt, blob, nil), chk.IsNil)

	// Verify
	props, err := cli.GetBlobProperties(cnt, blob)
	c.Assert(err, chk.IsNil)
	c.Assert(props.ContentLength, chk.Equals, int64(0))
	c.Assert(props.BlobType, chk.Equals, BlobTypeAppend)
}

func (s *StorageBlobSuite) TestPutAppendBlobAppendBlocks(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	blob := randName(5)
	c.Assert(cli.PutAppendBlob(cnt, blob, nil), chk.IsNil)

	chunk1 := []byte(randString(1024))
	chunk2 := []byte(randString(512))

	// Append first block
	c.Assert(cli.AppendBlock(cnt, blob, chunk1, nil), chk.IsNil)

	// Verify contents
	out, err := cli.GetBlobRange(cnt, blob, fmt.Sprintf("%v-%v", 0, len(chunk1)-1), nil)
	c.Assert(err, chk.IsNil)
	defer out.Close()
	blobContents, err := ioutil.ReadAll(out)
	c.Assert(err, chk.IsNil)
	c.Assert(blobContents, chk.DeepEquals, chunk1)
	out.Close()

	// Append second block
	c.Assert(cli.AppendBlock(cnt, blob, chunk2, nil), chk.IsNil)

	// Verify contents
	out, err = cli.GetBlobRange(cnt, blob, fmt.Sprintf("%v-%v", 0, len(chunk1)+len(chunk2)-1), nil)
	c.Assert(err, chk.IsNil)
	defer out.Close()
	blobContents, err = ioutil.ReadAll(out)
	c.Assert(err, chk.IsNil)
	c.Assert(blobContents, chk.DeepEquals, append(chunk1, chunk2...))
	out.Close()
}

func deleteTestContainers(cli BlobStorageClient) error {
	for {
		resp, err := cli.ListContainers(ListContainersParameters{Prefix: testContainerPrefix})
		if err != nil {
			return err
		}
		if len(resp.Containers) == 0 {
			break
		}
		for _, c := range resp.Containers {
			err = cli.DeleteContainer(c.Name)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (b BlobStorageClient) putSingleBlockBlob(container, name string, chunk []byte) error {
	if len(chunk) > MaxBlobBlockSize {
		return fmt.Errorf("storage: provided chunk (%d bytes) cannot fit into single-block blob (max %d bytes)", len(chunk), MaxBlobBlockSize)
	}

	uri := b.client.getEndpoint(blobServiceName, pathForBlob(container, name), url.Values{})
	headers := b.client.getStandardHeaders()
	headers["x-ms-blob-type"] = string(BlobTypeBlock)
	headers["Content-Length"] = fmt.Sprintf("%v", len(chunk))

	resp, err := b.client.exec("PUT", uri, headers, bytes.NewReader(chunk))
	if err != nil {
		return err
	}
	return checkRespCode(resp.statusCode, []int{http.StatusCreated})
}

func (s *StorageBlobSuite) TestPutAppendBlobSpecialChars(c *chk.C) {
	cli := getBlobClient(c)
	cnt := randContainer()
	c.Assert(cli.CreateContainer(cnt, ContainerAccessTypePrivate), chk.IsNil)
	defer cli.deleteContainer(cnt)

	blob := randNameWithSpecialChars(5)
	c.Assert(cli.PutAppendBlob(cnt, blob, nil), chk.IsNil)

	// Verify metadata
	props, err := cli.GetBlobProperties(cnt, blob)
	c.Assert(err, chk.IsNil)
	c.Assert(props.ContentLength, chk.Equals, int64(0))
	c.Assert(props.BlobType, chk.Equals, BlobTypeAppend)

	chunk1 := []byte(randString(1024))
	chunk2 := []byte(randString(512))

	// Append first block
	c.Assert(cli.AppendBlock(cnt, blob, chunk1, nil), chk.IsNil)

	// Verify contents
	out, err := cli.GetBlobRange(cnt, blob, fmt.Sprintf("%v-%v", 0, len(chunk1)-1), nil)
	c.Assert(err, chk.IsNil)
	defer out.Close()
	blobContents, err := ioutil.ReadAll(out)
	c.Assert(err, chk.IsNil)
	c.Assert(blobContents, chk.DeepEquals, chunk1)
	out.Close()

	// Append second block
	c.Assert(cli.AppendBlock(cnt, blob, chunk2, nil), chk.IsNil)

	// Verify contents
	out, err = cli.GetBlobRange(cnt, blob, fmt.Sprintf("%v-%v", 0, len(chunk1)+len(chunk2)-1), nil)
	c.Assert(err, chk.IsNil)
	defer out.Close()
	blobContents, err = ioutil.ReadAll(out)
	c.Assert(err, chk.IsNil)
	c.Assert(blobContents, chk.DeepEquals, append(chunk1, chunk2...))
	out.Close()
}

func randContainer() string {
	return testContainerPrefix + randString(32-len(testContainerPrefix))
}

func randString(n int) string {
	if n <= 0 {
		panic("negative number")
	}
	const alphanum = "0123456789abcdefghijklmnopqrstuvwxyz"
	var bytes = make([]byte, n)
	rand.Read(bytes)
	for i, b := range bytes {
		bytes[i] = alphanum[b%byte(len(alphanum))]
	}
	return string(bytes)
}

func randBytes(n int) []byte {
	data := make([]byte, n)
	if _, err := io.ReadFull(rand.Reader, data); err != nil {
		panic(err)
	}
	return data
}

func randName(n int) string {
	name := randString(n) + "/" + randString(n)
	return name
}

func randNameWithSpecialChars(n int) string {
	name := randString(n) + "/" + randString(n) + "-._~:?#[]@!$&'()*,;+= " + randString(n)
	return name
}
