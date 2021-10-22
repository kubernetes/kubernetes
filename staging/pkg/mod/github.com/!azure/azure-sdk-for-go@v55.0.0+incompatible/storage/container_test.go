package storage

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"net/url"
	"sort"
	"strconv"
	"time"

	"github.com/Azure/go-autorest/autorest/azure"
	chk "gopkg.in/check.v1"
)

type ContainerSuite struct{}

var _ = chk.Suite(&ContainerSuite{})

func (s *ContainerSuite) Test_containerBuildPath(c *chk.C) {
	cli := getBlobClient(c)
	cnt := cli.GetContainerReference("lol")
	c.Assert(cnt.buildPath(), chk.Equals, "/lol")
}

func (s *ContainerSuite) TestListContainersPagination(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()
	cli.deleteTestContainers(c)

	const n = 5
	const pageSize = 2

	cntNames := []string{}
	for i := 0; i < n; i++ {
		cntNames = append(cntNames, containerName(c, strconv.Itoa(i)))
	}
	sort.Strings(cntNames)

	// Create test containers
	created := []*Container{}
	for i := 0; i < n; i++ {
		cnt := cli.GetContainerReference(cntNames[i])
		c.Assert(cnt.Create(nil), chk.IsNil)
		created = append(created, cnt)
		cnt.Metadata = map[string]string{
			"hello": "world",
			"name":  cnt.Name,
		}
		c.Assert(cnt.SetMetadata(nil), chk.IsNil)
		defer cnt.Delete(nil)
	}

	// Paginate results
	seen := []Container{}
	marker := ""
	for {
		resp, err := cli.ListContainers(ListContainersParameters{
			MaxResults: pageSize,
			Marker:     marker,
			Include:    "metadata",
		})

		c.Assert(err, chk.IsNil)

		if len(resp.Containers) > pageSize {
			c.Fatalf("Got a bigger page. Expected: %d, got: %d", pageSize, len(resp.Containers))
		}

		for _, c := range resp.Containers {
			seen = append(seen, c)
		}

		marker = resp.NextMarker
		if marker == "" || len(resp.Containers) == 0 {
			break
		}
	}

	for i := range created {
		c.Assert(seen[i].Name, chk.DeepEquals, created[i].Name)
		c.Assert(seen[i].Metadata, chk.DeepEquals, created[i].Metadata)
	}
}

func (s *ContainerSuite) TestContainerExists(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// Container does not exist
	cnt1 := cli.GetContainerReference(containerName(c, "1"))
	ok, err := cnt1.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)

	// Container exists
	cnt2 := cli.GetContainerReference(containerName(c, "2"))
	err = cnt2.Create(nil)
	defer cnt2.Delete(nil)
	c.Assert(err, chk.IsNil)
	ok, err = cnt2.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)

	// Service SASURI test (funcs should fail, service SAS has not enough permissions)
	sasuriOptions := ContainerSASOptions{}
	sasuriOptions.Expiry = fixedTime
	sasuriOptions.Read = true
	sasuriOptions.Add = true
	sasuriOptions.Create = true
	sasuriOptions.Write = true
	sasuriOptions.Delete = true
	sasuriOptions.List = true

	sasuriString1, err := cnt1.GetSASURI(sasuriOptions)
	c.Assert(err, chk.IsNil)
	sasuri1, err := url.Parse(sasuriString1)
	c.Assert(err, chk.IsNil)
	cntServiceSAS1, err := GetContainerReferenceFromSASURI(*sasuri1)
	c.Assert(err, chk.IsNil)
	cntServiceSAS1.Client().HTTPClient = cli.client.HTTPClient

	ok, err = cntServiceSAS1.Exists()
	c.Assert(err, chk.NotNil)
	c.Assert(ok, chk.Equals, false)

	sasuriString2, err := cnt2.GetSASURI(sasuriOptions)
	c.Assert(err, chk.IsNil)
	sasuri2, err := url.Parse(sasuriString2)
	c.Assert(err, chk.IsNil)
	cntServiceSAS2, err := GetContainerReferenceFromSASURI(*sasuri2)
	c.Assert(err, chk.IsNil)
	cntServiceSAS2.Client().HTTPClient = cli.client.HTTPClient

	ok, err = cntServiceSAS2.Exists()
	c.Assert(err, chk.NotNil)
	c.Assert(ok, chk.Equals, false)

	// Account SASURI test
	token, err := cli.client.GetAccountSASToken(accountSASOptions)
	c.Assert(err, chk.IsNil)
	SAScli := NewAccountSASClient(cli.client.accountName, token, azure.PublicCloud).GetBlobService()

	cntAccountSAS1 := SAScli.GetContainerReference(cnt1.Name)
	cntAccountSAS1.Client().HTTPClient = cli.client.HTTPClient
	ok, err = cntAccountSAS1.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)

	cntAccountSAS2 := SAScli.GetContainerReference(cnt2.Name)
	cntAccountSAS2.Client().HTTPClient = cli.client.HTTPClient
	ok, err = cntAccountSAS2.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)
}

func (s *ContainerSuite) TestCreateContainerDeleteContainer(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()
	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	c.Assert(cnt.Delete(nil), chk.IsNil)
}

func (s *ContainerSuite) TestCreateContainerIfNotExists(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// Create non exisiting container
	cnt := cli.GetContainerReference(containerName(c))
	ok, err := cnt.CreateIfNotExists(nil)
	defer cnt.Delete(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)

}

func (s *ContainerSuite) TestCreateContainerIfExists(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	cnt.Create(nil)
	defer cnt.Delete(nil)

	// Try to create already exisiting container
	ok, err := cnt.CreateIfNotExists(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)
}

func (s *ContainerSuite) TestDeleteContainerIfExists(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// Nonexisting container
	cnt1 := cli.GetContainerReference(containerName(c, "1"))
	ok, err := cnt1.Exists()
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)
	ok, err = cnt1.DeleteIfExists(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, false)

	// Existing container
	cnt2 := cli.GetContainerReference(containerName(c, "2"))
	c.Assert(cnt2.Create(nil), chk.IsNil)
	ok, err = cnt2.DeleteIfExists(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(ok, chk.Equals, true)
}

func (s *ContainerSuite) TestListBlobsPagination(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()
	cnt := cli.GetContainerReference(containerName(c))

	err := cnt.Create(nil)
	defer cnt.Delete(nil)
	c.Assert(err, chk.IsNil)

	blobs := []string{}
	types := []BlobType{}
	const n = 5
	const pageSize = 2
	for i := 0; i < n; i++ {
		name := blobName(c, strconv.Itoa(i))
		b := cnt.GetBlobReference(name)
		c.Assert(b.putSingleBlockBlob([]byte("Hello, world!")), chk.IsNil)
		blobs = append(blobs, name)
		types = append(types, b.Properties.BlobType)
	}
	sort.Strings(blobs)

	listBlobsPagination(c, cnt, pageSize, blobs, types)

	// Service SAS test
	sasuriOptions := ContainerSASOptions{}
	sasuriOptions.Expiry = fixedTime
	sasuriOptions.Read = true
	sasuriOptions.Add = true
	sasuriOptions.Create = true
	sasuriOptions.Write = true
	sasuriOptions.Delete = true
	sasuriOptions.List = true

	sasuriString, err := cnt.GetSASURI(sasuriOptions)
	c.Assert(err, chk.IsNil)
	sasuri, err := url.Parse(sasuriString)
	c.Assert(err, chk.IsNil)
	cntServiceSAS, err := GetContainerReferenceFromSASURI(*sasuri)
	c.Assert(err, chk.IsNil)
	cntServiceSAS.Client().HTTPClient = cli.client.HTTPClient

	listBlobsPagination(c, cntServiceSAS, pageSize, blobs, types)

	// Account SAS test
	token, err := cli.client.GetAccountSASToken(accountSASOptions)
	c.Assert(err, chk.IsNil)
	SAScli := NewAccountSASClient(cli.client.accountName, token, azure.PublicCloud).GetBlobService()

	cntAccountSAS := SAScli.GetContainerReference(cnt.Name)
	cntAccountSAS.Client().HTTPClient = cli.client.HTTPClient

	listBlobsPagination(c, cntAccountSAS, pageSize, blobs, types)
}

func listBlobsPagination(c *chk.C, cnt *Container, pageSize uint, blobs []string, types []BlobType) {
	// Paginate
	seen := []string{}
	seenTypes := []BlobType{}
	marker := ""
	for {
		resp, err := cnt.ListBlobs(ListBlobsParameters{
			MaxResults: pageSize,
			Marker:     marker})
		c.Assert(err, chk.IsNil)

		for _, b := range resp.Blobs {
			seen = append(seen, b.Name)
			seenTypes = append(seenTypes, b.Properties.BlobType)
			c.Assert(b.Container, chk.Equals, cnt)
		}

		marker = resp.NextMarker
		if marker == "" || len(resp.Blobs) == 0 {
			break
		}
	}

	// Compare
	c.Assert(seen, chk.DeepEquals, blobs)
	c.Assert(seenTypes, chk.DeepEquals, types)
}

// listBlobsAsFiles is a helper function to list blobs as "folders" and "files".
func listBlobsAsFiles(cli BlobStorageClient, cnt *Container, parentDir string) (folders []string, files []string, err error) {
	var blobParams ListBlobsParameters
	var blobListResponse BlobListResponse

	// Top level "folders"
	blobParams = ListBlobsParameters{
		Delimiter: "/",
		Prefix:    parentDir,
	}

	blobListResponse, err = cnt.ListBlobs(blobParams)
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
//  - The "folder names" will be "absolute paths", e.g. listing things under "/usr/"
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
func (s *ContainerSuite) TestListBlobsTraversal(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

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
		b := cnt.GetBlobReference(blobName)
		err := b.CreateBlockBlob(nil)
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

func (s *ContainerSuite) TestListBlobsWithMetadata(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	expectMeta := make(map[string]BlobMetadata)

	// Put 4 blobs with metadata
	for i := 0; i < 4; i++ {
		name := blobName(c, strconv.Itoa(i))
		b := cnt.GetBlobReference(name)
		c.Assert(b.putSingleBlockBlob([]byte("Hello, world!")), chk.IsNil)
		b.Metadata = BlobMetadata{
			"Lol":      name,
			"Rofl_BAZ": "Waz Qux",
		}
		c.Assert(b.SetMetadata(nil), chk.IsNil)
		expectMeta[name] = BlobMetadata{
			"lol":      name,
			"rofl_baz": "Waz Qux",
		}
		_, err := b.CreateSnapshot(nil)
		c.Assert(err, chk.IsNil)
	}

	// Put one more blob with no metadata
	b := cnt.GetBlobReference(blobName(c, "nometa"))
	c.Assert(b.putSingleBlockBlob([]byte("Hello, world!")), chk.IsNil)
	expectMeta[b.Name] = nil

	// Get ListBlobs with include: metadata and snapshots
	resp, err := cnt.ListBlobs(ListBlobsParameters{
		Include: &IncludeBlobDataset{
			Metadata:  true,
			Snapshots: true,
		},
	})
	c.Assert(err, chk.IsNil)

	originalBlobs := make(map[string]Blob)
	snapshotBlobs := make(map[string]Blob)
	for _, v := range resp.Blobs {
		if v.Snapshot == (time.Time{}) {
			originalBlobs[v.Name] = v
		} else {
			snapshotBlobs[v.Name] = v

		}
	}
	c.Assert(originalBlobs, chk.HasLen, 5)
	c.Assert(snapshotBlobs, chk.HasLen, 4)

	// Verify the metadata is as expected
	for name := range expectMeta {
		c.Check(originalBlobs[name].Metadata, chk.DeepEquals, expectMeta[name])
		c.Check(snapshotBlobs[name].Metadata, chk.DeepEquals, expectMeta[name])
	}
}

func appendContainerPermission(perms ContainerPermissions, accessType ContainerAccessType,
	ID string, start time.Time, expiry time.Time,
	canRead bool, canWrite bool, canDelete bool) ContainerPermissions {

	perms.AccessType = accessType

	if ID != "" {
		capd := ContainerAccessPolicy{
			ID:         ID,
			StartTime:  start,
			ExpiryTime: expiry,
			CanRead:    canRead,
			CanWrite:   canWrite,
			CanDelete:  canDelete,
		}
		perms.AccessPolicies = append(perms.AccessPolicies, capd)
	}
	return perms
}

func (s *ContainerSuite) TestSetContainerPermissionsWithTimeoutSuccessfully(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	perms := ContainerPermissions{}
	perms = appendContainerPermission(perms, ContainerAccessTypeBlob, "GolangRocksOnAzure", fixedTime, fixedTime.Add(10*time.Hour), true, true, true)

	options := SetContainerPermissionOptions{
		Timeout: 30,
	}
	err := cnt.SetPermissions(perms, &options)
	c.Assert(err, chk.IsNil)
}

func (s *ContainerSuite) TestSetContainerPermissionsSuccessfully(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	perms := ContainerPermissions{}
	perms = appendContainerPermission(perms, ContainerAccessTypeBlob, "GolangRocksOnAzure", fixedTime, fixedTime.Add(10*time.Hour), true, true, true)

	err := cnt.SetPermissions(perms, nil)
	c.Assert(err, chk.IsNil)
}

func (s *ContainerSuite) TestSetThenGetContainerPermissionsSuccessfully(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.delete(nil)

	perms := ContainerPermissions{}
	perms = appendContainerPermission(perms, ContainerAccessTypeBlob, "AutoRestIsSuperCool", fixedTime, fixedTime.Add(10*time.Hour), true, true, true)
	perms = appendContainerPermission(perms, ContainerAccessTypeBlob, "GolangRocksOnAzure", fixedTime.Add(20*time.Hour), fixedTime.Add(30*time.Hour), true, false, false)
	c.Assert(perms.AccessPolicies, chk.HasLen, 2)

	err := cnt.SetPermissions(perms, nil)
	c.Assert(err, chk.IsNil)

	newPerms, err := cnt.GetPermissions(nil)
	c.Assert(err, chk.IsNil)

	// check container permissions itself.
	c.Assert(newPerms.AccessType, chk.Equals, perms.AccessType)

	// fixedTime check policy set.
	c.Assert(newPerms.AccessPolicies, chk.HasLen, 2)

	for i := range perms.AccessPolicies {
		c.Assert(newPerms.AccessPolicies[i].ID, chk.Equals, perms.AccessPolicies[i].ID)

		// test timestamps down the second
		// rounding start/expiry time original perms since the returned perms would have been rounded.
		// so need rounded vs rounded.
		c.Assert(newPerms.AccessPolicies[i].StartTime.UTC().Round(time.Second).Format(time.RFC1123),
			chk.Equals, perms.AccessPolicies[i].StartTime.UTC().Round(time.Second).Format(time.RFC1123))

		c.Assert(newPerms.AccessPolicies[i].ExpiryTime.UTC().Round(time.Second).Format(time.RFC1123),
			chk.Equals, perms.AccessPolicies[i].ExpiryTime.UTC().Round(time.Second).Format(time.RFC1123))

		c.Assert(newPerms.AccessPolicies[i].CanRead, chk.Equals, perms.AccessPolicies[i].CanRead)
		c.Assert(newPerms.AccessPolicies[i].CanWrite, chk.Equals, perms.AccessPolicies[i].CanWrite)
		c.Assert(newPerms.AccessPolicies[i].CanDelete, chk.Equals, perms.AccessPolicies[i].CanDelete)
	}
}

func (s *ContainerSuite) TestSetContainerPermissionsOnlySuccessfully(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	perms := ContainerPermissions{}
	perms = appendContainerPermission(perms, ContainerAccessTypeBlob, "GolangRocksOnAzure", fixedTime, fixedTime.Add(10*time.Hour), true, true, true)

	err := cnt.SetPermissions(perms, nil)
	c.Assert(err, chk.IsNil)
}

func (s *ContainerSuite) TestSetThenGetContainerPermissionsOnlySuccessfully(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	cnt := cli.GetContainerReference(containerName(c))
	c.Assert(cnt.Create(nil), chk.IsNil)
	defer cnt.Delete(nil)

	perms := ContainerPermissions{}
	perms = appendContainerPermission(perms, ContainerAccessTypeBlob, "", fixedTime, fixedTime.Add(10*time.Hour), true, true, true)

	err := cnt.SetPermissions(perms, nil)
	c.Assert(err, chk.IsNil)

	newPerms, err := cnt.GetPermissions(nil)
	c.Assert(err, chk.IsNil)

	// check container permissions itself.
	c.Assert(newPerms.AccessType, chk.Equals, perms.AccessType)

	// fixedTime check there are NO policies set
	c.Assert(newPerms.AccessPolicies, chk.HasLen, 0)
}

func (s *ContainerSuite) TestGetAndSetContainerMetadata(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// Get empty metadata
	cnt1 := cli.GetContainerReference(containerName(c, "1"))
	c.Assert(cnt1.Create(nil), chk.IsNil)
	defer cnt1.Delete(nil)

	err := cnt1.GetMetadata(nil)
	c.Assert(err, chk.IsNil)
	c.Assert(cnt1.Metadata, chk.HasLen, 0)

	// Get and set the metadata
	cnt2 := cli.GetContainerReference(containerName(c, "2"))
	c.Assert(cnt2.Create(nil), chk.IsNil)
	defer cnt2.Delete(nil)

	metaPut := map[string]string{
		"lol":      "rofl",
		"rofl_baz": "waz qux",
	}
	cnt2.Metadata = metaPut

	err = cnt2.SetMetadata(nil)
	c.Assert(err, chk.IsNil)

	err = cnt2.GetMetadata(nil)
	c.Assert(err, chk.IsNil)
	c.Check(cnt2.Metadata, chk.DeepEquals, metaPut)
}

func (s *ContainerSuite) TestGetContainerProperties(c *chk.C) {
	cli := getBlobClient(c)
	rec := cli.client.appendRecorder(c)
	defer rec.Stop()

	// Get empty metadata
	cnt1 := cli.GetContainerReference(containerName(c, "1"))
	c.Assert(cnt1.Create(nil), chk.IsNil)
	defer cnt1.Delete(nil)

	// should be empty until we get properties
	c.Assert(cnt1.Properties.Etag, chk.HasLen, 0)

	err := cnt1.GetProperties()
	c.Assert(err, chk.IsNil)
	c.Assert(cnt1.Properties.Etag, chk.Equals, `"0x8D9001BBA6C4080"`)
	c.Assert(cnt1.Properties.PublicAccess, chk.Equals, ContainerAccessType(""))
}

func (cli *BlobStorageClient) deleteTestContainers(c *chk.C) error {
	for {
		resp, err := cli.ListContainers(ListContainersParameters{})
		if err != nil {
			return err
		}
		if len(resp.Containers) == 0 {
			break
		}
		for _, c := range resp.Containers {
			err = c.Delete(nil)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func containerName(c *chk.C, extras ...string) string {
	return nameGenerator(32, "", alphanum, c, extras)
}
