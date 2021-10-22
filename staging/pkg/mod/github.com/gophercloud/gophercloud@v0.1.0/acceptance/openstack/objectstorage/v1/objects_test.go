// +build acceptance

package v1

import (
	"bytes"
	"strings"
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/objectstorage/v1/containers"
	"github.com/gophercloud/gophercloud/openstack/objectstorage/v1/objects"
	th "github.com/gophercloud/gophercloud/testhelper"
)

// numObjects is the number of objects to create for testing.
var numObjects = 2

func TestObjects(t *testing.T) {
	client, err := clients.NewObjectStorageV1Client()
	if err != nil {
		t.Fatalf("Unable to create client: %v", err)
	}

	// Make a slice of length numObjects to hold the random object names.
	oNames := make([]string, numObjects)
	for i := 0; i < len(oNames); i++ {
		oNames[i] = tools.RandomString("test-object-", 8)
	}

	// Create a container to hold the test objects.
	cName := tools.RandomString("test-container-", 8)
	header, err := containers.Create(client, cName, nil).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Create object headers: %+v\n", header)

	// Defer deletion of the container until after testing.
	defer func() {
		res := containers.Delete(client, cName)
		th.AssertNoErr(t, res.Err)
	}()

	// Create a slice of buffers to hold the test object content.
	oContents := make([]*bytes.Buffer, numObjects)
	for i := 0; i < numObjects; i++ {
		oContents[i] = bytes.NewBuffer([]byte(tools.RandomString("", 10)))
		createOpts := objects.CreateOpts{
			Content: oContents[i],
		}
		res := objects.Create(client, cName, oNames[i], createOpts)
		th.AssertNoErr(t, res.Err)
	}
	// Delete the objects after testing.
	defer func() {
		for i := 0; i < numObjects; i++ {
			res := objects.Delete(client, cName, oNames[i], nil)
			th.AssertNoErr(t, res.Err)
		}
	}()

	// List all created objects
	listOpts := objects.ListOpts{
		Full:   true,
		Prefix: "test-object-",
	}

	allPages, err := objects.List(client, cName, listOpts).AllPages()
	if err != nil {
		t.Fatalf("Unable to list objects: %v", err)
	}

	ons, err := objects.ExtractNames(allPages)
	if err != nil {
		t.Fatalf("Unable to extract objects: %v", err)
	}
	th.AssertEquals(t, len(ons), len(oNames))

	ois, err := objects.ExtractInfo(allPages)
	if err != nil {
		t.Fatalf("Unable to extract object info: %v", err)
	}
	th.AssertEquals(t, len(ois), len(oNames))

	// Copy the contents of one object to another.
	copyOpts := objects.CopyOpts{
		Destination: cName + "/" + oNames[1],
	}
	copyres := objects.Copy(client, cName, oNames[0], copyOpts)
	th.AssertNoErr(t, copyres.Err)

	// Download one of the objects that was created above.
	downloadres := objects.Download(client, cName, oNames[0], nil)
	th.AssertNoErr(t, downloadres.Err)

	o1Content, err := downloadres.ExtractContent()
	th.AssertNoErr(t, err)

	// Download the another object that was create above.
	downloadOpts := objects.DownloadOpts{
		Newest: true,
	}
	downloadres = objects.Download(client, cName, oNames[1], downloadOpts)
	th.AssertNoErr(t, downloadres.Err)
	o2Content, err := downloadres.ExtractContent()
	th.AssertNoErr(t, err)

	// Compare the two object's contents to test that the copy worked.
	th.AssertEquals(t, string(o2Content), string(o1Content))

	// Update an object's metadata.
	metadata := map[string]string{
		"Gophercloud-Test": "objects",
	}

	updateOpts := objects.UpdateOpts{
		Metadata: metadata,
	}
	updateres := objects.Update(client, cName, oNames[0], updateOpts)
	th.AssertNoErr(t, updateres.Err)

	// Delete the object's metadata after testing.
	defer func() {
		tempMap := make(map[string]string)
		for k := range metadata {
			tempMap[k] = ""
		}
		res := objects.Update(client, cName, oNames[0], &objects.UpdateOpts{Metadata: tempMap})
		th.AssertNoErr(t, res.Err)
	}()

	// Retrieve an object's metadata.
	getOpts := objects.GetOpts{
		Newest: true,
	}
	om, err := objects.Get(client, cName, oNames[0], getOpts).ExtractMetadata()
	th.AssertNoErr(t, err)
	for k := range metadata {
		if om[k] != metadata[strings.Title(k)] {
			t.Errorf("Expected custom metadata with key: %s", k)
			return
		}
	}
}

func TestObjectsListSubdir(t *testing.T) {
	client, err := clients.NewObjectStorageV1Client()
	if err != nil {
		t.Fatalf("Unable to create client: %v", err)
	}

	// Create a random subdirectory name.
	cSubdir1 := tools.RandomString("test-subdir-", 8)
	cSubdir2 := tools.RandomString("test-subdir-", 8)

	// Make a slice of length numObjects to hold the random object names.
	oNames1 := make([]string, numObjects)
	for i := 0; i < len(oNames1); i++ {
		oNames1[i] = cSubdir1 + "/" + tools.RandomString("test-object-", 8)
	}

	oNames2 := make([]string, numObjects)
	for i := 0; i < len(oNames2); i++ {
		oNames2[i] = cSubdir2 + "/" + tools.RandomString("test-object-", 8)
	}

	// Create a container to hold the test objects.
	cName := tools.RandomString("test-container-", 8)
	_, err = containers.Create(client, cName, nil).Extract()
	th.AssertNoErr(t, err)

	// Defer deletion of the container until after testing.
	defer func() {
		t.Logf("Deleting container %s", cName)
		res := containers.Delete(client, cName)
		th.AssertNoErr(t, res.Err)
	}()

	// Create a slice of buffers to hold the test object content.
	oContents1 := make([]*bytes.Buffer, numObjects)
	for i := 0; i < numObjects; i++ {
		oContents1[i] = bytes.NewBuffer([]byte(tools.RandomString("", 10)))
		createOpts := objects.CreateOpts{
			Content: oContents1[i],
		}
		res := objects.Create(client, cName, oNames1[i], createOpts)
		th.AssertNoErr(t, res.Err)
	}
	// Delete the objects after testing.
	defer func() {
		for i := 0; i < numObjects; i++ {
			t.Logf("Deleting object %s", oNames1[i])
			res := objects.Delete(client, cName, oNames1[i], nil)
			th.AssertNoErr(t, res.Err)
		}
	}()

	oContents2 := make([]*bytes.Buffer, numObjects)
	for i := 0; i < numObjects; i++ {
		oContents2[i] = bytes.NewBuffer([]byte(tools.RandomString("", 10)))
		createOpts := objects.CreateOpts{
			Content: oContents2[i],
		}
		res := objects.Create(client, cName, oNames2[i], createOpts)
		th.AssertNoErr(t, res.Err)
	}
	// Delete the objects after testing.
	defer func() {
		for i := 0; i < numObjects; i++ {
			t.Logf("Deleting object %s", oNames2[i])
			res := objects.Delete(client, cName, oNames2[i], nil)
			th.AssertNoErr(t, res.Err)
		}
	}()

	listOpts := objects.ListOpts{
		Full:      true,
		Delimiter: "/",
	}

	allPages, err := objects.List(client, cName, listOpts).AllPages()
	if err != nil {
		t.Fatal(err)
	}

	allObjects, err := objects.ExtractNames(allPages)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("%#v\n", allObjects)
	expected := []string{cSubdir1, cSubdir2}
	for _, e := range expected {
		var valid bool
		for _, a := range allObjects {
			if e+"/" == a {
				valid = true
			}
		}
		if !valid {
			t.Fatalf("could not find %s in results", e)
		}
	}

	listOpts = objects.ListOpts{
		Full:      true,
		Delimiter: "/",
		Prefix:    cSubdir2,
	}

	allPages, err = objects.List(client, cName, listOpts).AllPages()
	if err != nil {
		t.Fatal(err)
	}

	allObjects, err = objects.ExtractNames(allPages)
	if err != nil {
		t.Fatal(err)
	}

	th.AssertEquals(t, allObjects[0], cSubdir2+"/")
	t.Logf("%#v\n", allObjects)
}
