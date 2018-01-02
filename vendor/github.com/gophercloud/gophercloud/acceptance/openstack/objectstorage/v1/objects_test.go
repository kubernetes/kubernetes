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
	downloadres = objects.Download(client, cName, oNames[1], nil)
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
	om, err := objects.Get(client, cName, oNames[0], nil).ExtractMetadata()
	th.AssertNoErr(t, err)
	for k := range metadata {
		if om[k] != metadata[strings.Title(k)] {
			t.Errorf("Expected custom metadata with key: %s", k)
			return
		}
	}
}
