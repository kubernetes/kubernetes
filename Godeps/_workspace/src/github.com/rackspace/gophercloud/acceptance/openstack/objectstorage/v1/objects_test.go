// +build acceptance

package v1

import (
	"bytes"
	"strings"
	"testing"

	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/openstack/objectstorage/v1/containers"
	"github.com/rackspace/gophercloud/openstack/objectstorage/v1/objects"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

// numObjects is the number of objects to create for testing.
var numObjects = 2

func TestObjects(t *testing.T) {
	// Create a provider client for executing the HTTP request.
	// See common.go for more information.
	client := newClient(t)

	// Make a slice of length numObjects to hold the random object names.
	oNames := make([]string, numObjects)
	for i := 0; i < len(oNames); i++ {
		oNames[i] = tools.RandomString("test-object-", 8)
	}

	// Create a container to hold the test objects.
	cName := tools.RandomString("test-container-", 8)
	header, err := containers.Create(client, cName, nil).ExtractHeader()
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
		res := objects.Create(client, cName, oNames[i], oContents[i], nil)
		th.AssertNoErr(t, res.Err)
	}
	// Delete the objects after testing.
	defer func() {
		for i := 0; i < numObjects; i++ {
			res := objects.Delete(client, cName, oNames[i], nil)
			th.AssertNoErr(t, res.Err)
		}
	}()

	ons := make([]string, 0, len(oNames))
	err = objects.List(client, cName, &objects.ListOpts{Full: false, Prefix: "test-object-"}).EachPage(func(page pagination.Page) (bool, error) {
		names, err := objects.ExtractNames(page)
		th.AssertNoErr(t, err)
		ons = append(ons, names...)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.AssertEquals(t, len(ons), len(oNames))

	ois := make([]objects.Object, 0, len(oNames))
	err = objects.List(client, cName, &objects.ListOpts{Full: true, Prefix: "test-object-"}).EachPage(func(page pagination.Page) (bool, error) {
		info, err := objects.ExtractInfo(page)
		th.AssertNoErr(t, err)

		ois = append(ois, info...)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.AssertEquals(t, len(ois), len(oNames))

	// Copy the contents of one object to another.
	copyres := objects.Copy(client, cName, oNames[0], &objects.CopyOpts{Destination: cName + "/" + oNames[1]})
	th.AssertNoErr(t, copyres.Err)

	// Download one of the objects that was created above.
	o1Content, err := objects.Download(client, cName, oNames[0], nil).ExtractContent()
	th.AssertNoErr(t, err)

	// Download the another object that was create above.
	o2Content, err := objects.Download(client, cName, oNames[1], nil).ExtractContent()
	th.AssertNoErr(t, err)

	// Compare the two object's contents to test that the copy worked.
	th.AssertEquals(t, string(o2Content), string(o1Content))

	// Update an object's metadata.
	updateres := objects.Update(client, cName, oNames[0], &objects.UpdateOpts{Metadata: metadata})
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
