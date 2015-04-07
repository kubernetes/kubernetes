// +build acceptance

package v1

import (
	"strings"
	"testing"

	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/openstack/objectstorage/v1/containers"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

// numContainers is the number of containers to create for testing.
var numContainers = 2

func TestContainers(t *testing.T) {
	// Create a new client to execute the HTTP requests. See common.go for newClient body.
	client := newClient(t)

	// Create a slice of random container names.
	cNames := make([]string, numContainers)
	for i := 0; i < numContainers; i++ {
		cNames[i] = tools.RandomString("gophercloud-test-container-", 8)
	}

	// Create numContainers containers.
	for i := 0; i < len(cNames); i++ {
		res := containers.Create(client, cNames[i], nil)
		th.AssertNoErr(t, res.Err)
	}
	// Delete the numContainers containers after function completion.
	defer func() {
		for i := 0; i < len(cNames); i++ {
			res := containers.Delete(client, cNames[i])
			th.AssertNoErr(t, res.Err)
		}
	}()

	// List the numContainer names that were just created. To just list those,
	// the 'prefix' parameter is used.
	err := containers.List(client, &containers.ListOpts{Full: true, Prefix: "gophercloud-test-container-"}).EachPage(func(page pagination.Page) (bool, error) {
		containerList, err := containers.ExtractInfo(page)
		th.AssertNoErr(t, err)

		for _, n := range containerList {
			t.Logf("Container: Name [%s] Count [%d] Bytes [%d]",
				n.Name, n.Count, n.Bytes)
		}

		return true, nil
	})
	th.AssertNoErr(t, err)

	// List the info for the numContainer containers that were created.
	err = containers.List(client, &containers.ListOpts{Full: false, Prefix: "gophercloud-test-container-"}).EachPage(func(page pagination.Page) (bool, error) {
		containerList, err := containers.ExtractNames(page)
		th.AssertNoErr(t, err)
		for _, n := range containerList {
			t.Logf("Container: Name [%s]", n)
		}

		return true, nil
	})
	th.AssertNoErr(t, err)

	// Update one of the numContainer container metadata.
	updateres := containers.Update(client, cNames[0], &containers.UpdateOpts{Metadata: metadata})
	th.AssertNoErr(t, updateres.Err)
	// After the tests are done, delete the metadata that was set.
	defer func() {
		tempMap := make(map[string]string)
		for k := range metadata {
			tempMap[k] = ""
		}
		res := containers.Update(client, cNames[0], &containers.UpdateOpts{Metadata: tempMap})
		th.AssertNoErr(t, res.Err)
	}()

	// Retrieve a container's metadata.
	cm, err := containers.Get(client, cNames[0]).ExtractMetadata()
	th.AssertNoErr(t, err)
	for k := range metadata {
		if cm[k] != metadata[strings.Title(k)] {
			t.Errorf("Expected custom metadata with key: %s", k)
		}
	}
}

func TestListAllContainers(t *testing.T) {
	// Create a new client to execute the HTTP requests. See common.go for newClient body.
	client := newClient(t)

	numContainers := 20

	// Create a slice of random container names.
	cNames := make([]string, numContainers)
	for i := 0; i < numContainers; i++ {
		cNames[i] = tools.RandomString("gophercloud-test-container-", 8)
	}

	// Create numContainers containers.
	for i := 0; i < len(cNames); i++ {
		res := containers.Create(client, cNames[i], nil)
		th.AssertNoErr(t, res.Err)
	}
	// Delete the numContainers containers after function completion.
	defer func() {
		for i := 0; i < len(cNames); i++ {
			res := containers.Delete(client, cNames[i])
			th.AssertNoErr(t, res.Err)
		}
	}()

	// List all the numContainer names that were just created. To just list those,
	// the 'prefix' parameter is used.
	allPages, err := containers.List(client, &containers.ListOpts{Full: true, Limit: 5, Prefix: "gophercloud-test-container-"}).AllPages()
	th.AssertNoErr(t, err)
	containerInfoList, err := containers.ExtractInfo(allPages)
	th.AssertNoErr(t, err)
	for _, n := range containerInfoList {
		t.Logf("Container: Name [%s] Count [%d] Bytes [%d]",
			n.Name, n.Count, n.Bytes)
	}
	th.AssertEquals(t, numContainers, len(containerInfoList))

	// List the info for all the numContainer containers that were created.
	allPages, err = containers.List(client, &containers.ListOpts{Full: false, Limit: 2, Prefix: "gophercloud-test-container-"}).AllPages()
	th.AssertNoErr(t, err)
	containerNamesList, err := containers.ExtractNames(allPages)
	th.AssertNoErr(t, err)
	for _, n := range containerNamesList {
		t.Logf("Container: Name [%s]", n)
	}
	th.AssertEquals(t, numContainers, len(containerNamesList))
}
