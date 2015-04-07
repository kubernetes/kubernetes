// +build acceptance rackspace objectstorage v1

package v1

import (
	"bytes"
	"testing"

	osObjects "github.com/rackspace/gophercloud/openstack/objectstorage/v1/objects"
	"github.com/rackspace/gophercloud/pagination"
	raxContainers "github.com/rackspace/gophercloud/rackspace/objectstorage/v1/containers"
	raxObjects "github.com/rackspace/gophercloud/rackspace/objectstorage/v1/objects"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestObjects(t *testing.T) {
	c, err := createClient(t, false)
	th.AssertNoErr(t, err)

	res := raxContainers.Create(c, "gophercloud-test", nil)
	th.AssertNoErr(t, res.Err)

	defer func() {
		t.Logf("Deleting container...")
		res := raxContainers.Delete(c, "gophercloud-test")
		th.AssertNoErr(t, res.Err)
	}()

	content := bytes.NewBufferString("Lewis Carroll")
	options := &osObjects.CreateOpts{ContentType: "text/plain"}
	createres := raxObjects.Create(c, "gophercloud-test", "o1", content, options)
	th.AssertNoErr(t, createres.Err)

	defer func() {
		t.Logf("Deleting object o1...")
		res := raxObjects.Delete(c, "gophercloud-test", "o1", nil)
		th.AssertNoErr(t, res.Err)
	}()

	t.Logf("Objects Info available to the currently issued token:")
	count := 0
	err = raxObjects.List(c, "gophercloud-test", &osObjects.ListOpts{Full: true}).EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page %02d ---", count)

		objects, err := raxObjects.ExtractInfo(page)
		th.AssertNoErr(t, err)

		for i, object := range objects {
			t.Logf("[%02d]      name=[%s]", i, object.Name)
			t.Logf("            content-type=[%s]", object.ContentType)
			t.Logf("            bytes=[%d]", object.Bytes)
			t.Logf("            last-modified=[%s]", object.LastModified)
			t.Logf("            hash=[%s]", object.Hash)
		}

		count++
		return true, nil
	})
	th.AssertNoErr(t, err)
	if count == 0 {
		t.Errorf("No objects listed for your current token.")
	}
	t.Logf("Container Names available to the currently issued token:")
	count = 0
	err = raxObjects.List(c, "gophercloud-test", &osObjects.ListOpts{Full: false}).EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page %02d ---", count)

		names, err := raxObjects.ExtractNames(page)
		th.AssertNoErr(t, err)

		for i, name := range names {
			t.Logf("[%02d] %s", i, name)
		}

		count++
		return true, nil
	})
	th.AssertNoErr(t, err)
	if count == 0 {
		t.Errorf("No objects listed for your current token.")
	}

	copyres := raxObjects.Copy(c, "gophercloud-test", "o1", &raxObjects.CopyOpts{Destination: "gophercloud-test/o2"})
	th.AssertNoErr(t, copyres.Err)
	defer func() {
		t.Logf("Deleting object o2...")
		res := raxObjects.Delete(c, "gophercloud-test", "o2", nil)
		th.AssertNoErr(t, res.Err)
	}()

	o1Content, err := raxObjects.Download(c, "gophercloud-test", "o1", nil).ExtractContent()
	th.AssertNoErr(t, err)
	o2Content, err := raxObjects.Download(c, "gophercloud-test", "o2", nil).ExtractContent()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, string(o2Content), string(o1Content))

	updateres := raxObjects.Update(c, "gophercloud-test", "o2", osObjects.UpdateOpts{Metadata: map[string]string{"white": "mountains"}})
	th.AssertNoErr(t, updateres.Err)
	t.Logf("Headers from Update Account request: %+v\n", updateres.Header)
	defer func() {
		res := raxObjects.Update(c, "gophercloud-test", "o2", osObjects.UpdateOpts{Metadata: map[string]string{"white": ""}})
		th.AssertNoErr(t, res.Err)
		metadata, err := raxObjects.Get(c, "gophercloud-test", "o2", nil).ExtractMetadata()
		th.AssertNoErr(t, err)
		t.Logf("Metadata from Get Account request (after update reverted): %+v\n", metadata)
		th.CheckEquals(t, "", metadata["White"])
	}()

	getres := raxObjects.Get(c, "gophercloud-test", "o2", nil)
	th.AssertNoErr(t, getres.Err)
	t.Logf("Headers from Get Account request (after update): %+v\n", getres.Header)
	metadata, err := getres.ExtractMetadata()
	th.AssertNoErr(t, err)
	t.Logf("Metadata from Get Account request (after update): %+v\n", metadata)
	th.CheckEquals(t, "mountains", metadata["White"])

	createTempURLOpts := osObjects.CreateTempURLOpts{
		Method: osObjects.GET,
		TTL:    600,
	}
	tempURL, err := raxObjects.CreateTempURL(c, "gophercloud-test", "o1", createTempURLOpts)
	th.AssertNoErr(t, err)
	t.Logf("TempURL for object (%s): %s", "o1", tempURL)
}
