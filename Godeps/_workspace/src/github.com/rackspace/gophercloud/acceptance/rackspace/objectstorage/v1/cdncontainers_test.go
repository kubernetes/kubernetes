// +build acceptance rackspace objectstorage v1

package v1

import (
	"testing"

	osContainers "github.com/rackspace/gophercloud/openstack/objectstorage/v1/containers"
	"github.com/rackspace/gophercloud/pagination"
	raxCDNContainers "github.com/rackspace/gophercloud/rackspace/objectstorage/v1/cdncontainers"
	raxContainers "github.com/rackspace/gophercloud/rackspace/objectstorage/v1/containers"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestCDNContainers(t *testing.T) {
	raxClient, err := createClient(t, false)
	th.AssertNoErr(t, err)

	createres := raxContainers.Create(raxClient, "gophercloud-test", nil)
	th.AssertNoErr(t, createres.Err)
	t.Logf("Headers from Create Container request: %+v\n", createres.Header)
	defer func() {
		res := raxContainers.Delete(raxClient, "gophercloud-test")
		th.AssertNoErr(t, res.Err)
	}()

	raxCDNClient, err := createClient(t, true)
	th.AssertNoErr(t, err)

	r := raxCDNContainers.Enable(raxCDNClient, "gophercloud-test", raxCDNContainers.EnableOpts{CDNEnabled: true, TTL: 900})
	th.AssertNoErr(t, r.Err)
	t.Logf("Headers from Enable CDN Container request: %+v\n", r.Header)

	t.Logf("Container Names available to the currently issued token:")
	count := 0
	err = raxCDNContainers.List(raxCDNClient, &osContainers.ListOpts{Full: false}).EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page %02d ---", count)

		names, err := raxCDNContainers.ExtractNames(page)
		th.AssertNoErr(t, err)

		for i, name := range names {
			t.Logf("[%02d] %s", i, name)
		}

		count++
		return true, nil
	})
	th.AssertNoErr(t, err)
	if count == 0 {
		t.Errorf("No CDN containers listed for your current token.")
	}

	updateres := raxCDNContainers.Update(raxCDNClient, "gophercloud-test", raxCDNContainers.UpdateOpts{CDNEnabled: false})
	th.AssertNoErr(t, updateres.Err)
	t.Logf("Headers from Update CDN Container request: %+v\n", updateres.Header)

	metadata, err := raxCDNContainers.Get(raxCDNClient, "gophercloud-test").ExtractMetadata()
	th.AssertNoErr(t, err)
	t.Logf("Headers from Get CDN Container request (after update): %+v\n", metadata)
}
