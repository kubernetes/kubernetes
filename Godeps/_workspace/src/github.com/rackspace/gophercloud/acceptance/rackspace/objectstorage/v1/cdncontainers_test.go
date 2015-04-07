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
	enableRes := raxCDNContainers.Enable(raxCDNClient, "gophercloud-test", raxCDNContainers.EnableOpts{CDNEnabled: true, TTL: 900})
	t.Logf("Header map from Enable CDN Container request: %+v\n", enableRes.Header)
	enableHeader, err := enableRes.Extract()
	th.AssertNoErr(t, err)
	t.Logf("Headers from Enable CDN Container request: %+v\n", enableHeader)

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

	updateOpts := raxCDNContainers.UpdateOpts{XCDNEnabled: raxCDNContainers.Disabled, XLogRetention: raxCDNContainers.Enabled}
	updateHeader, err := raxCDNContainers.Update(raxCDNClient, "gophercloud-test", updateOpts).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Headers from Update CDN Container request: %+v\n", updateHeader)

	getRes := raxCDNContainers.Get(raxCDNClient, "gophercloud-test")
	getHeader, err := getRes.Extract()
	th.AssertNoErr(t, err)
	t.Logf("Headers from Get CDN Container request (after update): %+v\n", getHeader)
	metadata, err := getRes.ExtractMetadata()
	t.Logf("Metadata from Get CDN Container request (after update): %+v\n", metadata)
}
