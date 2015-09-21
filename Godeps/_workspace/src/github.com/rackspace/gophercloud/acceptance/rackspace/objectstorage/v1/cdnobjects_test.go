// +build acceptance rackspace objectstorage v1

package v1

import (
	"bytes"
	"testing"

	raxCDNContainers "github.com/rackspace/gophercloud/rackspace/objectstorage/v1/cdncontainers"
	raxCDNObjects "github.com/rackspace/gophercloud/rackspace/objectstorage/v1/cdnobjects"
	raxContainers "github.com/rackspace/gophercloud/rackspace/objectstorage/v1/containers"
	raxObjects "github.com/rackspace/gophercloud/rackspace/objectstorage/v1/objects"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestCDNObjects(t *testing.T) {
	raxClient, err := createClient(t, false)
	th.AssertNoErr(t, err)

	createContResult := raxContainers.Create(raxClient, "gophercloud-test", nil)
	th.AssertNoErr(t, createContResult.Err)
	t.Logf("Headers from Create Container request: %+v\n", createContResult.Header)
	defer func() {
		deleteResult := raxContainers.Delete(raxClient, "gophercloud-test")
		th.AssertNoErr(t, deleteResult.Err)
	}()

	header, err := raxObjects.Create(raxClient, "gophercloud-test", "test-object", bytes.NewBufferString("gophercloud cdn test"), nil).ExtractHeader()
	th.AssertNoErr(t, err)
	t.Logf("Headers from Create Object request: %+v\n", header)
	defer func() {
		deleteResult := raxObjects.Delete(raxClient, "gophercloud-test", "test-object", nil)
		th.AssertNoErr(t, deleteResult.Err)
	}()

	raxCDNClient, err := createClient(t, true)
	th.AssertNoErr(t, err)

	enableHeader, err := raxCDNContainers.Enable(raxCDNClient, "gophercloud-test", raxCDNContainers.EnableOpts{CDNEnabled: true, TTL: 900}).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Headers from Enable CDN Container request: %+v\n", enableHeader)

	objCDNURL, err := raxCDNObjects.CDNURL(raxCDNClient, "gophercloud-test", "test-object")
	th.AssertNoErr(t, err)
	t.Logf("%s CDN URL: %s\n", "test_object", objCDNURL)

	deleteHeader, err := raxCDNObjects.Delete(raxCDNClient, "gophercloud-test", "test-object", nil).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Headers from Delete CDN Object request: %+v\n", deleteHeader)
}
