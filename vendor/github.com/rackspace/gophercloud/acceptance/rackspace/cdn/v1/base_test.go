// +build acceptance

package v1

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/rackspace/cdn/v1/base"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestBaseOps(t *testing.T) {
	client := newClient(t)
	t.Log("Retrieving Home Document")
	testHomeDocumentGet(t, client)

	t.Log("Pinging root URL")
	testPing(t, client)
}

func testHomeDocumentGet(t *testing.T, client *gophercloud.ServiceClient) {
	hd, err := base.Get(client).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Retrieved home document: %+v", *hd)
}

func testPing(t *testing.T, client *gophercloud.ServiceClient) {
	err := base.Ping(client).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Successfully pinged root URL")
}
