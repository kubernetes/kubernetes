// +build acceptance

package v3

import (
	"fmt"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/rackspace/rackconnect/v3/publicips"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestPublicIPs(t *testing.T) {
	c := newClient(t)
	ipID := testListIPs(t, c)
	sID := testGetIP(t, c, ipID)
	testListIPsForServer(t, c, sID)
}

func testListIPs(t *testing.T, c *gophercloud.ServiceClient) string {
	allPages, err := publicips.List(c).AllPages()
	th.AssertNoErr(t, err)
	allip, err := publicips.ExtractPublicIPs(allPages)
	fmt.Printf("Listing all public IPs: %+v\n\n", allip)
	var ipID string
	if len(allip) > 0 {
		ipID = allip[0].ID
	}
	return ipID
}

func testGetIP(t *testing.T, c *gophercloud.ServiceClient, ipID string) string {
	ip, err := publicips.Get(c, ipID).Extract()
	th.AssertNoErr(t, err)
	fmt.Printf("Retrieved public IP (%s): %+v\n\n", ipID, ip)
	return ip.CloudServer.ID
}

func testListIPsForServer(t *testing.T, c *gophercloud.ServiceClient, sID string) {
	allPages, err := publicips.ListForServer(c, sID).AllPages()
	th.AssertNoErr(t, err)
	allip, err := publicips.ExtractPublicIPs(allPages)
	fmt.Printf("Listing all public IPs for server (%s): %+v\n\n", sID, allip)
}
