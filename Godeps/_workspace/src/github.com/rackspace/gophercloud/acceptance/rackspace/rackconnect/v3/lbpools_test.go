// +build acceptance

package v3

import (
	"fmt"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/rackspace/rackconnect/v3/lbpools"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestLBPools(t *testing.T) {
	c := newClient(t)
	pID := testListPools(t, c)
	testGetPools(t, c, pID)
	nID := testListNodes(t, c, pID)
	testListNodeDetails(t, c, pID)
	testGetNode(t, c, pID, nID)
	testGetNodeDetails(t, c, pID, nID)
}

func testListPools(t *testing.T, c *gophercloud.ServiceClient) string {
	allPages, err := lbpools.List(c).AllPages()
	th.AssertNoErr(t, err)
	allp, err := lbpools.ExtractPools(allPages)
	fmt.Printf("Listing all LB pools: %+v\n\n", allp)
	var pID string
	if len(allp) > 0 {
		pID = allp[0].ID
	}
	return pID
}

func testGetPools(t *testing.T, c *gophercloud.ServiceClient, pID string) {
	p, err := lbpools.Get(c, pID).Extract()
	th.AssertNoErr(t, err)
	fmt.Printf("Retrieved LB pool: %+v\n\n", p)
}

func testListNodes(t *testing.T, c *gophercloud.ServiceClient, pID string) string {
	allPages, err := lbpools.ListNodes(c, pID).AllPages()
	th.AssertNoErr(t, err)
	alln, err := lbpools.ExtractNodes(allPages)
	fmt.Printf("Listing all LB pool nodes for pool (%s): %+v\n\n", pID, alln)
	var nID string
	if len(alln) > 0 {
		nID = alln[0].ID
	}
	return nID
}

func testListNodeDetails(t *testing.T, c *gophercloud.ServiceClient, pID string) {
	allPages, err := lbpools.ListNodesDetails(c, pID).AllPages()
	th.AssertNoErr(t, err)
	alln, err := lbpools.ExtractNodesDetails(allPages)
	fmt.Printf("Listing all LB pool nodes details for pool (%s): %+v\n\n", pID, alln)
}

func testGetNode(t *testing.T, c *gophercloud.ServiceClient, pID, nID string) {
	n, err := lbpools.GetNode(c, pID, nID).Extract()
	th.AssertNoErr(t, err)
	fmt.Printf("Retrieved LB node: %+v\n\n", n)
}

func testGetNodeDetails(t *testing.T, c *gophercloud.ServiceClient, pID, nID string) {
	n, err := lbpools.GetNodeDetails(c, pID, nID).Extract()
	th.AssertNoErr(t, err)
	fmt.Printf("Retrieved LB node details: %+v\n\n", n)
}
