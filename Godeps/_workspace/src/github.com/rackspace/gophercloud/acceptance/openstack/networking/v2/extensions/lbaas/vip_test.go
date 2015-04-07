// +build acceptance networking lbaas lbaasvip

package lbaas

import (
	"testing"

	base "github.com/rackspace/gophercloud/acceptance/openstack/networking/v2"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/vips"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestVIPs(t *testing.T) {
	base.Setup(t)
	defer base.Teardown()

	// setup
	networkID, subnetID := SetupTopology(t)
	poolID := CreatePool(t, subnetID)

	// create VIP
	VIPID := createVIP(t, subnetID, poolID)

	// list VIPs
	listVIPs(t)

	// update VIP
	updateVIP(t, VIPID)

	// get VIP
	getVIP(t, VIPID)

	// delete VIP
	deleteVIP(t, VIPID)

	// teardown
	DeletePool(t, poolID)
	DeleteTopology(t, networkID)
}

func createVIP(t *testing.T, subnetID, poolID string) string {
	p, err := vips.Create(base.Client, vips.CreateOpts{
		Protocol:     "HTTP",
		Name:         "New_VIP",
		AdminStateUp: vips.Up,
		SubnetID:     subnetID,
		PoolID:       poolID,
		ProtocolPort: 80,
	}).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Created pool %s", p.ID)

	return p.ID
}

func listVIPs(t *testing.T) {
	err := vips.List(base.Client, vips.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		vipList, err := vips.ExtractVIPs(page)
		if err != nil {
			t.Errorf("Failed to extract VIPs: %v", err)
			return false, err
		}

		for _, vip := range vipList {
			t.Logf("Listing VIP: ID [%s] Name [%s] Address [%s] Port [%s] Connection Limit [%d]",
				vip.ID, vip.Name, vip.Address, vip.ProtocolPort, vip.ConnLimit)
		}

		return true, nil
	})

	th.AssertNoErr(t, err)
}

func updateVIP(t *testing.T, VIPID string) {
	i1000 := 1000
	_, err := vips.Update(base.Client, VIPID, vips.UpdateOpts{ConnLimit: &i1000}).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Updated VIP ID [%s]", VIPID)
}

func getVIP(t *testing.T, VIPID string) {
	vip, err := vips.Get(base.Client, VIPID).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Getting VIP ID [%s]: Status [%s]", vip.ID, vip.Status)
}

func deleteVIP(t *testing.T, VIPID string) {
	res := vips.Delete(base.Client, VIPID)

	th.AssertNoErr(t, res.Err)

	t.Logf("Deleted VIP %s", VIPID)
}
