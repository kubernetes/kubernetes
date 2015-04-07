// +build acceptance networking lbaas lbaasmember

package lbaas

import (
	"testing"

	base "github.com/rackspace/gophercloud/acceptance/openstack/networking/v2"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/members"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestMembers(t *testing.T) {
	base.Setup(t)
	defer base.Teardown()

	// setup
	networkID, subnetID := SetupTopology(t)
	poolID := CreatePool(t, subnetID)

	// create member
	memberID := createMember(t, poolID)

	// list members
	listMembers(t)

	// update member
	updateMember(t, memberID)

	// get member
	getMember(t, memberID)

	// delete member
	deleteMember(t, memberID)

	// teardown
	DeletePool(t, poolID)
	DeleteTopology(t, networkID)
}

func createMember(t *testing.T, poolID string) string {
	m, err := members.Create(base.Client, members.CreateOpts{
		Address:      "192.168.199.1",
		ProtocolPort: 8080,
		PoolID:       poolID,
	}).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Created member: ID [%s] Status [%s] Weight [%d] Address [%s] Port [%d]",
		m.ID, m.Status, m.Weight, m.Address, m.ProtocolPort)

	return m.ID
}

func listMembers(t *testing.T) {
	err := members.List(base.Client, members.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		memberList, err := members.ExtractMembers(page)
		if err != nil {
			t.Errorf("Failed to extract members: %v", err)
			return false, err
		}

		for _, m := range memberList {
			t.Logf("Listing member: ID [%s] Status [%s]", m.ID, m.Status)
		}

		return true, nil
	})

	th.AssertNoErr(t, err)
}

func updateMember(t *testing.T, memberID string) {
	m, err := members.Update(base.Client, memberID, members.UpdateOpts{AdminStateUp: true}).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Updated member ID [%s]", m.ID)
}

func getMember(t *testing.T, memberID string) {
	m, err := members.Get(base.Client, memberID).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Getting member ID [%s]", m.ID)
}

func deleteMember(t *testing.T, memberID string) {
	res := members.Delete(base.Client, memberID)
	th.AssertNoErr(t, res.Err)
	t.Logf("Deleted member %s", memberID)
}
