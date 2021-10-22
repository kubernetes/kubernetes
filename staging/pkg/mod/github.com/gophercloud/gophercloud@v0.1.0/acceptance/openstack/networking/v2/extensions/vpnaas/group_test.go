// +build acceptance networking vpnaas

package vpnaas

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/vpnaas/endpointgroups"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestGroupList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	allPages, err := endpointgroups.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allGroups, err := endpointgroups.ExtractEndpointGroups(allPages)
	th.AssertNoErr(t, err)

	for _, group := range allGroups {
		tools.PrintResource(t, group)
	}
}

func TestGroupCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	group, err := CreateEndpointGroup(t, client)
	th.AssertNoErr(t, err)
	defer DeleteEndpointGroup(t, client, group.ID)
	tools.PrintResource(t, group)

	newGroup, err := endpointgroups.Get(client, group.ID).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, newGroup)

	updatedName := "updatedname"
	updatedDescription := "updated description"
	updateOpts := endpointgroups.UpdateOpts{
		Name:        &updatedName,
		Description: &updatedDescription,
	}
	updatedGroup, err := endpointgroups.Update(client, group.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, updatedGroup)
}
