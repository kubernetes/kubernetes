// +build acceptance compute servergroups

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/servergroups"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestServergroupsCreateDelete(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	serverGroup, err := CreateServerGroup(t, client, "anti-affinity")
	th.AssertNoErr(t, err)
	defer DeleteServerGroup(t, client, serverGroup)

	serverGroup, err = servergroups.Get(client, serverGroup.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, serverGroup)

	allPages, err := servergroups.List(client).AllPages()
	th.AssertNoErr(t, err)

	allServerGroups, err := servergroups.ExtractServerGroups(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, sg := range allServerGroups {
		tools.PrintResource(t, serverGroup)

		if sg.ID == serverGroup.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}

func TestServergroupsAffinityPolicy(t *testing.T) {
	clients.RequireLong(t)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	serverGroup, err := CreateServerGroup(t, client, "affinity")
	th.AssertNoErr(t, err)
	defer DeleteServerGroup(t, client, serverGroup)

	firstServer, err := CreateServerInServerGroup(t, client, serverGroup)
	th.AssertNoErr(t, err)
	defer DeleteServer(t, client, firstServer)

	firstServer, err = servers.Get(client, firstServer.ID).Extract()
	th.AssertNoErr(t, err)

	secondServer, err := CreateServerInServerGroup(t, client, serverGroup)
	th.AssertNoErr(t, err)
	defer DeleteServer(t, client, secondServer)

	secondServer, err = servers.Get(client, secondServer.ID).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, firstServer.HostID, secondServer.HostID)
}
