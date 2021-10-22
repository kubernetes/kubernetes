// +build acceptance networking agents

package agents

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/agents"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestAgentsRead(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	allPages, err := agents.List(client, agents.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)

	allAgents, err := agents.ExtractAgents(allPages)
	th.AssertNoErr(t, err)

	for _, agent := range allAgents {
		t.Logf("Retrieved Networking V2 agent: %s", agent.ID)
		tools.PrintResource(t, agent)
	}
}
