// +build acceptance compute servers

package v2

import (
	"fmt"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/schedulerhints"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/servergroups"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	th "github.com/rackspace/gophercloud/testhelper"
)

func createServerGroup(t *testing.T, computeClient *gophercloud.ServiceClient) (*servergroups.ServerGroup, error) {
	sg, err := servergroups.Create(computeClient, &servergroups.CreateOpts{
		Name:     "test",
		Policies: []string{"affinity"},
	}).Extract()

	if err != nil {
		t.Fatalf("Unable to create server group: %v", err)
	}

	t.Logf("Created server group: %v", sg.ID)
	t.Logf("It has policies: %v", sg.Policies)

	return sg, nil
}

func getServerGroup(t *testing.T, computeClient *gophercloud.ServiceClient, sgID string) error {
	sg, err := servergroups.Get(computeClient, sgID).Extract()
	if err != nil {
		t.Fatalf("Unable to get server group: %v", err)
	}

	t.Logf("Got server group: %v", sg.Name)

	return nil
}

func createServerInGroup(t *testing.T, computeClient *gophercloud.ServiceClient, choices *ComputeChoices, serverGroup *servergroups.ServerGroup) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create server: %s\n", name)

	pwd := tools.MakeNewPassword("")

	serverCreateOpts := servers.CreateOpts{
		Name:      name,
		FlavorRef: choices.FlavorID,
		ImageRef:  choices.ImageID,
		AdminPass: pwd,
	}
	server, err := servers.Create(computeClient, schedulerhints.CreateOptsExt{
		serverCreateOpts,
		schedulerhints.SchedulerHints{
			Group: serverGroup.ID,
		},
	}).Extract()
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}

	th.AssertEquals(t, pwd, server.AdminPass)

	return server, err
}

func verifySchedulerWorked(t *testing.T, firstServer, secondServer *servers.Server) error {
	t.Logf("First server hostID: %v", firstServer.HostID)
	t.Logf("Second server hostID: %v", secondServer.HostID)
	if firstServer.HostID == secondServer.HostID {
		return nil
	}

	return fmt.Errorf("%s and %s were not scheduled on the same host.", firstServer.ID, secondServer.ID)
}

func TestServerGroups(t *testing.T) {
	choices, err := ComputeChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	computeClient, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	sg, err := createServerGroup(t, computeClient)
	if err != nil {
		t.Fatalf("Unable to create server group: %v", err)
	}
	defer func() {
		servergroups.Delete(computeClient, sg.ID)
		t.Logf("Server Group deleted.")
	}()

	err = getServerGroup(t, computeClient, sg.ID)
	if err != nil {
		t.Fatalf("Unable to get server group: %v", err)
	}

	firstServer, err := createServerInGroup(t, computeClient, choices, sg)
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}
	defer func() {
		servers.Delete(computeClient, firstServer.ID)
		t.Logf("Server deleted.")
	}()

	if err = waitForStatus(computeClient, firstServer, "ACTIVE"); err != nil {
		t.Fatalf("Unable to wait for server: %v", err)
	}

	firstServer, err = servers.Get(computeClient, firstServer.ID).Extract()

	secondServer, err := createServerInGroup(t, computeClient, choices, sg)
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}
	defer func() {
		servers.Delete(computeClient, secondServer.ID)
		t.Logf("Server deleted.")
	}()

	if err = waitForStatus(computeClient, secondServer, "ACTIVE"); err != nil {
		t.Fatalf("Unable to wait for server: %v", err)
	}

	secondServer, err = servers.Get(computeClient, secondServer.ID).Extract()

	if err = verifySchedulerWorked(t, firstServer, secondServer); err != nil {
		t.Fatalf("Scheduling did not work: %v", err)
	}
}
