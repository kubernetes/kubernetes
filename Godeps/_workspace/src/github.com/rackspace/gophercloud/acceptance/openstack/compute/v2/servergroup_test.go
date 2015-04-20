// +build acceptance compute servers

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/servergroups"
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

func TestServerGroups(t *testing.T) {
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
		t.Logf("ServerGroup deleted.")
	}()

	err = getServerGroup(t, computeClient, sg.ID)
	if err != nil {
		t.Fatalf("Unable to get server group: %v", err)
	}
}
