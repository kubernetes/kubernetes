// +build acceptance

package v3

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/groups"
)

func TestGroupCRUD(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	createOpts := groups.CreateOpts{
		Name:     "testgroup",
		DomainID: "default",
		Extra: map[string]interface{}{
			"email": "testgroup@example.com",
		},
	}

	// Create Group in the default domain
	group, err := CreateGroup(t, client, &createOpts)
	if err != nil {
		t.Fatalf("Unable to create group: %v", err)
	}
	defer DeleteGroup(t, client, group.ID)

	tools.PrintResource(t, group)
	tools.PrintResource(t, group.Extra)

	updateOpts := groups.UpdateOpts{
		Description: "Test Users",
		Extra: map[string]interface{}{
			"email": "thetestgroup@example.com",
		},
	}

	newGroup, err := groups.Update(client, group.ID, updateOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update group: %v", err)
	}

	tools.PrintResource(t, newGroup)
	tools.PrintResource(t, newGroup.Extra)

	listOpts := groups.ListOpts{
		DomainID: "default",
	}

	// List all Groups in default domain
	allPages, err := groups.List(client, listOpts).AllPages()
	if err != nil {
		t.Fatalf("Unable to list groups: %v", err)
	}

	allGroups, err := groups.ExtractGroups(allPages)
	if err != nil {
		t.Fatalf("Unable to extract groups: %v", err)
	}

	for _, g := range allGroups {
		tools.PrintResource(t, g)
		tools.PrintResource(t, g.Extra)
	}

	// Get the recently created group by ID
	p, err := groups.Get(client, group.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get group: %v", err)
	}

	tools.PrintResource(t, p)
}
