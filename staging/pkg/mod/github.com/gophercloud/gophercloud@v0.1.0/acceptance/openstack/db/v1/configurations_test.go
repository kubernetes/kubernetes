// +build acceptance db

package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/db/v1/configurations"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestConfigurationsCRUD(t *testing.T) {
	client, err := clients.NewDBV1Client()
	if err != nil {
		t.Fatalf("Unable to create a DB client: %v", err)
	}

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatalf("Unable to get environment settings")
	}

	createOpts := &configurations.CreateOpts{
		Name:        "test",
		Description: "description",
	}

	datastore := configurations.DatastoreOpts{
		Type:    choices.DBDatastoreType,
		Version: choices.DBDatastoreVersion,
	}
	createOpts.Datastore = &datastore

	values := make(map[string]interface{})
	values["collation_server"] = "latin1_swedish_ci"
	createOpts.Values = values

	cgroup, err := configurations.Create(client, createOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to create configuration: %v", err)
	}

	readCgroup, err := configurations.Get(client, cgroup.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to read configuration: %v", err)
	}

	tools.PrintResource(t, readCgroup)
	th.AssertEquals(t, readCgroup.Name, createOpts.Name)
	th.AssertEquals(t, readCgroup.Description, createOpts.Description)
	// TODO: verify datastore
	//th.AssertDeepEquals(t, readCgroup.Datastore, datastore)

	// Update cgroup
	newCgroupName := "New configuration name"
	newCgroupDescription := ""
	updateOpts := configurations.UpdateOpts{
		Name:        newCgroupName,
		Description: &newCgroupDescription,
	}
	err = configurations.Update(client, cgroup.ID, updateOpts).ExtractErr()
	th.AssertNoErr(t, err)

	newCgroup, err := configurations.Get(client, cgroup.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to read updated configuration: %v", err)
	}

	tools.PrintResource(t, newCgroup)
	th.AssertEquals(t, newCgroup.Name, newCgroupName)
	th.AssertEquals(t, newCgroup.Description, newCgroupDescription)

	err = configurations.Delete(client, cgroup.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete configuration: %v", err)
	}
}
