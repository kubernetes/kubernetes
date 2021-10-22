// +build acceptance db

package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/db/v1/instances"
)

// Because it takes so long to create an instance,
// all tests will be housed in a single function.
func TestInstances(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}

	client, err := clients.NewDBV1Client()
	if err != nil {
		t.Fatalf("Unable to create a DB client: %v", err)
	}

	// Create and Get an instance.
	instance, err := CreateInstance(t, client)
	if err != nil {
		t.Fatalf("Unable to create instance: %v", err)
	}
	defer DeleteInstance(t, client, instance.ID)
	tools.PrintResource(t, &instance)

	// List all instances.
	allPages, err := instances.List(client).AllPages()
	if err != nil {
		t.Fatalf("Unable to list instances: %v", err)
	}

	allInstances, err := instances.ExtractInstances(allPages)
	if err != nil {
		t.Fatalf("Unable to extract instances: %v", err)
	}

	for _, instance := range allInstances {
		tools.PrintResource(t, instance)
	}

	// Enable root user.
	_, err = instances.EnableRootUser(client, instance.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to enable root user: %v", err)
	}

	enabled, err := instances.IsRootEnabled(client, instance.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to check if root user is enabled: %v", err)
	}

	t.Logf("Root user is enabled: %t", enabled)

	// Restart
	err = instances.Restart(client, instance.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to restart instance: %v", err)
	}

	err = WaitForInstanceStatus(client, instance, "ACTIVE")
	if err != nil {
		t.Fatalf("Unable to restart instance: %v", err)
	}
}
