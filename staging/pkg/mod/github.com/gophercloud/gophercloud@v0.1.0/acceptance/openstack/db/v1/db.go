// Package v2 contains common functions for creating db resources for use
// in acceptance tests. See the `*_test.go` files for example usages.
package v1

import (
	"fmt"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/db/v1/databases"
	"github.com/gophercloud/gophercloud/openstack/db/v1/instances"
	"github.com/gophercloud/gophercloud/openstack/db/v1/users"
)

// CreateDatabase will create a database with a randomly generated name.
// An error will be returned if the database was unable to be created.
func CreateDatabase(t *testing.T, client *gophercloud.ServiceClient, instanceID string) error {
	name := tools.RandomString("ACPTTEST", 8)
	t.Logf("Attempting to create database: %s", name)

	createOpts := databases.BatchCreateOpts{
		databases.CreateOpts{
			Name: name,
		},
	}

	return databases.Create(client, instanceID, createOpts).ExtractErr()
}

// CreateInstance will create an instance with a randomly generated name.
// The flavor of the instance will be the value of the OS_FLAVOR_ID
// environment variable. The Datastore will be pulled from the
// OS_DATASTORE_TYPE_ID environment variable.
// An error will be returned if the instance was unable to be created.
func CreateInstance(t *testing.T, client *gophercloud.ServiceClient) (*instances.Instance, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires instance creation in short mode.")
	}

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		return nil, err
	}

	name := tools.RandomString("ACPTTEST", 8)
	t.Logf("Attempting to create instance: %s", name)

	createOpts := instances.CreateOpts{
		FlavorRef: choices.FlavorID,
		Size:      1,
		Name:      name,
		Datastore: &instances.DatastoreOpts{
			Type:    choices.DBDatastoreType,
			Version: choices.DBDatastoreVersion,
		},
	}

	instance, err := instances.Create(client, createOpts).Extract()
	if err != nil {
		return instance, err
	}

	if err := WaitForInstanceStatus(client, instance, "ACTIVE"); err != nil {
		return instance, err
	}

	return instances.Get(client, instance.ID).Extract()
}

// CreateUser will create a user with a randomly generated name.
// An error will be returned if the user was unable to be created.
func CreateUser(t *testing.T, client *gophercloud.ServiceClient, instanceID string) error {
	name := tools.RandomString("ACPTTEST", 8)
	password := tools.RandomString("", 8)
	t.Logf("Attempting to create user: %s", name)

	createOpts := users.BatchCreateOpts{
		users.CreateOpts{
			Name:     name,
			Password: password,
		},
	}

	return users.Create(client, instanceID, createOpts).ExtractErr()
}

// DeleteDatabase deletes a database. A fatal error will occur if the database
// failed to delete. This works best when used as a deferred function.
func DeleteDatabase(t *testing.T, client *gophercloud.ServiceClient, instanceID, name string) {
	t.Logf("Attempting to delete database: %s", name)
	err := databases.Delete(client, instanceID, name).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete database %s: %s", name, err)
	}

	t.Logf("Deleted database: %s", name)
}

// DeleteInstance deletes an instance. A fatal error will occur if the instance
// failed to delete. This works best when used as a deferred function.
func DeleteInstance(t *testing.T, client *gophercloud.ServiceClient, id string) {
	t.Logf("Attempting to delete instance: %s", id)
	err := instances.Delete(client, id).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete instance %s: %s", id, err)
	}

	t.Logf("Deleted instance: %s", id)
}

// DeleteUser deletes a user. A fatal error will occur if the user
// failed to delete. This works best when used as a deferred function.
func DeleteUser(t *testing.T, client *gophercloud.ServiceClient, instanceID, name string) {
	t.Logf("Attempting to delete user: %s", name)
	err := users.Delete(client, instanceID, name).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete users %s: %s", name, err)
	}

	t.Logf("Deleted users: %s", name)
}

// WaitForInstanceState will poll an instance's status until it either matches
// the specified status or the status becomes ERROR.
func WaitForInstanceStatus(
	client *gophercloud.ServiceClient, instance *instances.Instance, status string) error {
	return tools.WaitFor(func() (bool, error) {
		latest, err := instances.Get(client, instance.ID).Extract()
		if err != nil {
			return false, err
		}

		if latest.Status == status {
			return true, nil
		}

		if latest.Status == "ERROR" {
			return false, fmt.Errorf("Instance in ERROR state")
		}

		return false, nil
	})
}
