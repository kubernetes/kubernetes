// +build acceptance

package v3

import (
	"testing"

	"github.com/rackspace/gophercloud/openstack"
	tokens3 "github.com/rackspace/gophercloud/openstack/identity/v3/tokens"
)

func TestGetToken(t *testing.T) {
	// Obtain credentials from the environment.
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		t.Fatalf("Unable to acquire credentials: %v", err)
	}

	// Trim out unused fields. Skip if we don't have a UserID.
	ao.Username, ao.TenantID, ao.TenantName = "", "", ""
	if ao.UserID == "" {
		t.Logf("Skipping identity v3 tests because no OS_USERID is present.")
		return
	}

	// Create an unauthenticated client.
	provider, err := openstack.NewClient(ao.IdentityEndpoint)
	if err != nil {
		t.Fatalf("Unable to instantiate client: %v", err)
	}

	// Create a service client.
	service := openstack.NewIdentityV3(provider)

	// Use the service to create a token.
	token, err := tokens3.Create(service, ao, nil).Extract()
	if err != nil {
		t.Fatalf("Unable to get token: %v", err)
	}

	t.Logf("Acquired token: %s", token.ID)
}
