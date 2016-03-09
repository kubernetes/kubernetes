// +build acceptance

package v3

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack"
	th "github.com/rackspace/gophercloud/testhelper"
)

func createAuthenticatedClient(t *testing.T) *gophercloud.ServiceClient {
	// Obtain credentials from the environment.
	ao, err := openstack.AuthOptionsFromEnv()
	th.AssertNoErr(t, err)

	// Trim out unused fields.
	ao.Username, ao.TenantID, ao.TenantName = "", "", ""

	if ao.UserID == "" {
		t.Logf("Skipping identity v3 tests because no OS_USERID is present.")
		return nil
	}

	// Create a client and manually authenticate against v3.
	providerClient, err := openstack.NewClient(ao.IdentityEndpoint)
	if err != nil {
		t.Fatalf("Unable to instantiate client: %v", err)
	}

	err = openstack.AuthenticateV3(providerClient, ao)
	if err != nil {
		t.Fatalf("Unable to authenticate against identity v3: %v", err)
	}

	// Create a service client.
	return openstack.NewIdentityV3(providerClient)
}
