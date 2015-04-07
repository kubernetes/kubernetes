// +build acceptance identity

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack"
	th "github.com/rackspace/gophercloud/testhelper"
)

func v2AuthOptions(t *testing.T) gophercloud.AuthOptions {
	// Obtain credentials from the environment.
	ao, err := openstack.AuthOptionsFromEnv()
	th.AssertNoErr(t, err)

	// Trim out unused fields. Prefer authentication by API key to password.
	ao.UserID, ao.DomainID, ao.DomainName = "", "", ""
	if ao.APIKey != "" {
		ao.Password = ""
	}

	return ao
}

func createClient(t *testing.T, auth bool) *gophercloud.ServiceClient {
	ao := v2AuthOptions(t)

	provider, err := openstack.NewClient(ao.IdentityEndpoint)
	th.AssertNoErr(t, err)

	if auth {
		err = openstack.AuthenticateV2(provider, ao)
		th.AssertNoErr(t, err)
	}

	return openstack.NewIdentityV2(provider)
}

func unauthenticatedClient(t *testing.T) *gophercloud.ServiceClient {
	return createClient(t, false)
}

func authenticatedClient(t *testing.T) *gophercloud.ServiceClient {
	return createClient(t, true)
}
