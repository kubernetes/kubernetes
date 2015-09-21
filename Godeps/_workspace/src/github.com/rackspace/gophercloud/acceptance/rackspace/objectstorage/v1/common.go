// +build acceptance rackspace objectstorage v1

package v1

import (
	"os"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/rackspace"
	th "github.com/rackspace/gophercloud/testhelper"
)

func rackspaceAuthOptions(t *testing.T) gophercloud.AuthOptions {
	// Obtain credentials from the environment.
	options, err := rackspace.AuthOptionsFromEnv()
	th.AssertNoErr(t, err)
	options = tools.OnlyRS(options)

	if options.Username == "" {
		t.Fatal("Please provide a Rackspace username as RS_USERNAME.")
	}
	if options.APIKey == "" {
		t.Fatal("Please provide a Rackspace API key as RS_API_KEY.")
	}

	return options
}

func createClient(t *testing.T, cdn bool) (*gophercloud.ServiceClient, error) {
	region := os.Getenv("RS_REGION")
	if region == "" {
		t.Fatal("Please provide a Rackspace region as RS_REGION")
	}

	ao := rackspaceAuthOptions(t)

	provider, err := rackspace.NewClient(ao.IdentityEndpoint)
	th.AssertNoErr(t, err)

	err = rackspace.Authenticate(provider, ao)
	th.AssertNoErr(t, err)

	if cdn {
		return rackspace.NewObjectCDNV1(provider, gophercloud.EndpointOpts{
			Region: region,
		})
	}

	return rackspace.NewObjectStorageV1(provider, gophercloud.EndpointOpts{
		Region: region,
	})
}
