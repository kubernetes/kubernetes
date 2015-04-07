package v2

import (
	"os"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/rackspace"
	th "github.com/rackspace/gophercloud/testhelper"
)

var Client *gophercloud.ServiceClient

func NewClient() (*gophercloud.ServiceClient, error) {
	opts, err := rackspace.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	provider, err := rackspace.AuthenticatedClient(opts)
	if err != nil {
		return nil, err
	}

	return rackspace.NewNetworkV2(provider, gophercloud.EndpointOpts{
		Name:   "cloudNetworks",
		Region: os.Getenv("RS_REGION"),
	})
}

func Setup(t *testing.T) {
	client, err := NewClient()
	th.AssertNoErr(t, err)
	Client = client
}

func Teardown() {
	Client = nil
}
