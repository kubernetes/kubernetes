// +build acceptance lbs

package v1

import (
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/rackspace"
	th "github.com/rackspace/gophercloud/testhelper"
)

func newProvider() (*gophercloud.ProviderClient, error) {
	opts, err := rackspace.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}
	opts = tools.OnlyRS(opts)

	return rackspace.AuthenticatedClient(opts)
}

func newClient() (*gophercloud.ServiceClient, error) {
	provider, err := newProvider()
	if err != nil {
		return nil, err
	}

	return rackspace.NewLBV1(provider, gophercloud.EndpointOpts{
		Region: os.Getenv("RS_REGION"),
	})
}

func newComputeClient() (*gophercloud.ServiceClient, error) {
	provider, err := newProvider()
	if err != nil {
		return nil, err
	}

	return rackspace.NewComputeV2(provider, gophercloud.EndpointOpts{
		Region: os.Getenv("RS_REGION"),
	})
}

func setup(t *testing.T) *gophercloud.ServiceClient {
	client, err := newClient()
	th.AssertNoErr(t, err)

	return client
}

func intsToStr(ids []int) string {
	strIDs := []string{}
	for _, id := range ids {
		strIDs = append(strIDs, strconv.Itoa(id))
	}
	return strings.Join(strIDs, ", ")
}
