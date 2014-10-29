// +build acceptance

package v1

import (
	"os"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack"
	th "github.com/rackspace/gophercloud/testhelper"
)

var metadata = map[string]string{"gopher": "cloud"}

func newClient(t *testing.T) *gophercloud.ServiceClient {
	ao, err := openstack.AuthOptionsFromEnv()
	th.AssertNoErr(t, err)

	client, err := openstack.AuthenticatedClient(ao)
	th.AssertNoErr(t, err)

	c, err := openstack.NewObjectStorageV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
	th.AssertNoErr(t, err)
	return c
}
