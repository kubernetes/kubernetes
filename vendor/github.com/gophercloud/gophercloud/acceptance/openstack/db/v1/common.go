// +build acceptance db

package v1

import (
	"os"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"
	"github.com/gophercloud/gophercloud/openstack/db/v1/instances"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func newClient(t *testing.T) *gophercloud.ServiceClient {
	ao, err := openstack.AuthOptionsFromEnv()
	th.AssertNoErr(t, err)

	client, err := openstack.AuthenticatedClient(ao)
	th.AssertNoErr(t, err)

	c, err := openstack.NewDBV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
	th.AssertNoErr(t, err)

	return c
}

type context struct {
	test       *testing.T
	client     *gophercloud.ServiceClient
	instanceID string
	DBIDs      []string
	users      []string
}

func newContext(t *testing.T) context {
	return context{
		test:   t,
		client: newClient(t),
	}
}

func (c context) Logf(msg string, args ...interface{}) {
	if len(args) > 0 {
		c.test.Logf(msg, args...)
	} else {
		c.test.Log(msg)
	}
}

func (c context) AssertNoErr(err error) {
	th.AssertNoErr(c.test, err)
}

func (c context) WaitUntilActive(id string) {
	err := gophercloud.WaitFor(60, func() (bool, error) {
		inst, err := instances.Get(c.client, id).Extract()
		if err != nil {
			return false, err
		}
		if inst.Status == "ACTIVE" {
			return true, nil
		}
		return false, nil
	})

	c.AssertNoErr(err)
}
