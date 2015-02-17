// +build acceptance

package v1

import (
	"testing"

	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/cdn/v1/services"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/cdn/v1/services"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestService(t *testing.T) {
	client := newClient(t)

	t.Log("Creating Service")
	loc := testServiceCreate(t, client, "test-site-1")
	t.Logf("Created service at location: %s", loc)

	defer testServiceDelete(t, client, loc)

	t.Log("Updating Service")
	testServiceUpdate(t, client, loc)

	t.Log("Retrieving Service")
	testServiceGet(t, client, loc)

	t.Log("Listing Services")
	testServiceList(t, client)
}

func testServiceCreate(t *testing.T, client *gophercloud.ServiceClient, name string) string {
	createOpts := os.CreateOpts{
		Name: name,
		Domains: []os.Domain{
			os.Domain{
				Domain: "www." + name + ".com",
			},
		},
		Origins: []os.Origin{
			os.Origin{
				Origin: name + ".com",
				Port:   80,
				SSL:    false,
			},
		},
		FlavorID: "cdn",
	}
	l, err := services.Create(client, createOpts).Extract()
	th.AssertNoErr(t, err)
	return l
}

func testServiceGet(t *testing.T, client *gophercloud.ServiceClient, id string) {
	s, err := services.Get(client, id).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Retrieved service: %+v", *s)
}

func testServiceUpdate(t *testing.T, client *gophercloud.ServiceClient, id string) {
	opts := os.UpdateOpts{
		os.Append{
			Value: os.Domain{Domain: "newDomain.com", Protocol: "http"},
		},
	}

	loc, err := services.Update(client, id, opts).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Successfully updated service at location: %s", loc)
}

func testServiceList(t *testing.T, client *gophercloud.ServiceClient) {
	err := services.List(client, nil).EachPage(func(page pagination.Page) (bool, error) {
		serviceList, err := os.ExtractServices(page)
		th.AssertNoErr(t, err)

		for _, service := range serviceList {
			t.Logf("Listing service: %+v", service)
		}

		return true, nil
	})

	th.AssertNoErr(t, err)
}

func testServiceDelete(t *testing.T, client *gophercloud.ServiceClient, id string) {
	err := services.Delete(client, id).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Successfully deleted service (%s)", id)
}
