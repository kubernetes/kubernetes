package flavors

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestListFlavors(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleList(t)

	pages := 0
	err := List(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := ExtractFlavors(page)
		if err != nil {
			return false, err
		}

		expected := []Flavor{
			Flavor{
				ID:   "1",
				Name: "m1.tiny",
				RAM:  512,
				Links: []gophercloud.Link{
					gophercloud.Link{Href: "https://openstack.example.com/v1.0/1234/flavors/1", Rel: "self"},
					gophercloud.Link{Href: "https://openstack.example.com/flavors/1", Rel: "bookmark"},
				},
			},
			Flavor{
				ID:   "2",
				Name: "m1.small",
				RAM:  1024,
				Links: []gophercloud.Link{
					gophercloud.Link{Href: "https://openstack.example.com/v1.0/1234/flavors/2", Rel: "self"},
					gophercloud.Link{Href: "https://openstack.example.com/flavors/2", Rel: "bookmark"},
				},
			},
			Flavor{
				ID:   "3",
				Name: "m1.medium",
				RAM:  2048,
				Links: []gophercloud.Link{
					gophercloud.Link{Href: "https://openstack.example.com/v1.0/1234/flavors/3", Rel: "self"},
					gophercloud.Link{Href: "https://openstack.example.com/flavors/3", Rel: "bookmark"},
				},
			},
			Flavor{
				ID:   "4",
				Name: "m1.large",
				RAM:  4096,
				Links: []gophercloud.Link{
					gophercloud.Link{Href: "https://openstack.example.com/v1.0/1234/flavors/4", Rel: "self"},
					gophercloud.Link{Href: "https://openstack.example.com/flavors/4", Rel: "bookmark"},
				},
			},
		}

		th.AssertDeepEquals(t, expected, actual)
		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, pages)
}

func TestGetFlavor(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGet(t)

	actual, err := Get(fake.ServiceClient(), flavorID).Extract()
	th.AssertNoErr(t, err)

	expected := &Flavor{
		ID:   "1",
		Name: "m1.tiny",
		RAM:  512,
		Links: []gophercloud.Link{
			gophercloud.Link{Href: "https://openstack.example.com/v1.0/1234/flavors/1", Rel: "self"},
		},
	}

	th.AssertDeepEquals(t, expected, actual)
}
