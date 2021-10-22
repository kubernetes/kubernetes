package testing

import (
	"fmt"
	"net/http"
	"testing"

	common "github.com/gophercloud/gophercloud/openstack/common/extensions"
	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/extensions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")

		fmt.Fprintf(w, `
{
    "extensions": [
        {
            "updated": "2013-01-20T00:00:00-00:00",
            "name": "Neutron Service Type Management",
            "links": [],
            "namespace": "http://docs.openstack.org/ext/neutron/service-type/api/v1.0",
            "alias": "service-type",
            "description": "API for retrieving service providers for Neutron advanced services"
        }
    ]
}
      `)
	})

	count := 0

	extensions.List(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := extensions.ExtractExtensions(page)
		if err != nil {
			t.Errorf("Failed to extract extensions: %v", err)
		}

		expected := []extensions.Extension{
			{
				Extension: common.Extension{
					Updated:     "2013-01-20T00:00:00-00:00",
					Name:        "Neutron Service Type Management",
					Links:       []interface{}{},
					Namespace:   "http://docs.openstack.org/ext/neutron/service-type/api/v1.0",
					Alias:       "service-type",
					Description: "API for retrieving service providers for Neutron advanced services",
				},
			},
		}

		th.AssertDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/extensions/agent", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "extension": {
        "updated": "2013-02-03T10:00:00-00:00",
        "name": "agent",
        "links": [],
        "namespace": "http://docs.openstack.org/ext/agent/api/v2.0",
        "alias": "agent",
        "description": "The agent management extension."
    }
}
    `)
	})

	ext, err := extensions.Get(fake.ServiceClient(), "agent").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, ext.Updated, "2013-02-03T10:00:00-00:00")
	th.AssertEquals(t, ext.Name, "agent")
	th.AssertEquals(t, ext.Namespace, "http://docs.openstack.org/ext/agent/api/v2.0")
	th.AssertEquals(t, ext.Alias, "agent")
	th.AssertEquals(t, ext.Description, "The agent management extension.")
}
