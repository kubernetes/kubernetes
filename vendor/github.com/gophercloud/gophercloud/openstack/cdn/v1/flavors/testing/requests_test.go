package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/cdn/v1/flavors"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleListCDNFlavorsSuccessfully(t)

	count := 0

	err := flavors.List(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := flavors.ExtractFlavors(page)
		if err != nil {
			t.Errorf("Failed to extract flavors: %v", err)
			return false, err
		}

		expected := []flavors.Flavor{
			{
				ID: "europe",
				Providers: []flavors.Provider{
					{
						Provider: "Fastly",
						Links: []gophercloud.Link{
							gophercloud.Link{
								Href: "http://www.fastly.com",
								Rel:  "provider_url",
							},
						},
					},
				},
				Links: []gophercloud.Link{
					gophercloud.Link{
						Href: "https://www.poppycdn.io/v1.0/flavors/europe",
						Rel:  "self",
					},
				},
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, count)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGetCDNFlavorSuccessfully(t)

	expected := &flavors.Flavor{
		ID: "asia",
		Providers: []flavors.Provider{
			{
				Provider: "ChinaCache",
				Links: []gophercloud.Link{
					gophercloud.Link{
						Href: "http://www.chinacache.com",
						Rel:  "provider_url",
					},
				},
			},
		},
		Links: []gophercloud.Link{
			gophercloud.Link{
				Href: "https://www.poppycdn.io/v1.0/flavors/asia",
				Rel:  "self",
			},
		},
	}

	actual, err := flavors.Get(fake.ServiceClient(), "asia").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}
