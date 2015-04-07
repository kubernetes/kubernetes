package flavors

import (
	"testing"

	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/cdn/v1/flavors"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	os.HandleListCDNFlavorsSuccessfully(t)

	count := 0

	err := List(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := os.ExtractFlavors(page)
		if err != nil {
			t.Errorf("Failed to extract flavors: %v", err)
			return false, err
		}

		expected := []os.Flavor{
			os.Flavor{
				ID: "europe",
				Providers: []os.Provider{
					os.Provider{
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

	os.HandleGetCDNFlavorSuccessfully(t)

	expected := &os.Flavor{
		ID: "asia",
		Providers: []os.Provider{
			os.Provider{
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

	actual, err := Get(fake.ServiceClient(), "asia").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}
