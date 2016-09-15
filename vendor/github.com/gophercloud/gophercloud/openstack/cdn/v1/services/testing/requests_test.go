package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/cdn/v1/services"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleListCDNServiceSuccessfully(t)

	count := 0

	err := services.List(fake.ServiceClient(), &services.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := services.ExtractServices(page)
		if err != nil {
			t.Errorf("Failed to extract services: %v", err)
			return false, err
		}

		expected := []services.Service{
			{
				ID:   "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0",
				Name: "mywebsite.com",
				Domains: []services.Domain{
					{
						Domain: "www.mywebsite.com",
					},
				},
				Origins: []services.Origin{
					{
						Origin: "mywebsite.com",
						Port:   80,
						SSL:    false,
					},
				},
				Caching: []services.CacheRule{
					{
						Name: "default",
						TTL:  3600,
					},
					{
						Name: "home",
						TTL:  17200,
						Rules: []services.TTLRule{
							{
								Name:       "index",
								RequestURL: "/index.htm",
							},
						},
					},
					{
						Name: "images",
						TTL:  12800,
						Rules: []services.TTLRule{
							{
								Name:       "images",
								RequestURL: "*.png",
							},
						},
					},
				},
				Restrictions: []services.Restriction{
					{
						Name: "website only",
						Rules: []services.RestrictionRule{
							{
								Name:     "mywebsite.com",
								Referrer: "www.mywebsite.com",
							},
						},
					},
				},
				FlavorID: "asia",
				Status:   "deployed",
				Errors:   []services.Error{},
				Links: []gophercloud.Link{
					{
						Href: "https://www.poppycdn.io/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0",
						Rel:  "self",
					},
					{
						Href: "mywebsite.com.cdn123.poppycdn.net",
						Rel:  "access_url",
					},
					{
						Href: "https://www.poppycdn.io/v1.0/flavors/asia",
						Rel:  "flavor",
					},
				},
			},
			{
				ID:   "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f1",
				Name: "myothersite.com",
				Domains: []services.Domain{
					{
						Domain: "www.myothersite.com",
					},
				},
				Origins: []services.Origin{
					{
						Origin: "44.33.22.11",
						Port:   80,
						SSL:    false,
					},
					{
						Origin: "77.66.55.44",
						Port:   80,
						SSL:    false,
						Rules: []services.OriginRule{
							{
								Name:       "videos",
								RequestURL: "^/videos/*.m3u",
							},
						},
					},
				},
				Caching: []services.CacheRule{
					{
						Name: "default",
						TTL:  3600,
					},
				},
				Restrictions: []services.Restriction{},
				FlavorID:     "europe",
				Status:       "deployed",
				Links: []gophercloud.Link{
					gophercloud.Link{
						Href: "https://www.poppycdn.io/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f1",
						Rel:  "self",
					},
					gophercloud.Link{
						Href: "myothersite.com.poppycdn.net",
						Rel:  "access_url",
					},
					gophercloud.Link{
						Href: "https://www.poppycdn.io/v1.0/flavors/europe",
						Rel:  "flavor",
					},
				},
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCreateCDNServiceSuccessfully(t)

	createOpts := services.CreateOpts{
		Name: "mywebsite.com",
		Domains: []services.Domain{
			{
				Domain: "www.mywebsite.com",
			},
			{
				Domain: "blog.mywebsite.com",
			},
		},
		Origins: []services.Origin{
			{
				Origin: "mywebsite.com",
				Port:   80,
				SSL:    false,
			},
		},
		Restrictions: []services.Restriction{
			{
				Name: "website only",
				Rules: []services.RestrictionRule{
					{
						Name:     "mywebsite.com",
						Referrer: "www.mywebsite.com",
					},
				},
			},
		},
		Caching: []services.CacheRule{
			{
				Name: "default",
				TTL:  3600,
			},
		},
		FlavorID: "cdn",
	}

	expected := "https://global.cdn.api.rackspacecloud.com/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0"
	actual, err := services.Create(fake.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, expected, actual)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGetCDNServiceSuccessfully(t)

	expected := &services.Service{
		ID:   "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0",
		Name: "mywebsite.com",
		Domains: []services.Domain{
			{
				Domain:   "www.mywebsite.com",
				Protocol: "http",
			},
		},
		Origins: []services.Origin{
			{
				Origin: "mywebsite.com",
				Port:   80,
				SSL:    false,
			},
		},
		Caching: []services.CacheRule{
			{
				Name: "default",
				TTL:  3600,
			},
			{
				Name: "home",
				TTL:  17200,
				Rules: []services.TTLRule{
					{
						Name:       "index",
						RequestURL: "/index.htm",
					},
				},
			},
			{
				Name: "images",
				TTL:  12800,
				Rules: []services.TTLRule{
					{
						Name:       "images",
						RequestURL: "*.png",
					},
				},
			},
		},
		Restrictions: []services.Restriction{
			{
				Name: "website only",
				Rules: []services.RestrictionRule{
					{
						Name:     "mywebsite.com",
						Referrer: "www.mywebsite.com",
					},
				},
			},
		},
		FlavorID: "cdn",
		Status:   "deployed",
		Errors:   []services.Error{},
		Links: []gophercloud.Link{
			{
				Href: "https://global.cdn.api.rackspacecloud.com/v1.0/110011/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0",
				Rel:  "self",
			},
			{
				Href: "blog.mywebsite.com.cdn1.raxcdn.com",
				Rel:  "access_url",
			},
			{
				Href: "https://global.cdn.api.rackspacecloud.com/v1.0/110011/flavors/cdn",
				Rel:  "flavor",
			},
		},
	}

	actual, err := services.Get(fake.ServiceClient(), "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestSuccessfulUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleUpdateCDNServiceSuccessfully(t)

	expected := "https://www.poppycdn.io/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0"
	ops := services.UpdateOpts{
		// Append a single Domain
		services.Append{Value: services.Domain{Domain: "appended.mocksite4.com"}},
		// Insert a single Domain
		services.Insertion{
			Index: 4,
			Value: services.Domain{Domain: "inserted.mocksite4.com"},
		},
		// Bulk addition
		services.Append{
			Value: services.DomainList{
				{Domain: "bulkadded1.mocksite4.com"},
				{Domain: "bulkadded2.mocksite4.com"},
			},
		},
		// Replace a single Origin
		services.Replacement{
			Index: 2,
			Value: services.Origin{Origin: "44.33.22.11", Port: 80, SSL: false},
		},
		// Bulk replace Origins
		services.Replacement{
			Index: 0, // Ignored
			Value: services.OriginList{
				{Origin: "44.33.22.11", Port: 80, SSL: false},
				{Origin: "55.44.33.22", Port: 443, SSL: true},
			},
		},
		// Remove a single CacheRule
		services.Removal{
			Index: 8,
			Path:  services.PathCaching,
		},
		// Bulk removal
		services.Removal{
			All:  true,
			Path: services.PathCaching,
		},
		// Service name replacement
		services.NameReplacement{
			NewName: "differentServiceName",
		},
	}

	actual, err := services.Update(fake.ServiceClient(), "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0", ops).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, expected, actual)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleDeleteCDNServiceSuccessfully(t)

	err := services.Delete(fake.ServiceClient(), "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0").ExtractErr()
	th.AssertNoErr(t, err)
}
