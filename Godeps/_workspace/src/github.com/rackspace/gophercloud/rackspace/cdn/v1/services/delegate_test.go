package services

import (
	"testing"

	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/cdn/v1/services"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	os.HandleListCDNServiceSuccessfully(t)

	count := 0

	err := List(fake.ServiceClient(), &os.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := os.ExtractServices(page)
		if err != nil {
			t.Errorf("Failed to extract services: %v", err)
			return false, err
		}

		expected := []os.Service{
			os.Service{
				ID:   "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0",
				Name: "mywebsite.com",
				Domains: []os.Domain{
					os.Domain{
						Domain: "www.mywebsite.com",
					},
				},
				Origins: []os.Origin{
					os.Origin{
						Origin: "mywebsite.com",
						Port:   80,
						SSL:    false,
					},
				},
				Caching: []os.CacheRule{
					os.CacheRule{
						Name: "default",
						TTL:  3600,
					},
					os.CacheRule{
						Name: "home",
						TTL:  17200,
						Rules: []os.TTLRule{
							os.TTLRule{
								Name:       "index",
								RequestURL: "/index.htm",
							},
						},
					},
					os.CacheRule{
						Name: "images",
						TTL:  12800,
						Rules: []os.TTLRule{
							os.TTLRule{
								Name:       "images",
								RequestURL: "*.png",
							},
						},
					},
				},
				Restrictions: []os.Restriction{
					os.Restriction{
						Name: "website only",
						Rules: []os.RestrictionRule{
							os.RestrictionRule{
								Name:     "mywebsite.com",
								Referrer: "www.mywebsite.com",
							},
						},
					},
				},
				FlavorID: "asia",
				Status:   "deployed",
				Errors:   []os.Error{},
				Links: []gophercloud.Link{
					gophercloud.Link{
						Href: "https://www.poppycdn.io/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0",
						Rel:  "self",
					},
					gophercloud.Link{
						Href: "mywebsite.com.cdn123.poppycdn.net",
						Rel:  "access_url",
					},
					gophercloud.Link{
						Href: "https://www.poppycdn.io/v1.0/flavors/asia",
						Rel:  "flavor",
					},
				},
			},
			os.Service{
				ID:   "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f1",
				Name: "myothersite.com",
				Domains: []os.Domain{
					os.Domain{
						Domain: "www.myothersite.com",
					},
				},
				Origins: []os.Origin{
					os.Origin{
						Origin: "44.33.22.11",
						Port:   80,
						SSL:    false,
					},
					os.Origin{
						Origin: "77.66.55.44",
						Port:   80,
						SSL:    false,
						Rules: []os.OriginRule{
							os.OriginRule{
								Name:       "videos",
								RequestURL: "^/videos/*.m3u",
							},
						},
					},
				},
				Caching: []os.CacheRule{
					os.CacheRule{
						Name: "default",
						TTL:  3600,
					},
				},
				Restrictions: []os.Restriction{},
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

	os.HandleCreateCDNServiceSuccessfully(t)

	createOpts := os.CreateOpts{
		Name: "mywebsite.com",
		Domains: []os.Domain{
			os.Domain{
				Domain: "www.mywebsite.com",
			},
			os.Domain{
				Domain: "blog.mywebsite.com",
			},
		},
		Origins: []os.Origin{
			os.Origin{
				Origin: "mywebsite.com",
				Port:   80,
				SSL:    false,
			},
		},
		Restrictions: []os.Restriction{
			os.Restriction{
				Name: "website only",
				Rules: []os.RestrictionRule{
					os.RestrictionRule{
						Name:     "mywebsite.com",
						Referrer: "www.mywebsite.com",
					},
				},
			},
		},
		Caching: []os.CacheRule{
			os.CacheRule{
				Name: "default",
				TTL:  3600,
			},
		},
		FlavorID: "cdn",
	}

	expected := "https://global.cdn.api.rackspacecloud.com/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0"
	actual, err := Create(fake.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, expected, actual)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	os.HandleGetCDNServiceSuccessfully(t)

	expected := &os.Service{
		ID:   "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0",
		Name: "mywebsite.com",
		Domains: []os.Domain{
			os.Domain{
				Domain:   "www.mywebsite.com",
				Protocol: "http",
			},
		},
		Origins: []os.Origin{
			os.Origin{
				Origin: "mywebsite.com",
				Port:   80,
				SSL:    false,
			},
		},
		Caching: []os.CacheRule{
			os.CacheRule{
				Name: "default",
				TTL:  3600,
			},
			os.CacheRule{
				Name: "home",
				TTL:  17200,
				Rules: []os.TTLRule{
					os.TTLRule{
						Name:       "index",
						RequestURL: "/index.htm",
					},
				},
			},
			os.CacheRule{
				Name: "images",
				TTL:  12800,
				Rules: []os.TTLRule{
					os.TTLRule{
						Name:       "images",
						RequestURL: "*.png",
					},
				},
			},
		},
		Restrictions: []os.Restriction{
			os.Restriction{
				Name: "website only",
				Rules: []os.RestrictionRule{
					os.RestrictionRule{
						Name:     "mywebsite.com",
						Referrer: "www.mywebsite.com",
					},
				},
			},
		},
		FlavorID: "cdn",
		Status:   "deployed",
		Errors:   []os.Error{},
		Links: []gophercloud.Link{
			gophercloud.Link{
				Href: "https://global.cdn.api.rackspacecloud.com/v1.0/110011/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0",
				Rel:  "self",
			},
			gophercloud.Link{
				Href: "blog.mywebsite.com.cdn1.raxcdn.com",
				Rel:  "access_url",
			},
			gophercloud.Link{
				Href: "https://global.cdn.api.rackspacecloud.com/v1.0/110011/flavors/cdn",
				Rel:  "flavor",
			},
		},
	}

	actual, err := Get(fake.ServiceClient(), "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestSuccessfulUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	os.HandleUpdateCDNServiceSuccessfully(t)

	expected := "https://www.poppycdn.io/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0"
	ops := []os.Patch{
		// Append a single Domain
		os.Append{Value: os.Domain{Domain: "appended.mocksite4.com"}},
		// Insert a single Domain
		os.Insertion{
			Index: 4,
			Value: os.Domain{Domain: "inserted.mocksite4.com"},
		},
		// Bulk addition
		os.Append{
			Value: os.DomainList{
				os.Domain{Domain: "bulkadded1.mocksite4.com"},
				os.Domain{Domain: "bulkadded2.mocksite4.com"},
			},
		},
		// Replace a single Origin
		os.Replacement{
			Index: 2,
			Value: os.Origin{Origin: "44.33.22.11", Port: 80, SSL: false},
		},
		// Bulk replace Origins
		os.Replacement{
			Index: 0, // Ignored
			Value: os.OriginList{
				os.Origin{Origin: "44.33.22.11", Port: 80, SSL: false},
				os.Origin{Origin: "55.44.33.22", Port: 443, SSL: true},
			},
		},
		// Remove a single CacheRule
		os.Removal{
			Index: 8,
			Path:  os.PathCaching,
		},
		// Bulk removal
		os.Removal{
			All:  true,
			Path: os.PathCaching,
		},
		// Service name replacement
		os.NameReplacement{
			NewName: "differentServiceName",
		},
	}

	actual, err := Update(fake.ServiceClient(), "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0", ops).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, expected, actual)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	os.HandleDeleteCDNServiceSuccessfully(t)

	err := Delete(fake.ServiceClient(), "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0").ExtractErr()
	th.AssertNoErr(t, err)
}
