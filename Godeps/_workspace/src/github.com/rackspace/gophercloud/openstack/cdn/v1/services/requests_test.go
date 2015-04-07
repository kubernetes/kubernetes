package services

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleListCDNServiceSuccessfully(t)

	count := 0

	err := List(fake.ServiceClient(), &ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractServices(page)
		if err != nil {
			t.Errorf("Failed to extract services: %v", err)
			return false, err
		}

		expected := []Service{
			Service{
				ID:   "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0",
				Name: "mywebsite.com",
				Domains: []Domain{
					Domain{
						Domain: "www.mywebsite.com",
					},
				},
				Origins: []Origin{
					Origin{
						Origin: "mywebsite.com",
						Port:   80,
						SSL:    false,
					},
				},
				Caching: []CacheRule{
					CacheRule{
						Name: "default",
						TTL:  3600,
					},
					CacheRule{
						Name: "home",
						TTL:  17200,
						Rules: []TTLRule{
							TTLRule{
								Name:       "index",
								RequestURL: "/index.htm",
							},
						},
					},
					CacheRule{
						Name: "images",
						TTL:  12800,
						Rules: []TTLRule{
							TTLRule{
								Name:       "images",
								RequestURL: "*.png",
							},
						},
					},
				},
				Restrictions: []Restriction{
					Restriction{
						Name: "website only",
						Rules: []RestrictionRule{
							RestrictionRule{
								Name:     "mywebsite.com",
								Referrer: "www.mywebsite.com",
							},
						},
					},
				},
				FlavorID: "asia",
				Status:   "deployed",
				Errors:   []Error{},
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
			Service{
				ID:   "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f1",
				Name: "myothersite.com",
				Domains: []Domain{
					Domain{
						Domain: "www.myothersite.com",
					},
				},
				Origins: []Origin{
					Origin{
						Origin: "44.33.22.11",
						Port:   80,
						SSL:    false,
					},
					Origin{
						Origin: "77.66.55.44",
						Port:   80,
						SSL:    false,
						Rules: []OriginRule{
							OriginRule{
								Name:       "videos",
								RequestURL: "^/videos/*.m3u",
							},
						},
					},
				},
				Caching: []CacheRule{
					CacheRule{
						Name: "default",
						TTL:  3600,
					},
				},
				Restrictions: []Restriction{},
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

	createOpts := CreateOpts{
		Name: "mywebsite.com",
		Domains: []Domain{
			Domain{
				Domain: "www.mywebsite.com",
			},
			Domain{
				Domain: "blog.mywebsite.com",
			},
		},
		Origins: []Origin{
			Origin{
				Origin: "mywebsite.com",
				Port:   80,
				SSL:    false,
			},
		},
		Restrictions: []Restriction{
			Restriction{
				Name: "website only",
				Rules: []RestrictionRule{
					RestrictionRule{
						Name:     "mywebsite.com",
						Referrer: "www.mywebsite.com",
					},
				},
			},
		},
		Caching: []CacheRule{
			CacheRule{
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

	HandleGetCDNServiceSuccessfully(t)

	expected := &Service{
		ID:   "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0",
		Name: "mywebsite.com",
		Domains: []Domain{
			Domain{
				Domain:   "www.mywebsite.com",
				Protocol: "http",
			},
		},
		Origins: []Origin{
			Origin{
				Origin: "mywebsite.com",
				Port:   80,
				SSL:    false,
			},
		},
		Caching: []CacheRule{
			CacheRule{
				Name: "default",
				TTL:  3600,
			},
			CacheRule{
				Name: "home",
				TTL:  17200,
				Rules: []TTLRule{
					TTLRule{
						Name:       "index",
						RequestURL: "/index.htm",
					},
				},
			},
			CacheRule{
				Name: "images",
				TTL:  12800,
				Rules: []TTLRule{
					TTLRule{
						Name:       "images",
						RequestURL: "*.png",
					},
				},
			},
		},
		Restrictions: []Restriction{
			Restriction{
				Name: "website only",
				Rules: []RestrictionRule{
					RestrictionRule{
						Name:     "mywebsite.com",
						Referrer: "www.mywebsite.com",
					},
				},
			},
		},
		FlavorID: "cdn",
		Status:   "deployed",
		Errors:   []Error{},
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

	HandleUpdateCDNServiceSuccessfully(t)

	expected := "https://www.poppycdn.io/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0"
	ops := UpdateOpts{
		// Append a single Domain
		Append{Value: Domain{Domain: "appended.mocksite4.com"}},
		// Insert a single Domain
		Insertion{
			Index: 4,
			Value: Domain{Domain: "inserted.mocksite4.com"},
		},
		// Bulk addition
		Append{
			Value: DomainList{
				Domain{Domain: "bulkadded1.mocksite4.com"},
				Domain{Domain: "bulkadded2.mocksite4.com"},
			},
		},
		// Replace a single Origin
		Replacement{
			Index: 2,
			Value: Origin{Origin: "44.33.22.11", Port: 80, SSL: false},
		},
		// Bulk replace Origins
		Replacement{
			Index: 0, // Ignored
			Value: OriginList{
				Origin{Origin: "44.33.22.11", Port: 80, SSL: false},
				Origin{Origin: "55.44.33.22", Port: 443, SSL: true},
			},
		},
		// Remove a single CacheRule
		Removal{
			Index: 8,
			Path:  PathCaching,
		},
		// Bulk removal
		Removal{
			All:  true,
			Path: PathCaching,
		},
		// Service name replacement
		NameReplacement{
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

	HandleDeleteCDNServiceSuccessfully(t)

	err := Delete(fake.ServiceClient(), "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0").ExtractErr()
	th.AssertNoErr(t, err)
}
