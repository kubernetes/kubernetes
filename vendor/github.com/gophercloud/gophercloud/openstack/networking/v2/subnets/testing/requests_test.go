package testing

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/subnets"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "subnets": [
        {
            "name": "private-subnet",
            "enable_dhcp": true,
            "network_id": "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
            "tenant_id": "26a7980765d0414dbc1fc1f88cdb7e6e",
            "dns_nameservers": [],
            "allocation_pools": [
                {
                    "start": "10.0.0.2",
                    "end": "10.0.0.254"
                }
            ],
            "host_routes": [],
            "ip_version": 4,
            "gateway_ip": "10.0.0.1",
            "cidr": "10.0.0.0/24",
            "id": "08eae331-0402-425a-923c-34f7cfe39c1b"
        },
        {
            "name": "my_subnet",
            "enable_dhcp": true,
            "network_id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
            "tenant_id": "4fd44f30292945e481c7b8a0c8908869",
            "dns_nameservers": [],
            "allocation_pools": [
                {
                    "start": "192.0.0.2",
                    "end": "192.255.255.254"
                }
            ],
            "host_routes": [],
            "ip_version": 4,
            "gateway_ip": "192.0.0.1",
            "cidr": "192.0.0.0/8",
            "id": "54d6f61d-db07-451c-9ab3-b9609b6b6f0b"
        },
        {
            "name": "my_gatewayless_subnet",
            "enable_dhcp": true,
            "network_id": "d32019d3-bc6e-4319-9c1d-6722fc136a23",
            "tenant_id": "4fd44f30292945e481c7b8a0c8908869",
            "dns_nameservers": [],
            "allocation_pools": [
                {
                    "start": "192.168.1.2",
                    "end": "192.168.1.254"
                }
            ],
            "host_routes": [],
            "ip_version": 4,
            "gateway_ip": null,
            "cidr": "192.168.1.0/24",
            "id": "54d6f61d-db07-451c-9ab3-b9609b6b6f0c"
        }
    ]
}
      `)
	})

	count := 0

	subnets.List(fake.ServiceClient(), subnets.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := subnets.ExtractSubnets(page)
		if err != nil {
			t.Errorf("Failed to extract subnets: %v", err)
			return false, nil
		}

		expected := []subnets.Subnet{
			{
				Name:           "private-subnet",
				EnableDHCP:     true,
				NetworkID:      "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
				TenantID:       "26a7980765d0414dbc1fc1f88cdb7e6e",
				DNSNameservers: []string{},
				AllocationPools: []subnets.AllocationPool{
					{
						Start: "10.0.0.2",
						End:   "10.0.0.254",
					},
				},
				HostRoutes: []subnets.HostRoute{},
				IPVersion:  4,
				GatewayIP:  "10.0.0.1",
				CIDR:       "10.0.0.0/24",
				ID:         "08eae331-0402-425a-923c-34f7cfe39c1b",
			},
			{
				Name:           "my_subnet",
				EnableDHCP:     true,
				NetworkID:      "d32019d3-bc6e-4319-9c1d-6722fc136a22",
				TenantID:       "4fd44f30292945e481c7b8a0c8908869",
				DNSNameservers: []string{},
				AllocationPools: []subnets.AllocationPool{
					{
						Start: "192.0.0.2",
						End:   "192.255.255.254",
					},
				},
				HostRoutes: []subnets.HostRoute{},
				IPVersion:  4,
				GatewayIP:  "192.0.0.1",
				CIDR:       "192.0.0.0/8",
				ID:         "54d6f61d-db07-451c-9ab3-b9609b6b6f0b",
			},
			subnets.Subnet{
				Name:           "my_gatewayless_subnet",
				EnableDHCP:     true,
				NetworkID:      "d32019d3-bc6e-4319-9c1d-6722fc136a23",
				TenantID:       "4fd44f30292945e481c7b8a0c8908869",
				DNSNameservers: []string{},
				AllocationPools: []subnets.AllocationPool{
					{
						Start: "192.168.1.2",
						End:   "192.168.1.254",
					},
				},
				HostRoutes: []subnets.HostRoute{},
				IPVersion:  4,
				GatewayIP:  "",
				CIDR:       "192.168.1.0/24",
				ID:         "54d6f61d-db07-451c-9ab3-b9609b6b6f0c",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets/54d6f61d-db07-451c-9ab3-b9609b6b6f0b", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "subnet": {
        "name": "my_subnet",
        "enable_dhcp": true,
        "network_id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
        "tenant_id": "4fd44f30292945e481c7b8a0c8908869",
        "dns_nameservers": [],
        "allocation_pools": [
            {
                "start": "192.0.0.2",
                "end": "192.255.255.254"
            }
        ],
        "host_routes": [],
        "ip_version": 4,
        "gateway_ip": "192.0.0.1",
        "cidr": "192.0.0.0/8",
        "id": "54d6f61d-db07-451c-9ab3-b9609b6b6f0b"
    }
}
			`)
	})

	s, err := subnets.Get(fake.ServiceClient(), "54d6f61d-db07-451c-9ab3-b9609b6b6f0b").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "my_subnet")
	th.AssertEquals(t, s.EnableDHCP, true)
	th.AssertEquals(t, s.NetworkID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
	th.AssertEquals(t, s.TenantID, "4fd44f30292945e481c7b8a0c8908869")
	th.AssertDeepEquals(t, s.DNSNameservers, []string{})
	th.AssertDeepEquals(t, s.AllocationPools, []subnets.AllocationPool{
		{
			Start: "192.0.0.2",
			End:   "192.255.255.254",
		},
	})
	th.AssertDeepEquals(t, s.HostRoutes, []subnets.HostRoute{})
	th.AssertEquals(t, s.IPVersion, 4)
	th.AssertEquals(t, s.GatewayIP, "192.0.0.1")
	th.AssertEquals(t, s.CIDR, "192.0.0.0/8")
	th.AssertEquals(t, s.ID, "54d6f61d-db07-451c-9ab3-b9609b6b6f0b")
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "subnet": {
        "network_id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
        "ip_version": 4,
        "gateway_ip": "192.168.199.1",
        "cidr": "192.168.199.0/24",
        "dns_nameservers": ["foo"],
        "allocation_pools": [
            {
                "start": "192.168.199.2",
                "end": "192.168.199.254"
            }
        ],
        "host_routes": [{"destination":"","nexthop": "bar"}]
    }
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "subnet": {
        "name": "",
        "enable_dhcp": true,
        "network_id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
        "tenant_id": "4fd44f30292945e481c7b8a0c8908869",
        "dns_nameservers": [],
        "allocation_pools": [
            {
                "start": "192.168.199.2",
                "end": "192.168.199.254"
            }
        ],
        "host_routes": [],
        "ip_version": 4,
        "gateway_ip": "192.168.199.1",
        "cidr": "192.168.199.0/24",
        "id": "3b80198d-4f7b-4f77-9ef5-774d54e17126"
    }
}
		`)
	})

	var gatewayIP = "192.168.199.1"
	opts := subnets.CreateOpts{
		NetworkID: "d32019d3-bc6e-4319-9c1d-6722fc136a22",
		IPVersion: 4,
		CIDR:      "192.168.199.0/24",
		GatewayIP: &gatewayIP,
		AllocationPools: []subnets.AllocationPool{
			{
				Start: "192.168.199.2",
				End:   "192.168.199.254",
			},
		},
		DNSNameservers: []string{"foo"},
		HostRoutes: []subnets.HostRoute{
			{NextHop: "bar"},
		},
	}
	s, err := subnets.Create(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "")
	th.AssertEquals(t, s.EnableDHCP, true)
	th.AssertEquals(t, s.NetworkID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
	th.AssertEquals(t, s.TenantID, "4fd44f30292945e481c7b8a0c8908869")
	th.AssertDeepEquals(t, s.DNSNameservers, []string{})
	th.AssertDeepEquals(t, s.AllocationPools, []subnets.AllocationPool{
		{
			Start: "192.168.199.2",
			End:   "192.168.199.254",
		},
	})
	th.AssertDeepEquals(t, s.HostRoutes, []subnets.HostRoute{})
	th.AssertEquals(t, s.IPVersion, 4)
	th.AssertEquals(t, s.GatewayIP, "192.168.199.1")
	th.AssertEquals(t, s.CIDR, "192.168.199.0/24")
	th.AssertEquals(t, s.ID, "3b80198d-4f7b-4f77-9ef5-774d54e17126")
}

func TestCreateNoGateway(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "subnet": {
        "network_id": "d32019d3-bc6e-4319-9c1d-6722fc136a23",
        "ip_version": 4,
        "cidr": "192.168.1.0/24",
        "gateway_ip": null,
        "allocation_pools": [
            {
                "start": "192.168.1.2",
                "end": "192.168.1.254"
            }
        ]
    }
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "subnet": {
        "name": "",
        "enable_dhcp": true,
        "network_id": "d32019d3-bc6e-4319-9c1d-6722fc136a23",
        "tenant_id": "4fd44f30292945e481c7b8a0c8908869",
        "allocation_pools": [
            {
                "start": "192.168.1.2",
                "end": "192.168.1.254"
            }
        ],
        "host_routes": [],
        "ip_version": 4,
        "gateway_ip": null,
        "cidr": "192.168.1.0/24",
        "id": "54d6f61d-db07-451c-9ab3-b9609b6b6f0c"
    }
}
		`)
	})

	var noGateway = ""
	opts := subnets.CreateOpts{
		NetworkID: "d32019d3-bc6e-4319-9c1d-6722fc136a23",
		IPVersion: 4,
		CIDR:      "192.168.1.0/24",
		GatewayIP: &noGateway,
		AllocationPools: []subnets.AllocationPool{
			{
				Start: "192.168.1.2",
				End:   "192.168.1.254",
			},
		},
		DNSNameservers: []string{},
	}
	s, err := subnets.Create(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "")
	th.AssertEquals(t, s.EnableDHCP, true)
	th.AssertEquals(t, s.NetworkID, "d32019d3-bc6e-4319-9c1d-6722fc136a23")
	th.AssertEquals(t, s.TenantID, "4fd44f30292945e481c7b8a0c8908869")
	th.AssertDeepEquals(t, s.AllocationPools, []subnets.AllocationPool{
		{
			Start: "192.168.1.2",
			End:   "192.168.1.254",
		},
	})
	th.AssertDeepEquals(t, s.HostRoutes, []subnets.HostRoute{})
	th.AssertEquals(t, s.IPVersion, 4)
	th.AssertEquals(t, s.GatewayIP, "")
	th.AssertEquals(t, s.CIDR, "192.168.1.0/24")
	th.AssertEquals(t, s.ID, "54d6f61d-db07-451c-9ab3-b9609b6b6f0c")
}

func TestCreateDefaultGateway(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "subnet": {
        "network_id": "d32019d3-bc6e-4319-9c1d-6722fc136a23",
        "ip_version": 4,
        "cidr": "192.168.1.0/24",
        "allocation_pools": [
            {
                "start": "192.168.1.2",
                "end": "192.168.1.254"
            }
        ]
    }
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "subnet": {
        "name": "",
        "enable_dhcp": true,
        "network_id": "d32019d3-bc6e-4319-9c1d-6722fc136a23",
        "tenant_id": "4fd44f30292945e481c7b8a0c8908869",
        "allocation_pools": [
            {
                "start": "192.168.1.2",
                "end": "192.168.1.254"
            }
        ],
        "host_routes": [],
        "ip_version": 4,
        "gateway_ip": "192.168.1.1",
        "cidr": "192.168.1.0/24",
        "id": "54d6f61d-db07-451c-9ab3-b9609b6b6f0c"
    }
}
		`)
	})

	opts := subnets.CreateOpts{
		NetworkID: "d32019d3-bc6e-4319-9c1d-6722fc136a23",
		IPVersion: 4,
		CIDR:      "192.168.1.0/24",
		AllocationPools: []subnets.AllocationPool{
			{
				Start: "192.168.1.2",
				End:   "192.168.1.254",
			},
		},
		DNSNameservers: []string{},
	}
	s, err := subnets.Create(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "")
	th.AssertEquals(t, s.EnableDHCP, true)
	th.AssertEquals(t, s.NetworkID, "d32019d3-bc6e-4319-9c1d-6722fc136a23")
	th.AssertEquals(t, s.TenantID, "4fd44f30292945e481c7b8a0c8908869")
	th.AssertDeepEquals(t, s.AllocationPools, []subnets.AllocationPool{
		{
			Start: "192.168.1.2",
			End:   "192.168.1.254",
		},
	})
	th.AssertDeepEquals(t, s.HostRoutes, []subnets.HostRoute{})
	th.AssertEquals(t, s.IPVersion, 4)
	th.AssertEquals(t, s.GatewayIP, "192.168.1.1")
	th.AssertEquals(t, s.CIDR, "192.168.1.0/24")
	th.AssertEquals(t, s.ID, "54d6f61d-db07-451c-9ab3-b9609b6b6f0c")
}

func TestRequiredCreateOpts(t *testing.T) {
	res := subnets.Create(fake.ServiceClient(), subnets.CreateOpts{})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}

	res = subnets.Create(fake.ServiceClient(), subnets.CreateOpts{NetworkID: "foo"})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}

	res = subnets.Create(fake.ServiceClient(), subnets.CreateOpts{NetworkID: "foo", CIDR: "bar", IPVersion: 40})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets/08eae331-0402-425a-923c-34f7cfe39c1b", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "subnet": {
        "name": "my_new_subnet",
        "dns_nameservers": ["foo"],
        "host_routes": [{"destination":"","nexthop": "bar"}]
    }
}
		`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "subnet": {
        "name": "my_new_subnet",
        "enable_dhcp": true,
        "network_id": "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
        "tenant_id": "26a7980765d0414dbc1fc1f88cdb7e6e",
        "dns_nameservers": [],
        "allocation_pools": [
            {
                "start": "10.0.0.2",
                "end": "10.0.0.254"
            }
        ],
        "host_routes": [],
        "ip_version": 4,
        "gateway_ip": "10.0.0.1",
        "cidr": "10.0.0.0/24",
        "id": "08eae331-0402-425a-923c-34f7cfe39c1b"
    }
}
	`)
	})

	opts := subnets.UpdateOpts{
		Name:           "my_new_subnet",
		DNSNameservers: []string{"foo"},
		HostRoutes: []subnets.HostRoute{
			{NextHop: "bar"},
		},
	}
	s, err := subnets.Update(fake.ServiceClient(), "08eae331-0402-425a-923c-34f7cfe39c1b", opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "my_new_subnet")
	th.AssertEquals(t, s.ID, "08eae331-0402-425a-923c-34f7cfe39c1b")
}

func TestUpdateGateway(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets/08eae331-0402-425a-923c-34f7cfe39c1b", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "subnet": {
        "name": "my_new_subnet",
        "gateway_ip": "10.0.0.1"
    }
}
		`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "subnet": {
        "name": "my_new_subnet",
        "enable_dhcp": true,
        "network_id": "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
        "tenant_id": "26a7980765d0414dbc1fc1f88cdb7e6e",
        "dns_nameservers": [],
        "allocation_pools": [
            {
                "start": "10.0.0.2",
                "end": "10.0.0.254"
            }
        ],
        "host_routes": [],
        "ip_version": 4,
        "gateway_ip": "10.0.0.1",
        "cidr": "10.0.0.0/24",
        "id": "08eae331-0402-425a-923c-34f7cfe39c1b"
    }
}
	`)
	})

	var gatewayIP = "10.0.0.1"
	opts := subnets.UpdateOpts{
		Name:      "my_new_subnet",
		GatewayIP: &gatewayIP,
	}
	s, err := subnets.Update(fake.ServiceClient(), "08eae331-0402-425a-923c-34f7cfe39c1b", opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "my_new_subnet")
	th.AssertEquals(t, s.ID, "08eae331-0402-425a-923c-34f7cfe39c1b")
	th.AssertEquals(t, s.GatewayIP, "10.0.0.1")
}

func TestUpdateRemoveGateway(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets/08eae331-0402-425a-923c-34f7cfe39c1b", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "subnet": {
        "name": "my_new_subnet",
        "gateway_ip": null
    }
}
		`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "subnet": {
        "name": "my_new_subnet",
        "enable_dhcp": true,
        "network_id": "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
        "tenant_id": "26a7980765d0414dbc1fc1f88cdb7e6e",
        "dns_nameservers": [],
        "allocation_pools": [
            {
                "start": "10.0.0.2",
                "end": "10.0.0.254"
            }
        ],
        "host_routes": [],
        "ip_version": 4,
        "gateway_ip": null,
        "cidr": "10.0.0.0/24",
        "id": "08eae331-0402-425a-923c-34f7cfe39c1b"
    }
}
	`)
	})

	var noGateway = ""
	opts := subnets.UpdateOpts{
		Name:      "my_new_subnet",
		GatewayIP: &noGateway,
	}
	s, err := subnets.Update(fake.ServiceClient(), "08eae331-0402-425a-923c-34f7cfe39c1b", opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "my_new_subnet")
	th.AssertEquals(t, s.ID, "08eae331-0402-425a-923c-34f7cfe39c1b")
	th.AssertEquals(t, s.GatewayIP, "")
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets/08eae331-0402-425a-923c-34f7cfe39c1b", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := subnets.Delete(fake.ServiceClient(), "08eae331-0402-425a-923c-34f7cfe39c1b")
	th.AssertNoErr(t, res.Err)
}
