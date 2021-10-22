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

		fmt.Fprintf(w, SubnetListResult)
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
			Subnet1,
			Subnet2,
			Subnet3,
			Subnet4,
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

		fmt.Fprintf(w, SubnetGetResult)
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
	th.AssertEquals(t, s.SubnetPoolID, "b80340c7-9960-4f67-a99c-02501656284b")
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, SubnetCreateRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, SubnetCreateResult)
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
		SubnetPoolID: "b80340c7-9960-4f67-a99c-02501656284b",
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
	th.AssertEquals(t, s.SubnetPoolID, "b80340c7-9960-4f67-a99c-02501656284b")
}

func TestCreateNoGateway(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, SubnetCreateWithNoGatewayRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, SubnetCreateWithNoGatewayResponse)
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
		th.TestJSONRequest(t, r, SubnetCreateWithDefaultGatewayRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, SubnetCreateWithDefaultGatewayResponse)
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

func TestCreateIPv6RaAddressMode(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, SubnetCreateWithIPv6RaAddressModeRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, SubnetCreateWithIPv6RaAddressModeResponse)
	})

	var gatewayIP = "2001:db8:0:a::1"
	opts := subnets.CreateOpts{
		NetworkID:       "d32019d3-bc6e-4319-9c1d-6722fc136a22",
		IPVersion:       6,
		CIDR:            "2001:db8:0:a:0:0:0:0/64",
		GatewayIP:       &gatewayIP,
		IPv6AddressMode: "slaac",
		IPv6RAMode:      "slaac",
	}
	s, err := subnets.Create(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "")
	th.AssertEquals(t, s.EnableDHCP, true)
	th.AssertEquals(t, s.NetworkID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
	th.AssertEquals(t, s.TenantID, "4fd44f30292945e481c7b8a0c8908869")
	th.AssertEquals(t, s.IPVersion, 6)
	th.AssertEquals(t, s.GatewayIP, "2001:db8:0:a::1")
	th.AssertEquals(t, s.CIDR, "2001:db8:0:a:0:0:0:0/64")
	th.AssertEquals(t, s.ID, "3b80198d-4f7b-4f77-9ef5-774d54e17126")
	th.AssertEquals(t, s.IPv6AddressMode, "slaac")
	th.AssertEquals(t, s.IPv6RAMode, "slaac")
}

func TestCreateWithNoCIDR(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, SubnetCreateRequestWithNoCIDR)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, SubnetCreateResult)
	})

	opts := subnets.CreateOpts{
		NetworkID:      "d32019d3-bc6e-4319-9c1d-6722fc136a22",
		IPVersion:      4,
		DNSNameservers: []string{"foo"},
		HostRoutes: []subnets.HostRoute{
			{NextHop: "bar"},
		},
		SubnetPoolID: "b80340c7-9960-4f67-a99c-02501656284b",
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
	th.AssertEquals(t, s.SubnetPoolID, "b80340c7-9960-4f67-a99c-02501656284b")
}

func TestCreateWithPrefixlen(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, SubnetCreateRequestWithPrefixlen)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, SubnetCreateResult)
	})

	opts := subnets.CreateOpts{
		NetworkID:      "d32019d3-bc6e-4319-9c1d-6722fc136a22",
		IPVersion:      4,
		DNSNameservers: []string{"foo"},
		HostRoutes: []subnets.HostRoute{
			{NextHop: "bar"},
		},
		SubnetPoolID: "b80340c7-9960-4f67-a99c-02501656284b",
		Prefixlen:    12,
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
	th.AssertEquals(t, s.SubnetPoolID, "b80340c7-9960-4f67-a99c-02501656284b")
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
		th.TestJSONRequest(t, r, SubnetUpdateRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, SubnetUpdateResponse)
	})

	dnsNameservers := []string{"foo"}
	name := "my_new_subnet"
	opts := subnets.UpdateOpts{
		Name:           &name,
		DNSNameservers: &dnsNameservers,
		HostRoutes: &[]subnets.HostRoute{
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
		th.TestJSONRequest(t, r, SubnetUpdateGatewayRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, SubnetUpdateGatewayResponse)
	})

	var gatewayIP = "10.0.0.1"
	name := "my_new_subnet"
	opts := subnets.UpdateOpts{
		Name:      &name,
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
		th.TestJSONRequest(t, r, SubnetUpdateRemoveGatewayRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, SubnetUpdateRemoveGatewayResponse)
	})

	var noGateway = ""
	name := "my_new_subnet"
	opts := subnets.UpdateOpts{
		Name:      &name,
		GatewayIP: &noGateway,
	}
	s, err := subnets.Update(fake.ServiceClient(), "08eae331-0402-425a-923c-34f7cfe39c1b", opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "my_new_subnet")
	th.AssertEquals(t, s.ID, "08eae331-0402-425a-923c-34f7cfe39c1b")
	th.AssertEquals(t, s.GatewayIP, "")
}

func TestUpdateHostRoutes(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets/08eae331-0402-425a-923c-34f7cfe39c1b", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, SubnetUpdateHostRoutesRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, SubnetUpdateHostRoutesResponse)
	})

	HostRoutes := []subnets.HostRoute{
		{
			DestinationCIDR: "192.168.1.1/24",
			NextHop:         "bar",
		},
	}

	name := "my_new_subnet"
	opts := subnets.UpdateOpts{
		Name:       &name,
		HostRoutes: &HostRoutes,
	}
	s, err := subnets.Update(fake.ServiceClient(), "08eae331-0402-425a-923c-34f7cfe39c1b", opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "my_new_subnet")
	th.AssertEquals(t, s.ID, "08eae331-0402-425a-923c-34f7cfe39c1b")
	th.AssertDeepEquals(t, s.HostRoutes, HostRoutes)
}

func TestUpdateRemoveHostRoutes(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets/08eae331-0402-425a-923c-34f7cfe39c1b", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, SubnetUpdateRemoveHostRoutesRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, SubnetUpdateRemoveHostRoutesResponse)
	})

	noHostRoutes := []subnets.HostRoute{}
	opts := subnets.UpdateOpts{
		HostRoutes: &noHostRoutes,
	}
	s, err := subnets.Update(fake.ServiceClient(), "08eae331-0402-425a-923c-34f7cfe39c1b", opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "my_new_subnet")
	th.AssertEquals(t, s.ID, "08eae331-0402-425a-923c-34f7cfe39c1b")
	th.AssertDeepEquals(t, s.HostRoutes, noHostRoutes)
}

func TestUpdateAllocationPool(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/subnets/08eae331-0402-425a-923c-34f7cfe39c1b", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, SubnetUpdateAllocationPoolRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, SubnetUpdateAllocationPoolResponse)
	})

	name := "my_new_subnet"
	opts := subnets.UpdateOpts{
		Name: &name,
		AllocationPools: []subnets.AllocationPool{
			{
				Start: "10.1.0.2",
				End:   "10.1.0.254",
			},
		},
	}
	s, err := subnets.Update(fake.ServiceClient(), "08eae331-0402-425a-923c-34f7cfe39c1b", opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "my_new_subnet")
	th.AssertEquals(t, s.ID, "08eae331-0402-425a-923c-34f7cfe39c1b")
	th.AssertDeepEquals(t, s.AllocationPools, []subnets.AllocationPool{
		{
			Start: "10.1.0.2",
			End:   "10.1.0.254",
		},
	})
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
