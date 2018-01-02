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
		th.TestJSONRequest(t, r, SubnetUpdateGatewayRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, SubnetUpdateGatewayResponse)
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
		th.TestJSONRequest(t, r, SubnetUpdateRemoveGatewayRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, SubnetUpdateRemoveGatewayResponse)
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

	opts := subnets.UpdateOpts{
		Name: "my_new_subnet",
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
