package testing

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/ports", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, ListResponse)
	})

	count := 0

	ports.List(fake.ServiceClient(), ports.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ports.ExtractPorts(page)
		if err != nil {
			t.Errorf("Failed to extract subnets: %v", err)
			return false, nil
		}

		expected := []ports.Port{
			{
				Status:       "ACTIVE",
				Name:         "",
				AdminStateUp: true,
				NetworkID:    "70c1db1f-b701-45bd-96e0-a313ee3430b3",
				TenantID:     "",
				DeviceOwner:  "network:router_gateway",
				MACAddress:   "fa:16:3e:58:42:ed",
				FixedIPs: []ports.IP{
					{
						SubnetID:  "008ba151-0b8c-4a67-98b5-0d2b87666062",
						IPAddress: "172.24.4.2",
					},
				},
				ID:             "d80b1a3b-4fc1-49f3-952e-1e2ab7081d8b",
				SecurityGroups: []string{},
				DeviceID:       "9ae135f4-b6e0-4dad-9e91-3c223e385824",
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

	th.Mux.HandleFunc("/v2.0/ports/46d4bfb9-b26e-41f3-bd2e-e6dcc1ccedb2", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, GetResponse)
	})

	n, err := ports.Get(fake.ServiceClient(), "46d4bfb9-b26e-41f3-bd2e-e6dcc1ccedb2").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Status, "ACTIVE")
	th.AssertEquals(t, n.Name, "")
	th.AssertEquals(t, n.AdminStateUp, true)
	th.AssertEquals(t, n.NetworkID, "a87cc70a-3e15-4acf-8205-9b711a3531b7")
	th.AssertEquals(t, n.TenantID, "7e02058126cc4950b75f9970368ba177")
	th.AssertEquals(t, n.DeviceOwner, "network:router_interface")
	th.AssertEquals(t, n.MACAddress, "fa:16:3e:23:fd:d7")
	th.AssertDeepEquals(t, n.FixedIPs, []ports.IP{
		{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.1"},
	})
	th.AssertEquals(t, n.ID, "46d4bfb9-b26e-41f3-bd2e-e6dcc1ccedb2")
	th.AssertDeepEquals(t, n.SecurityGroups, []string{})
	th.AssertEquals(t, n.Status, "ACTIVE")
	th.AssertEquals(t, n.DeviceID, "5e3898d7-11be-483e-9732-b2f5eccd2b2e")
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/ports", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, CreateRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, CreateResponse)
	})

	asu := true
	options := ports.CreateOpts{
		Name:         "private-port",
		AdminStateUp: &asu,
		NetworkID:    "a87cc70a-3e15-4acf-8205-9b711a3531b7",
		FixedIPs: []ports.IP{
			{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.2"},
		},
		SecurityGroups: &[]string{"foo"},
		AllowedAddressPairs: []ports.AddressPair{
			{IPAddress: "10.0.0.4", MACAddress: "fa:16:3e:c9:cb:f0"},
		},
	}
	n, err := ports.Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Status, "DOWN")
	th.AssertEquals(t, n.Name, "private-port")
	th.AssertEquals(t, n.AdminStateUp, true)
	th.AssertEquals(t, n.NetworkID, "a87cc70a-3e15-4acf-8205-9b711a3531b7")
	th.AssertEquals(t, n.TenantID, "d6700c0c9ffa4f1cb322cd4a1f3906fa")
	th.AssertEquals(t, n.DeviceOwner, "")
	th.AssertEquals(t, n.MACAddress, "fa:16:3e:c9:cb:f0")
	th.AssertDeepEquals(t, n.FixedIPs, []ports.IP{
		{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.2"},
	})
	th.AssertEquals(t, n.ID, "65c0ee9f-d634-4522-8954-51021b570b0d")
	th.AssertDeepEquals(t, n.SecurityGroups, []string{"f0ac4394-7e4a-4409-9701-ba8be283dbc3"})
	th.AssertDeepEquals(t, n.AllowedAddressPairs, []ports.AddressPair{
		{IPAddress: "10.0.0.4", MACAddress: "fa:16:3e:c9:cb:f0"},
	})
}

func TestCreateOmitSecurityGroups(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/ports", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, CreateOmitSecurityGroupsRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, CreateOmitSecurityGroupsResponse)
	})

	asu := true
	options := ports.CreateOpts{
		Name:         "private-port",
		AdminStateUp: &asu,
		NetworkID:    "a87cc70a-3e15-4acf-8205-9b711a3531b7",
		FixedIPs: []ports.IP{
			{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.2"},
		},
		AllowedAddressPairs: []ports.AddressPair{
			{IPAddress: "10.0.0.4", MACAddress: "fa:16:3e:c9:cb:f0"},
		},
	}
	n, err := ports.Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Status, "DOWN")
	th.AssertEquals(t, n.Name, "private-port")
	th.AssertEquals(t, n.AdminStateUp, true)
	th.AssertEquals(t, n.NetworkID, "a87cc70a-3e15-4acf-8205-9b711a3531b7")
	th.AssertEquals(t, n.TenantID, "d6700c0c9ffa4f1cb322cd4a1f3906fa")
	th.AssertEquals(t, n.DeviceOwner, "")
	th.AssertEquals(t, n.MACAddress, "fa:16:3e:c9:cb:f0")
	th.AssertDeepEquals(t, n.FixedIPs, []ports.IP{
		{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.2"},
	})
	th.AssertEquals(t, n.ID, "65c0ee9f-d634-4522-8954-51021b570b0d")
	th.AssertDeepEquals(t, n.SecurityGroups, []string{"f0ac4394-7e4a-4409-9701-ba8be283dbc3"})
	th.AssertDeepEquals(t, n.AllowedAddressPairs, []ports.AddressPair{
		{IPAddress: "10.0.0.4", MACAddress: "fa:16:3e:c9:cb:f0"},
	})
}

func TestCreateWithNoSecurityGroup(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/ports", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, CreateWithNoSecurityGroupsRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, CreateWithNoSecurityGroupsResponse)
	})

	asu := true
	options := ports.CreateOpts{
		Name:         "private-port",
		AdminStateUp: &asu,
		NetworkID:    "a87cc70a-3e15-4acf-8205-9b711a3531b7",
		FixedIPs: []ports.IP{
			{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.2"},
		},
		SecurityGroups: &[]string{},
		AllowedAddressPairs: []ports.AddressPair{
			{IPAddress: "10.0.0.4", MACAddress: "fa:16:3e:c9:cb:f0"},
		},
	}
	n, err := ports.Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Status, "DOWN")
	th.AssertEquals(t, n.Name, "private-port")
	th.AssertEquals(t, n.AdminStateUp, true)
	th.AssertEquals(t, n.NetworkID, "a87cc70a-3e15-4acf-8205-9b711a3531b7")
	th.AssertEquals(t, n.TenantID, "d6700c0c9ffa4f1cb322cd4a1f3906fa")
	th.AssertEquals(t, n.DeviceOwner, "")
	th.AssertEquals(t, n.MACAddress, "fa:16:3e:c9:cb:f0")
	th.AssertDeepEquals(t, n.FixedIPs, []ports.IP{
		{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.2"},
	})
	th.AssertEquals(t, n.ID, "65c0ee9f-d634-4522-8954-51021b570b0d")
	th.AssertDeepEquals(t, n.AllowedAddressPairs, []ports.AddressPair{
		{IPAddress: "10.0.0.4", MACAddress: "fa:16:3e:c9:cb:f0"},
	})
}

func TestRequiredCreateOpts(t *testing.T) {
	res := ports.Create(fake.ServiceClient(), ports.CreateOpts{})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/ports/65c0ee9f-d634-4522-8954-51021b570b0d", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, UpdateRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, UpdateResponse)
	})

	options := ports.UpdateOpts{
		Name: "new_port_name",
		FixedIPs: []ports.IP{
			{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
		},
		SecurityGroups: &[]string{"f0ac4394-7e4a-4409-9701-ba8be283dbc3"},
		AllowedAddressPairs: &[]ports.AddressPair{
			{IPAddress: "10.0.0.4", MACAddress: "fa:16:3e:c9:cb:f0"},
		},
	}

	s, err := ports.Update(fake.ServiceClient(), "65c0ee9f-d634-4522-8954-51021b570b0d", options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "new_port_name")
	th.AssertDeepEquals(t, s.FixedIPs, []ports.IP{
		{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
	})
	th.AssertDeepEquals(t, s.AllowedAddressPairs, []ports.AddressPair{
		{IPAddress: "10.0.0.4", MACAddress: "fa:16:3e:c9:cb:f0"},
	})
	th.AssertDeepEquals(t, s.SecurityGroups, []string{"f0ac4394-7e4a-4409-9701-ba8be283dbc3"})
}

func TestUpdateOmitSecurityGroups(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/ports/65c0ee9f-d634-4522-8954-51021b570b0d", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, UpdateOmitSecurityGroupsRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, UpdateOmitSecurityGroupsResponse)
	})

	options := ports.UpdateOpts{
		Name: "new_port_name",
		FixedIPs: []ports.IP{
			{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
		},
		AllowedAddressPairs: &[]ports.AddressPair{
			{IPAddress: "10.0.0.4", MACAddress: "fa:16:3e:c9:cb:f0"},
		},
	}

	s, err := ports.Update(fake.ServiceClient(), "65c0ee9f-d634-4522-8954-51021b570b0d", options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "new_port_name")
	th.AssertDeepEquals(t, s.FixedIPs, []ports.IP{
		{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
	})
	th.AssertDeepEquals(t, s.AllowedAddressPairs, []ports.AddressPair{
		{IPAddress: "10.0.0.4", MACAddress: "fa:16:3e:c9:cb:f0"},
	})
	th.AssertDeepEquals(t, s.SecurityGroups, []string{"f0ac4394-7e4a-4409-9701-ba8be283dbc3"})
}

func TestRemoveSecurityGroups(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/ports/65c0ee9f-d634-4522-8954-51021b570b0d", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, RemoveSecurityGroupRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, RemoveSecurityGroupResponse)
	})

	options := ports.UpdateOpts{
		Name: "new_port_name",
		FixedIPs: []ports.IP{
			{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
		},
		SecurityGroups: &[]string{},
		AllowedAddressPairs: &[]ports.AddressPair{
			{IPAddress: "10.0.0.4", MACAddress: "fa:16:3e:c9:cb:f0"},
		},
	}

	s, err := ports.Update(fake.ServiceClient(), "65c0ee9f-d634-4522-8954-51021b570b0d", options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "new_port_name")
	th.AssertDeepEquals(t, s.FixedIPs, []ports.IP{
		{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
	})
	th.AssertDeepEquals(t, s.AllowedAddressPairs, []ports.AddressPair{
		{IPAddress: "10.0.0.4", MACAddress: "fa:16:3e:c9:cb:f0"},
	})
	th.AssertDeepEquals(t, s.SecurityGroups, []string(nil))
}

func TestRemoveAllowedAddressPairs(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/ports/65c0ee9f-d634-4522-8954-51021b570b0d", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, RemoveAllowedAddressPairsRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, RemoveAllowedAddressPairsResponse)
	})

	options := ports.UpdateOpts{
		Name: "new_port_name",
		FixedIPs: []ports.IP{
			{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
		},
		SecurityGroups:      &[]string{"f0ac4394-7e4a-4409-9701-ba8be283dbc3"},
		AllowedAddressPairs: &[]ports.AddressPair{},
	}

	s, err := ports.Update(fake.ServiceClient(), "65c0ee9f-d634-4522-8954-51021b570b0d", options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "new_port_name")
	th.AssertDeepEquals(t, s.FixedIPs, []ports.IP{
		{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
	})
	th.AssertDeepEquals(t, s.AllowedAddressPairs, []ports.AddressPair(nil))
	th.AssertDeepEquals(t, s.SecurityGroups, []string{"f0ac4394-7e4a-4409-9701-ba8be283dbc3"})
}

func TestDontUpdateAllowedAddressPairs(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/ports/65c0ee9f-d634-4522-8954-51021b570b0d", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, DontUpdateAllowedAddressPairsRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, DontUpdateAllowedAddressPairsResponse)
	})

	options := ports.UpdateOpts{
		Name: "new_port_name",
		FixedIPs: []ports.IP{
			{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
		},
		SecurityGroups: &[]string{"f0ac4394-7e4a-4409-9701-ba8be283dbc3"},
	}

	s, err := ports.Update(fake.ServiceClient(), "65c0ee9f-d634-4522-8954-51021b570b0d", options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "new_port_name")
	th.AssertDeepEquals(t, s.FixedIPs, []ports.IP{
		{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
	})
	th.AssertDeepEquals(t, s.AllowedAddressPairs, []ports.AddressPair{
		{IPAddress: "10.0.0.4", MACAddress: "fa:16:3e:c9:cb:f0"},
	})
	th.AssertDeepEquals(t, s.SecurityGroups, []string{"f0ac4394-7e4a-4409-9701-ba8be283dbc3"})
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/ports/65c0ee9f-d634-4522-8954-51021b570b0d", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := ports.Delete(fake.ServiceClient(), "65c0ee9f-d634-4522-8954-51021b570b0d")
	th.AssertNoErr(t, res.Err)
}
