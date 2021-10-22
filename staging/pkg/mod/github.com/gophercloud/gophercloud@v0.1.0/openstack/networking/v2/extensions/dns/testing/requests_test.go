package testing

import (
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/dns"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/floatingips"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
	th "github.com/gophercloud/gophercloud/testhelper"
)

type PortDNS struct {
	ports.Port
	dns.PortDNSExt
}

type FloatingIPDNS struct {
	floatingips.FloatingIP
	dns.FloatingIPDNSExt
}

type NetworkDNS struct {
	networks.Network
	dns.NetworkDNSExt
}

func TestPortList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	PortHandleListSuccessfully(t)

	var actual []PortDNS

	expected := []PortDNS{
		{
			Port: ports.Port{
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
			PortDNSExt: dns.PortDNSExt{
				DNSName: "test-port",
				DNSAssignment: []map[string]string{
					{
						"hostname":   "test-port",
						"ip_address": "172.24.4.2",
						"fqdn":       "test-port.openstack.local.",
					},
				},
			},
		},
	}

	var listOptsBuilder ports.ListOptsBuilder
	listOptsBuilder = dns.PortListOptsExt{
		ListOptsBuilder: ports.ListOpts{},
		DNSName:         "test-port",
	}

	allPages, err := ports.List(fake.ServiceClient(), listOptsBuilder).AllPages()
	th.AssertNoErr(t, err)

	err = ports.ExtractPortsInto(allPages, &actual)
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, expected, actual)
}

func TestPortGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	PortHandleGet(t)

	var s PortDNS

	err := ports.Get(fake.ServiceClient(), "46d4bfb9-b26e-41f3-bd2e-e6dcc1ccedb2").ExtractInto(&s)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Status, "ACTIVE")
	th.AssertEquals(t, s.Name, "")
	th.AssertEquals(t, s.AdminStateUp, true)
	th.AssertEquals(t, s.NetworkID, "a87cc70a-3e15-4acf-8205-9b711a3531b7")
	th.AssertEquals(t, s.TenantID, "7e02058126cc4950b75f9970368ba177")
	th.AssertEquals(t, s.DeviceOwner, "network:router_interface")
	th.AssertEquals(t, s.MACAddress, "fa:16:3e:23:fd:d7")
	th.AssertDeepEquals(t, s.FixedIPs, []ports.IP{
		{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.1"},
	})
	th.AssertEquals(t, s.ID, "46d4bfb9-b26e-41f3-bd2e-e6dcc1ccedb2")
	th.AssertDeepEquals(t, s.SecurityGroups, []string{})
	th.AssertEquals(t, s.DeviceID, "5e3898d7-11be-483e-9732-b2f5eccd2b2e")

	th.AssertEquals(t, s.DNSName, "test-port")
	th.AssertDeepEquals(t, s.DNSAssignment, []map[string]string{
		{
			"hostname":   "test-port",
			"ip_address": "172.24.4.2",
			"fqdn":       "test-port.openstack.local.",
		},
	})
}

func TestPortCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	PortHandleCreate(t)

	var s PortDNS

	asu := true
	portCreateOpts := ports.CreateOpts{
		Name:         "private-port",
		AdminStateUp: &asu,
		NetworkID:    "a87cc70a-3e15-4acf-8205-9b711a3531b7",
		FixedIPs: []ports.IP{
			{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.2"},
		},
		SecurityGroups: &[]string{"foo"},
	}

	createOpts := dns.PortCreateOptsExt{
		CreateOptsBuilder: portCreateOpts,
		DNSName:           "test-port",
	}

	err := ports.Create(fake.ServiceClient(), createOpts).ExtractInto(&s)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Status, "DOWN")
	th.AssertEquals(t, s.Name, "private-port")
	th.AssertEquals(t, s.AdminStateUp, true)
	th.AssertEquals(t, s.NetworkID, "a87cc70a-3e15-4acf-8205-9b711a3531b7")
	th.AssertEquals(t, s.TenantID, "d6700c0c9ffa4f1cb322cd4a1f3906fa")
	th.AssertEquals(t, s.DeviceOwner, "")
	th.AssertEquals(t, s.MACAddress, "fa:16:3e:c9:cb:f0")
	th.AssertDeepEquals(t, s.FixedIPs, []ports.IP{
		{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.2"},
	})
	th.AssertEquals(t, s.ID, "65c0ee9f-d634-4522-8954-51021b570b0d")
	th.AssertDeepEquals(t, s.SecurityGroups, []string{"f0ac4394-7e4a-4409-9701-ba8be283dbc3"})

	th.AssertEquals(t, s.DNSName, "test-port")
	th.AssertDeepEquals(t, s.DNSAssignment, []map[string]string{
		{
			"hostname":   "test-port",
			"ip_address": "172.24.4.2",
			"fqdn":       "test-port.openstack.local.",
		},
	})
}

func TestPortRequiredCreateOpts(t *testing.T) {
	res := ports.Create(fake.ServiceClient(), dns.PortCreateOptsExt{CreateOptsBuilder: ports.CreateOpts{}})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
}

func TestPortUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	PortHandleUpdate(t)

	var s PortDNS

	name := "new_port_name"
	portUpdateOpts := ports.UpdateOpts{
		Name: &name,
		FixedIPs: []ports.IP{
			{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
		},
		SecurityGroups: &[]string{"f0ac4394-7e4a-4409-9701-ba8be283dbc3"},
	}

	dnsName := "test-port1"
	updateOpts := dns.PortUpdateOptsExt{
		UpdateOptsBuilder: portUpdateOpts,
		DNSName:           &dnsName,
	}

	err := ports.Update(fake.ServiceClient(), "65c0ee9f-d634-4522-8954-51021b570b0d", updateOpts).ExtractInto(&s)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "new_port_name")
	th.AssertDeepEquals(t, s.FixedIPs, []ports.IP{
		{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
	})
	th.AssertDeepEquals(t, s.SecurityGroups, []string{"f0ac4394-7e4a-4409-9701-ba8be283dbc3"})
	th.AssertEquals(t, s.DNSName, "test-port1")
	th.AssertDeepEquals(t, s.DNSAssignment, []map[string]string{
		{
			"hostname":   "test-port1",
			"ip_address": "172.24.4.2",
			"fqdn":       "test-port1.openstack.local.",
		},
	})
}

func TestFloatingIPGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	FloatingIPHandleGet(t)

	var actual FloatingIPDNS
	err := floatingips.Get(fake.ServiceClient(), "2f95fd2b-9f6a-4e8e-9e9a-2cbe286cbf9e").ExtractInto(&actual)
	th.AssertNoErr(t, err)

	expected := FloatingIPDNS{
		FloatingIP: floatingips.FloatingIP{
			FloatingNetworkID: "6d67c30a-ddb4-49a1-bec3-a65b286b4170",
			FixedIP:           "",
			FloatingIP:        "192.0.0.4",
			TenantID:          "017d8de156df4177889f31a9bd6edc00",
			Status:            "DOWN",
			PortID:            "",
			ID:                "2f95fd2b-9f6a-4e8e-9e9a-2cbe286cbf9e",
			RouterID:          "1117c30a-ddb4-49a1-bec3-a65b286b4170",
		},
		FloatingIPDNSExt: dns.FloatingIPDNSExt{
			DNSName:   "test-fip",
			DNSDomain: "local.",
		},
	}

	th.CheckDeepEquals(t, expected, actual)
}

func TestFloatingIPCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	FloatingIPHandleCreate(t)

	var actual FloatingIPDNS

	fipCreateOpts := floatingips.CreateOpts{
		FloatingNetworkID: "6d67c30a-ddb4-49a1-bec3-a65b286b4170",
	}

	options := dns.FloatingIPCreateOptsExt{
		CreateOptsBuilder: fipCreateOpts,
		DNSName:           "test-fip",
		DNSDomain:         "local.",
	}

	err := floatingips.Create(fake.ServiceClient(), options).ExtractInto(&actual)
	th.AssertNoErr(t, err)

	expected := FloatingIPDNS{
		FloatingIP: floatingips.FloatingIP{
			FloatingNetworkID: "6d67c30a-ddb4-49a1-bec3-a65b286b4170",
			FixedIP:           "",
			FloatingIP:        "192.0.0.4",
			TenantID:          "017d8de156df4177889f31a9bd6edc00",
			Status:            "DOWN",
			PortID:            "",
			ID:                "2f95fd2b-9f6a-4e8e-9e9a-2cbe286cbf9e",
			RouterID:          "1117c30a-ddb4-49a1-bec3-a65b286b4170",
		},
		FloatingIPDNSExt: dns.FloatingIPDNSExt{
			DNSName:   "test-fip",
			DNSDomain: "local.",
		},
	}

	th.CheckDeepEquals(t, expected, actual)
}

func TestNetworkGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	NetworkHandleGet(t)

	var actual NetworkDNS

	err := networks.Get(fake.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").ExtractInto(&actual)
	th.AssertNoErr(t, err)

	expected := NetworkDNS{
		Network: networks.Network{
			Name:         "public",
			Subnets:      []string{"54d6f61d-db07-451c-9ab3-b9609b6b6f0b"},
			Status:       "ACTIVE",
			TenantID:     "4fd44f30292945e481c7b8a0c8908869",
			AdminStateUp: true,
			Shared:       true,
			ID:           "d32019d3-bc6e-4319-9c1d-6722fc136a22",
		},
		NetworkDNSExt: dns.NetworkDNSExt{
			DNSDomain: "local.",
		},
	}

	th.CheckDeepEquals(t, expected, actual)
}

func TestNetworkCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	NetworkHandleCreate(t)

	var actual NetworkDNS

	iTrue := true
	networkCreateOpts := networks.CreateOpts{Name: "private", AdminStateUp: &iTrue}
	createOpts := dns.NetworkCreateOptsExt{
		CreateOptsBuilder: networkCreateOpts,
		DNSDomain:         "local.",
	}

	err := networks.Create(fake.ServiceClient(), createOpts).ExtractInto(&actual)
	th.AssertNoErr(t, err)

	expected := NetworkDNS{
		Network: networks.Network{
			Name:         "private",
			Subnets:      []string{"08eae331-0402-425a-923c-34f7cfe39c1b"},
			Status:       "ACTIVE",
			TenantID:     "26a7980765d0414dbc1fc1f88cdb7e6e",
			AdminStateUp: true,
			Shared:       false,
			ID:           "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
		},
		NetworkDNSExt: dns.NetworkDNSExt{
			DNSDomain: "local.",
		},
	}

	th.CheckDeepEquals(t, expected, actual)
}

func TestNetworkUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	NetworkHandleUpdate(t)

	var actual NetworkDNS

	name := "new_network_name"
	networkUpdateOpts := networks.UpdateOpts{Name: &name, AdminStateUp: new(bool)}
	updateOpts := dns.NetworkUpdateOptsExt{
		UpdateOptsBuilder: networkUpdateOpts,
		DNSDomain:         new(string),
	}

	err := networks.Update(fake.ServiceClient(), "db193ab3-96e3-4cb3-8fc5-05f4296d0324", updateOpts).ExtractInto(&actual)
	th.AssertNoErr(t, err)

	expected := NetworkDNS{
		Network: networks.Network{
			Name:         "new_network_name",
			Subnets:      []string{"08eae331-0402-425a-923c-34f7cfe39c1b"},
			Status:       "ACTIVE",
			TenantID:     "26a7980765d0414dbc1fc1f88cdb7e6e",
			AdminStateUp: false,
			Shared:       false,
			ID:           "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
		},
		NetworkDNSExt: dns.NetworkDNSExt{
			DNSDomain: "",
		},
	}

	th.CheckDeepEquals(t, expected, actual)
}
