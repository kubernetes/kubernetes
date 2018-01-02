package testing

import (
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/portsbinding"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleListSuccessfully(t)

	type PortWithExt struct {
		ports.Port
		portsbinding.PortsBindingExt
	}
	var actual []PortWithExt

	expected := []PortWithExt{
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
			PortsBindingExt: portsbinding.PortsBindingExt{
				VNICType: "normal",
				HostID:   "devstack",
			},
		},
	}

	allPages, err := ports.List(fake.ServiceClient(), ports.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)

	err = ports.ExtractPortsInto(allPages, &actual)
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, expected, actual)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGet(t)

	var s struct {
		ports.Port
		portsbinding.PortsBindingExt
	}

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

	th.AssertEquals(t, s.HostID, "devstack")
	th.AssertEquals(t, s.VNICType, "normal")
	th.AssertEquals(t, s.VIFType, "ovs")
	th.AssertDeepEquals(t, s.VIFDetails, map[string]interface{}{"port_filter": true, "ovs_hybrid_plug": true})
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCreate(t)

	var s struct {
		ports.Port
		portsbinding.PortsBindingExt
	}

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

	createOpts := portsbinding.CreateOptsExt{
		CreateOptsBuilder: portCreateOpts,
		HostID:            "HOST1",
		VNICType:          "normal",
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
	th.AssertEquals(t, s.HostID, "HOST1")
	th.AssertEquals(t, s.VNICType, "normal")
}

func TestRequiredCreateOpts(t *testing.T) {
	res := ports.Create(fake.ServiceClient(), portsbinding.CreateOptsExt{CreateOptsBuilder: ports.CreateOpts{}})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleUpdate(t)

	var s struct {
		ports.Port
		portsbinding.PortsBindingExt
	}

	portUpdateOpts := ports.UpdateOpts{
		Name: "new_port_name",
		FixedIPs: []ports.IP{
			{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
		},
		SecurityGroups: &[]string{"f0ac4394-7e4a-4409-9701-ba8be283dbc3"},
	}

	updateOpts := portsbinding.UpdateOptsExt{
		UpdateOptsBuilder: portUpdateOpts,
		HostID:            "HOST1",
		VNICType:          "normal",
	}

	err := ports.Update(fake.ServiceClient(), "65c0ee9f-d634-4522-8954-51021b570b0d", updateOpts).ExtractInto(&s)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "new_port_name")
	th.AssertDeepEquals(t, s.FixedIPs, []ports.IP{
		{SubnetID: "a0304c3a-4f08-4c43-88af-d796509c97d2", IPAddress: "10.0.0.3"},
	})
	th.AssertDeepEquals(t, s.SecurityGroups, []string{"f0ac4394-7e4a-4409-9701-ba8be283dbc3"})
	th.AssertEquals(t, s.HostID, "HOST1")
	th.AssertEquals(t, s.VNICType, "normal")
}
