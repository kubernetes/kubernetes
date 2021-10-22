package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/baremetal/v1/nodes"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListDetailNodes(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleNodeListDetailSuccessfully(t)

	pages := 0
	err := nodes.ListDetail(client.ServiceClient(), nodes.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := nodes.ExtractNodes(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 3 {
			t.Fatalf("Expected 3 nodes, got %d", len(actual))
		}
		th.CheckDeepEquals(t, NodeFoo, actual[0])
		th.CheckDeepEquals(t, NodeBar, actual[1])
		th.CheckDeepEquals(t, NodeBaz, actual[2])

		return true, nil
	})

	th.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestListNodes(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleNodeListSuccessfully(t)

	pages := 0
	err := nodes.List(client.ServiceClient(), nodes.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := nodes.ExtractNodes(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 3 {
			t.Fatalf("Expected 3 nodes, got %d", len(actual))
		}
		th.AssertEquals(t, "foo", actual[0].Name)
		th.AssertEquals(t, "bar", actual[1].Name)
		th.AssertEquals(t, "baz", actual[2].Name)

		return true, nil
	})

	th.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestListOpts(t *testing.T) {
	// Detail cannot take Fields
	opts := nodes.ListOpts{
		Fields: []string{"name", "uuid"},
	}

	_, err := opts.ToNodeListDetailQuery()
	th.AssertEquals(t, err.Error(), "fields is not a valid option when getting a detailed listing of nodes")

	// Regular ListOpts can
	query, err := opts.ToNodeListQuery()
	th.AssertEquals(t, query, "?fields=name&fields=uuid")
	th.AssertNoErr(t, err)
}

func TestCreateNode(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleNodeCreationSuccessfully(t, SingleNodeBody)

	actual, err := nodes.Create(client.ServiceClient(), nodes.CreateOpts{
		Name:          "foo",
		Driver:        "ipmi",
		BootInterface: "pxe",
		DriverInfo: map[string]interface{}{
			"ipmi_port":      "6230",
			"ipmi_username":  "admin",
			"deploy_kernel":  "http://172.22.0.1/images/tinyipa-stable-rocky.vmlinuz",
			"ipmi_address":   "192.168.122.1",
			"deploy_ramdisk": "http://172.22.0.1/images/tinyipa-stable-rocky.gz",
			"ipmi_password":  "admin",
		},
	}).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, NodeFoo, *actual)
}

func TestDeleteNode(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleNodeDeletionSuccessfully(t)

	res := nodes.Delete(client.ServiceClient(), "asdfasdfasdf")
	th.AssertNoErr(t, res.Err)
}

func TestGetNode(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleNodeGetSuccessfully(t)

	c := client.ServiceClient()
	actual, err := nodes.Get(c, "1234asdf").Extract()
	if err != nil {
		t.Fatalf("Unexpected Get error: %v", err)
	}

	th.CheckDeepEquals(t, NodeFoo, *actual)
}

func TestUpdateNode(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleNodeUpdateSuccessfully(t, SingleNodeBody)

	c := client.ServiceClient()
	actual, err := nodes.Update(c, "1234asdf", nodes.UpdateOpts{
		nodes.UpdateOperation{
			Op:   nodes.ReplaceOp,
			Path: "/properties",
			Value: map[string]interface{}{
				"root_gb": 25,
			},
		},
	}).Extract()
	if err != nil {
		t.Fatalf("Unexpected Update error: %v", err)
	}

	th.CheckDeepEquals(t, NodeFoo, *actual)
}

func TestUpdateRequiredOp(t *testing.T) {
	c := client.ServiceClient()
	_, err := nodes.Update(c, "1234asdf", nodes.UpdateOpts{
		nodes.UpdateOperation{
			Path:  "/driver",
			Value: "new-driver",
		},
	}).Extract()

	if _, ok := err.(gophercloud.ErrMissingInput); !ok {
		t.Fatal("ErrMissingInput was expected to occur")
	}

}

func TestUpdateRequiredPath(t *testing.T) {
	c := client.ServiceClient()
	_, err := nodes.Update(c, "1234asdf", nodes.UpdateOpts{
		nodes.UpdateOperation{
			Op:    nodes.ReplaceOp,
			Value: "new-driver",
		},
	}).Extract()

	if _, ok := err.(gophercloud.ErrMissingInput); !ok {
		t.Fatal("ErrMissingInput was expected to occur")
	}
}

func TestValidateNode(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleNodeValidateSuccessfully(t)

	c := client.ServiceClient()
	actual, err := nodes.Validate(c, "1234asdf").Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, NodeFooValidation, *actual)
}

func TestInjectNMI(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleInjectNMISuccessfully(t)

	c := client.ServiceClient()
	err := nodes.InjectNMI(c, "1234asdf").ExtractErr()
	th.AssertNoErr(t, err)
}

func TestSetBootDevice(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleSetBootDeviceSuccessfully(t)

	c := client.ServiceClient()
	err := nodes.SetBootDevice(c, "1234asdf", nodes.BootDeviceOpts{
		BootDevice: "pxe",
		Persistent: false,
	}).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestGetBootDevice(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetBootDeviceSuccessfully(t)

	c := client.ServiceClient()
	bootDevice, err := nodes.GetBootDevice(c, "1234asdf").Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, NodeBootDevice, *bootDevice)
}

func TestGetSupportedBootDevices(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSupportedBootDeviceSuccessfully(t)

	c := client.ServiceClient()
	bootDevices, err := nodes.GetSupportedBootDevices(c, "1234asdf").Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, NodeSupportedBootDevice, bootDevices)
}

func TestNodeChangeProvisionStateActive(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleNodeChangeProvisionStateActive(t)

	c := client.ServiceClient()
	err := nodes.ChangeProvisionState(c, "1234asdf", nodes.ProvisionStateOpts{
		Target:      nodes.TargetActive,
		ConfigDrive: "http://127.0.0.1/images/test-node-config-drive.iso.gz",
	}).ExtractErr()

	th.AssertNoErr(t, err)
}

func TestHandleNodeChangeProvisionStateConfigDrive(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleNodeChangeProvisionStateConfigDrive(t)

	c := client.ServiceClient()

	err := nodes.ChangeProvisionState(c, "1234asdf", nodes.ProvisionStateOpts{
		Target:      nodes.TargetActive,
		ConfigDrive: ConfigDriveMap,
	}).ExtractErr()

	th.AssertNoErr(t, err)
}

func TestNodeChangeProvisionStateClean(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleNodeChangeProvisionStateClean(t)

	c := client.ServiceClient()
	err := nodes.ChangeProvisionState(c, "1234asdf", nodes.ProvisionStateOpts{
		Target: nodes.TargetClean,
		CleanSteps: []nodes.CleanStep{
			{
				Interface: "deploy",
				Step:      "upgrade_firmware",
				Args: map[string]string{
					"force": "True",
				},
			},
		},
	}).ExtractErr()

	th.AssertNoErr(t, err)
}

func TestNodeChangeProvisionStateCleanWithConflict(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleNodeChangeProvisionStateCleanWithConflict(t)

	c := client.ServiceClient()
	err := nodes.ChangeProvisionState(c, "1234asdf", nodes.ProvisionStateOpts{
		Target: nodes.TargetClean,
		CleanSteps: []nodes.CleanStep{
			{
				Interface: "deploy",
				Step:      "upgrade_firmware",
				Args: map[string]string{
					"force": "True",
				},
			},
		},
	}).ExtractErr()

	if _, ok := err.(gophercloud.ErrDefault409); !ok {
		t.Fatal("ErrDefault409 was expected to occur")
	}
}

func TestCleanStepRequiresInterface(t *testing.T) {
	c := client.ServiceClient()
	err := nodes.ChangeProvisionState(c, "1234asdf", nodes.ProvisionStateOpts{
		Target: nodes.TargetClean,
		CleanSteps: []nodes.CleanStep{
			{
				Step: "upgrade_firmware",
				Args: map[string]string{
					"force": "True",
				},
			},
		},
	}).ExtractErr()

	if _, ok := err.(gophercloud.ErrMissingInput); !ok {
		t.Fatal("ErrMissingInput was expected to occur")
	}
}

func TestCleanStepRequiresStep(t *testing.T) {
	c := client.ServiceClient()
	err := nodes.ChangeProvisionState(c, "1234asdf", nodes.ProvisionStateOpts{
		Target: nodes.TargetClean,
		CleanSteps: []nodes.CleanStep{
			{
				Interface: "deploy",
				Args: map[string]string{
					"force": "True",
				},
			},
		},
	}).ExtractErr()

	if _, ok := err.(gophercloud.ErrMissingInput); !ok {
		t.Fatal("ErrMissingInput was expected to occur")
	}
}

func TestChangePowerState(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleChangePowerStateSuccessfully(t)

	opts := nodes.PowerStateOpts{
		Target:  nodes.PowerOn,
		Timeout: 100,
	}

	c := client.ServiceClient()
	err := nodes.ChangePowerState(c, "1234asdf", opts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestChangePowerStateWithConflict(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleChangePowerStateWithConflict(t)

	opts := nodes.PowerStateOpts{
		Target:  nodes.PowerOn,
		Timeout: 100,
	}

	c := client.ServiceClient()
	err := nodes.ChangePowerState(c, "1234asdf", opts).ExtractErr()
	if _, ok := err.(gophercloud.ErrDefault409); !ok {
		t.Fatal("ErrDefault409 was expected to occur")
	}
}

func TestSetRAIDConfig(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleSetRAIDConfig(t)

	sizeGB := 100
	isRootVolume := true

	config := nodes.RAIDConfigOpts{
		LogicalDisks: []nodes.LogicalDisk{
			{
				SizeGB:       &sizeGB,
				IsRootVolume: &isRootVolume,
				RAIDLevel:    nodes.RAID1,
			},
		},
	}

	c := client.ServiceClient()
	err := nodes.SetRAIDConfig(c, "1234asdf", config).ExtractErr()
	th.AssertNoErr(t, err)
}

// Without specifying a size, we need to send a string: "MAX"
func TestSetRAIDConfigMaxSize(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleSetRAIDConfigMaxSize(t)

	isRootVolume := true

	config := nodes.RAIDConfigOpts{
		LogicalDisks: []nodes.LogicalDisk{
			{
				IsRootVolume: &isRootVolume,
				RAIDLevel:    nodes.RAID1,
			},
		},
	}

	c := client.ServiceClient()
	err := nodes.SetRAIDConfig(c, "1234asdf", config).ExtractErr()
	th.AssertNoErr(t, err)
}
