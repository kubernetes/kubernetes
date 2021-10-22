package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/baremetal/v1/nodes"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// NodeListBody contains the canned body of a nodes.List response, without detail.
const NodeListBody = `
 {
  "nodes": [
    {
      "instance_uuid": null,
      "links": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/d2630783-6ec8-4836-b556-ab427c4b581e",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/d2630783-6ec8-4836-b556-ab427c4b581e",
          "rel": "bookmark"
        }
      ],
      "maintenance": false,
      "name": "foo",
      "power_state": null,
      "provision_state": "enroll"
    },
    {
      "instance_uuid": null,
      "links": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/08c84581-58f5-4ea2-a0c6-dd2e5d2b3662",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/08c84581-58f5-4ea2-a0c6-dd2e5d2b3662",
          "rel": "bookmark"
        }
      ],
      "maintenance": false,
      "name": "bar",
      "power_state": null,
      "provision_state": "enroll"
    },
    {
      "instance_uuid": null,
      "links": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/c9afd385-5d89-4ecb-9e1c-68194da6b474",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/c9afd385-5d89-4ecb-9e1c-68194da6b474",
          "rel": "bookmark"
        }
      ],
      "maintenance": false,
      "name": "baz",
      "power_state": null,
      "provision_state": "enroll"
    }
  ]
}
`

// NodeListDetailBody contains the canned body of a nodes.ListDetail response.
const NodeListDetailBody = `
 {
  "nodes": [
    {
      "bios_interface": "no-bios",
      "boot_interface": "pxe",
      "chassis_uuid": null,
      "clean_step": {},
      "conductor_group": "",
      "console_enabled": false,
      "console_interface": "no-console",
      "created_at": "2019-01-31T19:59:28+00:00",
      "deploy_interface": "iscsi",
      "deploy_step": {},
      "driver": "ipmi",
      "driver_info": {
        "ipmi_port": "6230",
        "ipmi_username": "admin",
        "deploy_kernel": "http://172.22.0.1/images/tinyipa-stable-rocky.vmlinuz",
        "ipmi_address": "192.168.122.1",
        "deploy_ramdisk": "http://172.22.0.1/images/tinyipa-stable-rocky.gz",
        "ipmi_password": "admin"

      },
      "driver_internal_info": {},
      "extra": {},
      "fault": null,
      "inspect_interface": "no-inspect",
      "inspection_finished_at": null,
      "inspection_started_at": null,
      "instance_info": {},
      "instance_uuid": null,
      "last_error": null,
      "links": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/d2630783-6ec8-4836-b556-ab427c4b581e",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/d2630783-6ec8-4836-b556-ab427c4b581e",
          "rel": "bookmark"
        }
      ],
      "maintenance": false,
      "maintenance_reason": null,
      "management_interface": "ipmitool",
      "name": "foo",
      "network_interface": "flat",
      "portgroups": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/portgroups",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/portgroups",
          "rel": "bookmark"
        }
      ],
      "ports": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/ports",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/ports",
          "rel": "bookmark"
        }
      ],
      "power_interface": "ipmitool",
      "power_state": null,
      "properties": {},
      "provision_state": "enroll",
      "provision_updated_at": null,
      "raid_config": {},
      "raid_interface": "no-raid",
      "rescue_interface": "no-rescue",
      "reservation": null,
      "resource_class": null,
      "states": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/states",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/states",
          "rel": "bookmark"
        }
      ],
      "storage_interface": "noop",
      "target_power_state": null,
      "target_provision_state": null,
      "target_raid_config": {},
      "traits": [],
      "updated_at": null,
      "uuid": "d2630783-6ec8-4836-b556-ab427c4b581e",
      "vendor_interface": "ipmitool",
      "volume": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/volume",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/volume",
          "rel": "bookmark"
        }
      ]
    },
    {
      "bios_interface": "no-bios",
      "boot_interface": "pxe",
      "chassis_uuid": null,
      "clean_step": {},
      "conductor_group": "",
      "console_enabled": false,
      "console_interface": "no-console",
      "created_at": "2019-01-31T19:59:29+00:00",
      "deploy_interface": "iscsi",
      "deploy_step": {},
      "driver": "ipmi",
      "driver_info": {},
      "driver_internal_info": {},
      "extra": {},
      "fault": null,
      "inspect_interface": "no-inspect",
      "inspection_finished_at": null,
      "inspection_started_at": null,
      "instance_info": {},
      "instance_uuid": null,
      "last_error": null,
      "links": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/08c84581-58f5-4ea2-a0c6-dd2e5d2b3662",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/08c84581-58f5-4ea2-a0c6-dd2e5d2b3662",
          "rel": "bookmark"
        }
      ],
      "maintenance": false,
      "maintenance_reason": null,
      "management_interface": "ipmitool",
      "name": "bar",
      "network_interface": "flat",
      "portgroups": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/08c84581-58f5-4ea2-a0c6-dd2e5d2b3662/portgroups",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/08c84581-58f5-4ea2-a0c6-dd2e5d2b3662/portgroups",
          "rel": "bookmark"
        }
      ],
      "ports": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/08c84581-58f5-4ea2-a0c6-dd2e5d2b3662/ports",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/08c84581-58f5-4ea2-a0c6-dd2e5d2b3662/ports",
          "rel": "bookmark"
        }
      ],
      "power_interface": "ipmitool",
      "power_state": null,
      "properties": {},
      "provision_state": "enroll",
      "provision_updated_at": null,
      "raid_config": {},
      "raid_interface": "no-raid",
      "rescue_interface": "no-rescue",
      "reservation": null,
      "resource_class": null,
      "states": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/08c84581-58f5-4ea2-a0c6-dd2e5d2b3662/states",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/08c84581-58f5-4ea2-a0c6-dd2e5d2b3662/states",
          "rel": "bookmark"
        }
      ],
      "storage_interface": "noop",
      "target_power_state": null,
      "target_provision_state": null,
      "target_raid_config": {},
      "traits": [],
      "updated_at": null,
      "uuid": "08c84581-58f5-4ea2-a0c6-dd2e5d2b3662",
      "vendor_interface": "ipmitool",
      "volume": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/08c84581-58f5-4ea2-a0c6-dd2e5d2b3662/volume",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/08c84581-58f5-4ea2-a0c6-dd2e5d2b3662/volume",
          "rel": "bookmark"
        }
      ]
    },
    {
      "bios_interface": "no-bios",
      "boot_interface": "pxe",
      "chassis_uuid": null,
      "clean_step": {},
      "conductor_group": "",
      "console_enabled": false,
      "console_interface": "no-console",
      "created_at": "2019-01-31T19:59:30+00:00",
      "deploy_interface": "iscsi",
      "deploy_step": {},
      "driver": "ipmi",
      "driver_info": {},
      "driver_internal_info": {},
      "extra": {},
      "fault": null,
      "inspect_interface": "no-inspect",
      "inspection_finished_at": null,
      "inspection_started_at": null,
      "instance_info": {},
      "instance_uuid": null,
      "last_error": null,
      "links": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/c9afd385-5d89-4ecb-9e1c-68194da6b474",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/c9afd385-5d89-4ecb-9e1c-68194da6b474",
          "rel": "bookmark"
        }
      ],
      "maintenance": false,
      "maintenance_reason": null,
      "management_interface": "ipmitool",
      "name": "baz",
      "network_interface": "flat",
      "portgroups": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/c9afd385-5d89-4ecb-9e1c-68194da6b474/portgroups",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/c9afd385-5d89-4ecb-9e1c-68194da6b474/portgroups",
          "rel": "bookmark"
        }
      ],
      "ports": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/c9afd385-5d89-4ecb-9e1c-68194da6b474/ports",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/c9afd385-5d89-4ecb-9e1c-68194da6b474/ports",
          "rel": "bookmark"
        }
      ],
      "power_interface": "ipmitool",
      "power_state": null,
      "properties": {},
      "provision_state": "enroll",
      "provision_updated_at": null,
      "raid_config": {},
      "raid_interface": "no-raid",
      "rescue_interface": "no-rescue",
      "reservation": null,
      "resource_class": null,
      "states": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/c9afd385-5d89-4ecb-9e1c-68194da6b474/states",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/c9afd385-5d89-4ecb-9e1c-68194da6b474/states",
          "rel": "bookmark"
        }
      ],
      "storage_interface": "noop",
      "target_power_state": null,
      "target_provision_state": null,
      "target_raid_config": {},
      "traits": [],
      "updated_at": null,
      "uuid": "c9afd385-5d89-4ecb-9e1c-68194da6b474",
      "vendor_interface": "ipmitool",
      "volume": [
        {
          "href": "http://ironic.example.com:6385/v1/nodes/c9afd385-5d89-4ecb-9e1c-68194da6b474/volume",
          "rel": "self"
        },
        {
          "href": "http://ironic.example.com:6385/nodes/c9afd385-5d89-4ecb-9e1c-68194da6b474/volume",
          "rel": "bookmark"
        }
      ]
    }
  ]
}
`

// SingleNodeBody is the canned body of a Get request on an existing node.
const SingleNodeBody = `
{
  "bios_interface": "no-bios",
  "boot_interface": "pxe",
  "chassis_uuid": null,
  "clean_step": {},
  "conductor_group": "",
  "console_enabled": false,
  "console_interface": "no-console",
  "created_at": "2019-01-31T19:59:28+00:00",
  "deploy_interface": "iscsi",
  "deploy_step": {},
  "driver": "ipmi",
  "driver_info": {
    "ipmi_port": "6230",
    "ipmi_username": "admin",
    "deploy_kernel": "http://172.22.0.1/images/tinyipa-stable-rocky.vmlinuz",
    "ipmi_address": "192.168.122.1",
    "deploy_ramdisk": "http://172.22.0.1/images/tinyipa-stable-rocky.gz",
    "ipmi_password": "admin"
  },
  "driver_internal_info": {},
  "extra": {},
  "fault": null,
  "inspect_interface": "no-inspect",
  "inspection_finished_at": null,
  "inspection_started_at": null,
  "instance_info": {},
  "instance_uuid": null,
  "last_error": null,
  "links": [
    {
      "href": "http://ironic.example.com:6385/v1/nodes/d2630783-6ec8-4836-b556-ab427c4b581e",
      "rel": "self"
    },
    {
      "href": "http://ironic.example.com:6385/nodes/d2630783-6ec8-4836-b556-ab427c4b581e",
      "rel": "bookmark"
    }
  ],
  "maintenance": false,
  "maintenance_reason": null,
  "management_interface": "ipmitool",
  "name": "foo",
  "network_interface": "flat",
  "portgroups": [
    {
      "href": "http://ironic.example.com:6385/v1/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/portgroups",
      "rel": "self"
    },
    {
      "href": "http://ironic.example.com:6385/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/portgroups",
      "rel": "bookmark"
    }
  ],
  "ports": [
    {
      "href": "http://ironic.example.com:6385/v1/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/ports",
      "rel": "self"
    },
    {
      "href": "http://ironic.example.com:6385/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/ports",
      "rel": "bookmark"
    }
  ],
  "power_interface": "ipmitool",
  "power_state": null,
  "properties": {},
  "provision_state": "enroll",
  "provision_updated_at": null,
  "raid_config": {},
  "raid_interface": "no-raid",
  "rescue_interface": "no-rescue",
  "reservation": null,
  "resource_class": null,
  "states": [
    {
      "href": "http://ironic.example.com:6385/v1/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/states",
      "rel": "self"
    },
    {
      "href": "http://ironic.example.com:6385/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/states",
      "rel": "bookmark"
    }
  ],
  "storage_interface": "noop",
  "target_power_state": null,
  "target_provision_state": null,
  "target_raid_config": {},
  "traits": [],
  "updated_at": null,
  "uuid": "d2630783-6ec8-4836-b556-ab427c4b581e",
  "vendor_interface": "ipmitool",
  "volume": [
    {
      "href": "http://ironic.example.com:6385/v1/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/volume",
      "rel": "self"
    },
    {
      "href": "http://ironic.example.com:6385/nodes/d2630783-6ec8-4836-b556-ab427c4b581e/volume",
      "rel": "bookmark"
    }
  ]
}
`

const NodeValidationBody = `
{
  "bios": {
    "reason": "Driver ipmi does not support bios (disabled or not implemented).",
    "result": false
  },
  "boot": {
    "reason": "Cannot validate image information for node a62b8495-52e2-407b-b3cb-62775d04c2b8 because one or more parameters are missing from its instance_info and insufficent information is present to boot from a remote volume. Missing are: ['ramdisk', 'kernel', 'image_source']",
    "result": false
  },
  "console": {
    "reason": "Driver ipmi does not support console (disabled or not implemented).",
    "result": false
  },
  "deploy": {
    "reason": "Cannot validate image information for node a62b8495-52e2-407b-b3cb-62775d04c2b8 because one or more parameters are missing from its instance_info and insufficent information is present to boot from a remote volume. Missing are: ['ramdisk', 'kernel', 'image_source']",
    "result": false
  },
  "inspect": {
    "reason": "Driver ipmi does not support inspect (disabled or not implemented).",
    "result": false
  },
  "management": {
    "result": true
  },
  "network": {
    "result": true
  },
  "power": {
    "result": true
  },
  "raid": {
    "reason": "Driver ipmi does not support raid (disabled or not implemented).",
    "result": false
  },
  "rescue": {
    "reason": "Driver ipmi does not support rescue (disabled or not implemented).",
    "result": false
  },
  "storage": {
    "result": true
  }
}
`

const NodeBootDeviceBody = `
{
  "boot_device":"pxe",
  "persistent":false
}
`

const NodeSupportedBootDeviceBody = `
{
  "supported_boot_devices": [
    "pxe",
    "disk"
  ]
}
`

const NodeProvisionStateActiveBody = `
{
    "target": "active",
    "configdrive": "http://127.0.0.1/images/test-node-config-drive.iso.gz"
}
`
const NodeProvisionStateCleanBody = `
{
    "target": "clean",
    "clean_steps": [
        {
            "interface": "deploy",
            "step": "upgrade_firmware",
            "args": {
                "force": "True"
            }
        }
    ]
}
`

const NodeProvisionStateConfigDriveBody = `
{
	"target": "active",
	"configdrive": {
		"user_data": {
			"ignition": {
			"version": "2.2.0"
			},
			"systemd": {
				"units": [
					{
						"enabled": true,
						"name": "example.service"
					}
				]
			}
		}
	}
}
`

var (
	NodeFoo = nodes.Node{
		UUID:                 "d2630783-6ec8-4836-b556-ab427c4b581e",
		Name:                 "foo",
		PowerState:           "",
		TargetPowerState:     "",
		ProvisionState:       "enroll",
		TargetProvisionState: "",
		Maintenance:          false,
		MaintenanceReason:    "",
		Fault:                "",
		LastError:            "",
		Reservation:          "",
		Driver:               "ipmi",
		DriverInfo: map[string]interface{}{
			"ipmi_port":      "6230",
			"ipmi_username":  "admin",
			"deploy_kernel":  "http://172.22.0.1/images/tinyipa-stable-rocky.vmlinuz",
			"ipmi_address":   "192.168.122.1",
			"deploy_ramdisk": "http://172.22.0.1/images/tinyipa-stable-rocky.gz",
			"ipmi_password":  "admin",
		},
		DriverInternalInfo:  map[string]interface{}{},
		Properties:          map[string]interface{}{},
		InstanceInfo:        map[string]interface{}{},
		InstanceUUID:        "",
		ChassisUUID:         "",
		Extra:               map[string]interface{}{},
		ConsoleEnabled:      false,
		RAIDConfig:          map[string]interface{}{},
		TargetRAIDConfig:    map[string]interface{}{},
		CleanStep:           map[string]interface{}{},
		DeployStep:          map[string]interface{}{},
		ResourceClass:       "",
		BootInterface:       "pxe",
		ConsoleInterface:    "no-console",
		DeployInterface:     "iscsi",
		InspectInterface:    "no-inspect",
		ManagementInterface: "ipmitool",
		NetworkInterface:    "flat",
		PowerInterface:      "ipmitool",
		RAIDInterface:       "no-raid",
		RescueInterface:     "no-rescue",
		StorageInterface:    "noop",
		Traits:              []string{},
		VendorInterface:     "ipmitool",
		ConductorGroup:      "",
		Protected:           false,
		ProtectedReason:     "",
	}

	NodeFooValidation = nodes.NodeValidation{
		Boot: nodes.DriverValidation{
			Result: false,
			Reason: "Cannot validate image information for node a62b8495-52e2-407b-b3cb-62775d04c2b8 because one or more parameters are missing from its instance_info and insufficent information is present to boot from a remote volume. Missing are: ['ramdisk', 'kernel', 'image_source']",
		},
		Console: nodes.DriverValidation{
			Result: false,
			Reason: "Driver ipmi does not support console (disabled or not implemented).",
		},
		Deploy: nodes.DriverValidation{
			Result: false,
			Reason: "Cannot validate image information for node a62b8495-52e2-407b-b3cb-62775d04c2b8 because one or more parameters are missing from its instance_info and insufficent information is present to boot from a remote volume. Missing are: ['ramdisk', 'kernel', 'image_source']",
		},
		Inspect: nodes.DriverValidation{
			Result: false,
			Reason: "Driver ipmi does not support inspect (disabled or not implemented).",
		},
		Management: nodes.DriverValidation{
			Result: true,
		},
		Network: nodes.DriverValidation{
			Result: true,
		},
		Power: nodes.DriverValidation{
			Result: true,
		},
		RAID: nodes.DriverValidation{
			Result: false,
			Reason: "Driver ipmi does not support raid (disabled or not implemented).",
		},
		Rescue: nodes.DriverValidation{
			Result: false,
			Reason: "Driver ipmi does not support rescue (disabled or not implemented).",
		},
		Storage: nodes.DriverValidation{
			Result: true,
		},
	}

	NodeBootDevice = nodes.BootDeviceOpts{
		BootDevice: "pxe",
		Persistent: false,
	}

	NodeSupportedBootDevice = []string{
		"pxe",
		"disk",
	}

	NodeBar = nodes.Node{
		UUID:                 "08c84581-58f5-4ea2-a0c6-dd2e5d2b3662",
		Name:                 "bar",
		PowerState:           "",
		TargetPowerState:     "",
		ProvisionState:       "enroll",
		TargetProvisionState: "",
		Maintenance:          false,
		MaintenanceReason:    "",
		Fault:                "",
		LastError:            "",
		Reservation:          "",
		Driver:               "ipmi",
		DriverInfo:           map[string]interface{}{},
		DriverInternalInfo:   map[string]interface{}{},
		Properties:           map[string]interface{}{},
		InstanceInfo:         map[string]interface{}{},
		InstanceUUID:         "",
		ChassisUUID:          "",
		Extra:                map[string]interface{}{},
		ConsoleEnabled:       false,
		RAIDConfig:           map[string]interface{}{},
		TargetRAIDConfig:     map[string]interface{}{},
		CleanStep:            map[string]interface{}{},
		DeployStep:           map[string]interface{}{},
		ResourceClass:        "",
		BootInterface:        "pxe",
		ConsoleInterface:     "no-console",
		DeployInterface:      "iscsi",
		InspectInterface:     "no-inspect",
		ManagementInterface:  "ipmitool",
		NetworkInterface:     "flat",
		PowerInterface:       "ipmitool",
		RAIDInterface:        "no-raid",
		RescueInterface:      "no-rescue",
		StorageInterface:     "noop",
		Traits:               []string{},
		VendorInterface:      "ipmitool",
		ConductorGroup:       "",
		Protected:            false,
		ProtectedReason:      "",
	}

	NodeBaz = nodes.Node{
		UUID:                 "c9afd385-5d89-4ecb-9e1c-68194da6b474",
		Name:                 "baz",
		PowerState:           "",
		TargetPowerState:     "",
		ProvisionState:       "enroll",
		TargetProvisionState: "",
		Maintenance:          false,
		MaintenanceReason:    "",
		Fault:                "",
		LastError:            "",
		Reservation:          "",
		Driver:               "ipmi",
		DriverInfo:           map[string]interface{}{},
		DriverInternalInfo:   map[string]interface{}{},
		Properties:           map[string]interface{}{},
		InstanceInfo:         map[string]interface{}{},
		InstanceUUID:         "",
		ChassisUUID:          "",
		Extra:                map[string]interface{}{},
		ConsoleEnabled:       false,
		RAIDConfig:           map[string]interface{}{},
		TargetRAIDConfig:     map[string]interface{}{},
		CleanStep:            map[string]interface{}{},
		DeployStep:           map[string]interface{}{},
		ResourceClass:        "",
		BootInterface:        "pxe",
		ConsoleInterface:     "no-console",
		DeployInterface:      "iscsi",
		InspectInterface:     "no-inspect",
		ManagementInterface:  "ipmitool",
		NetworkInterface:     "flat",
		PowerInterface:       "ipmitool",
		RAIDInterface:        "no-raid",
		RescueInterface:      "no-rescue",
		StorageInterface:     "noop",
		Traits:               []string{},
		VendorInterface:      "ipmitool",
		ConductorGroup:       "",
		Protected:            false,
		ProtectedReason:      "",
	}

	ConfigDriveMap = nodes.ConfigDrive{
		UserData: map[string]interface{}{
			"ignition": map[string]string{
				"version": "2.2.0",
			},
			"systemd": map[string]interface{}{
				"units": []map[string]interface{}{{
					"name":    "example.service",
					"enabled": true,
				},
				},
			},
		},
	}
)

// HandleNodeListSuccessfully sets up the test server to respond to a server List request.
func HandleNodeListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/nodes", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.Header().Add("Content-Type", "application/json")
		r.ParseForm()

		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, NodeListBody)

		case "9e5476bd-a4ec-4653-93d6-72c93aa682ba":
			fmt.Fprintf(w, `{ "servers": [] }`)
		default:
			t.Fatalf("/nodes invoked with unexpected marker=[%s]", marker)
		}
	})
}

// HandleNodeListSuccessfully sets up the test server to respond to a server List request.
func HandleNodeListDetailSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/nodes/detail", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.Header().Add("Content-Type", "application/json")
		r.ParseForm()

		fmt.Fprintf(w, NodeListDetailBody)
	})
}

// HandleServerCreationSuccessfully sets up the test server to respond to a server creation request
// with a given response.
func HandleNodeCreationSuccessfully(t *testing.T, response string) {
	th.Mux.HandleFunc("/nodes", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{
          "boot_interface": "pxe",
          "driver": "ipmi",
          "driver_info": {
            "deploy_kernel": "http://172.22.0.1/images/tinyipa-stable-rocky.vmlinuz",
            "deploy_ramdisk": "http://172.22.0.1/images/tinyipa-stable-rocky.gz",
            "ipmi_address": "192.168.122.1",
            "ipmi_password": "admin",
            "ipmi_port": "6230",
            "ipmi_username": "admin"
          },
          "name": "foo"
        }`)

		w.WriteHeader(http.StatusAccepted)
		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, response)
	})
}

// HandleNodeDeletionSuccessfully sets up the test server to respond to a server deletion request.
func HandleNodeDeletionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/nodes/asdfasdfasdf", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

func HandleNodeGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/nodes/1234asdf", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, SingleNodeBody)
	})
}

func HandleNodeUpdateSuccessfully(t *testing.T, response string) {
	th.Mux.HandleFunc("/nodes/1234asdf", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestJSONRequest(t, r, `[{"op": "replace", "path": "/properties", "value": {"root_gb": 25}}]`)

		fmt.Fprintf(w, response)
	})
}

func HandleNodeValidateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/nodes/1234asdf/validate", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, NodeValidationBody)
	})
}

// HandleInjectNMISuccessfully sets up the test server to respond to a node InjectNMI request
func HandleInjectNMISuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/nodes/1234asdf/management/inject_nmi", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, "{}")

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleSetBootDeviceSuccessfully sets up the test server to respond to a set boot device request for a node
func HandleSetBootDeviceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/nodes/1234asdf/management/boot_device", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, NodeBootDeviceBody)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleGetBootDeviceSuccessfully sets up the test server to respond to a get boot device request for a node
func HandleGetBootDeviceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/nodes/1234asdf/management/boot_device", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, NodeBootDeviceBody)
	})
}

// HandleGetBootDeviceSuccessfully sets up the test server to respond to a get boot device request for a node
func HandleGetSupportedBootDeviceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/nodes/1234asdf/management/boot_device/supported", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, NodeSupportedBootDeviceBody)
	})
}

func HandleNodeChangeProvisionStateActive(t *testing.T) {
	th.Mux.HandleFunc("/nodes/1234asdf/states/provision", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, NodeProvisionStateActiveBody)
		w.WriteHeader(http.StatusAccepted)
	})
}

func HandleNodeChangeProvisionStateClean(t *testing.T) {
	th.Mux.HandleFunc("/nodes/1234asdf/states/provision", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, NodeProvisionStateCleanBody)
		w.WriteHeader(http.StatusAccepted)
	})
}

func HandleNodeChangeProvisionStateCleanWithConflict(t *testing.T) {
	th.Mux.HandleFunc("/nodes/1234asdf/states/provision", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, NodeProvisionStateCleanBody)
		w.WriteHeader(http.StatusConflict)
	})
}

func HandleNodeChangeProvisionStateConfigDrive(t *testing.T) {
	th.Mux.HandleFunc("/nodes/1234asdf/states/provision", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, NodeProvisionStateConfigDriveBody)
		w.WriteHeader(http.StatusAccepted)
	})
}

// HandleChangePowerStateSuccessfully sets up the test server to respond to a change power state request for a node
func HandleChangePowerStateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/nodes/1234asdf/states/power", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{
			"target": "power on",
			"timeout": 100
		}`)

		w.WriteHeader(http.StatusAccepted)
	})
}

// HandleChangePowerStateWithConflict sets up the test server to respond to a change power state request for a node with a 409 error
func HandleChangePowerStateWithConflict(t *testing.T) {
	th.Mux.HandleFunc("/nodes/1234asdf/states/power", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{
			"target": "power on",
			"timeout": 100
		}`)

		w.WriteHeader(http.StatusConflict)
	})
}

func HandleSetRAIDConfig(t *testing.T) {
	th.Mux.HandleFunc("/nodes/1234asdf/states/raid", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `
			{
			   "logical_disks" : [
				  {
					 "size_gb" : 100,
					 "is_root_volume" : true,
					 "raid_level" : "1"
				  }
			   ]
			}
		`)

		w.WriteHeader(http.StatusNoContent)
	})
}

func HandleSetRAIDConfigMaxSize(t *testing.T) {
	th.Mux.HandleFunc("/nodes/1234asdf/states/raid", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `
			{
			   "logical_disks" : [
				  {
					 "size_gb" : "MAX",
					 "is_root_volume" : true,
					 "raid_level" : "1"
				  }
			   ]
			}
		`)

		w.WriteHeader(http.StatusNoContent)
	})
}
