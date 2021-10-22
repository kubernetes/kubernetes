package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/baremetal/v1/drivers"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ListDriversBody contains the canned body of a drivers.ListDrivers response, without details.
const ListDriversBody = `
{
  "drivers": [
    {
      "hosts": [
        "897ab1dad809"
      ],
      "links": [
        {
          "href": "http://127.0.0.1:6385/v1/drivers/agent_ipmitool",
          "rel": "self"
        },
        {
          "href": "http://127.0.0.1:6385/drivers/agent_ipmitool",
          "rel": "bookmark"
        }
      ],
      "name": "agent_ipmitool",
      "properties": [
        {
          "href": "http://127.0.0.1:6385/v1/drivers/agent_ipmitool/properties",
          "rel": "self"
        },
        {
          "href": "http://127.0.0.1:6385/drivers/agent_ipmitool/properties",
          "rel": "bookmark"
        }
      ],
      "type": "classic"
    },
    {
      "hosts": [
        "897ab1dad809"
      ],
      "links": [
        {
          "href": "http://127.0.0.1:6385/v1/drivers/fake",
          "rel": "self"
        },
        {
          "href": "http://127.0.0.1:6385/drivers/fake",
          "rel": "bookmark"
        }
      ],
      "name": "fake",
      "properties": [
        {
          "href": "http://127.0.0.1:6385/v1/drivers/fake/properties",
          "rel": "self"
        },
        {
          "href": "http://127.0.0.1:6385/drivers/fake/properties",
          "rel": "bookmark"
        }
      ],
      "type": "classic"
    },
    {
      "hosts": [
        "897ab1dad809"
      ],
      "links": [
        {
          "href": "http://127.0.0.1:6385/v1/drivers/ipmi",
          "rel": "self"
        },
        {
          "href": "http://127.0.0.1:6385/drivers/ipmi",
          "rel": "bookmark"
        }
      ],
      "name": "ipmi",
      "properties": [
        {
          "href": "http://127.0.0.1:6385/v1/drivers/ipmi/properties",
          "rel": "self"
        },
        {
          "href": "http://127.0.0.1:6385/drivers/ipmi/properties",
          "rel": "bookmark"
        }
      ],
      "type": "dynamic"
    }
  ]
}
`
const SingleDriverDetails = `
{
  "default_bios_interface": "no-bios",
  "default_boot_interface": "pxe",
  "default_console_interface": "no-console",
  "default_deploy_interface": "iscsi",
  "default_inspect_interface": "no-inspect",
  "default_management_interface": "ipmitool",
  "default_network_interface": "flat",
  "default_power_interface": "ipmitool",
  "default_raid_interface": "no-raid",
  "default_rescue_interface": "no-rescue",
  "default_storage_interface": "noop",
  "default_vendor_interface": "no-vendor",
  "enabled_bios_interfaces": [
    "no-bios"
  ],
  "enabled_boot_interfaces": [
    "pxe"
  ],
  "enabled_console_interfaces": [
    "no-console"
  ],
  "enabled_deploy_interfaces": [
    "iscsi",
    "direct"
  ],
  "enabled_inspect_interfaces": [
    "no-inspect"
  ],
  "enabled_management_interfaces": [
    "ipmitool"
  ],
  "enabled_network_interfaces": [
    "flat",
    "noop"
  ],
  "enabled_power_interfaces": [
    "ipmitool"
  ],
  "enabled_raid_interfaces": [
    "no-raid",
    "agent"
  ],
  "enabled_rescue_interfaces": [
    "no-rescue"
  ],
  "enabled_storage_interfaces": [
    "noop"
  ],
  "enabled_vendor_interfaces": [
    "no-vendor"
  ],
  "hosts": [
    "897ab1dad809"
  ],
  "links": [
    {
      "href": "http://127.0.0.1:6385/v1/drivers/ipmi",
      "rel": "self"
    },
    {
      "href": "http://127.0.0.1:6385/drivers/ipmi",
      "rel": "bookmark"
    }
  ],
  "name": "ipmi",
  "properties": [
    {
      "href": "http://127.0.0.1:6385/v1/drivers/ipmi/properties",
      "rel": "self"
    },
    {
      "href": "http://127.0.0.1:6385/drivers/ipmi/properties",
      "rel": "bookmark"
    }
  ],
  "type": "dynamic"
}
`

const SingleDriverProperties = `
{
  "deploy_forces_oob_reboot": "Whether Ironic should force a reboot of the Node via the out-of-band channel after deployment is complete. Provides compatibility with older deploy ramdisks. Defaults to False. Optional.",
  "deploy_kernel": "UUID (from Glance) of the deployment kernel. Required.",
  "deploy_ramdisk": "UUID (from Glance) of the ramdisk that is mounted at boot time. Required.",
  "image_http_proxy": "URL of a proxy server for HTTP connections. Optional.",
  "image_https_proxy": "URL of a proxy server for HTTPS connections. Optional.",
  "image_no_proxy": "A comma-separated list of host names, IP addresses and domain names (with optional :port) that will be excluded from proxying. To denote a domain name, use a dot to prefix the domain name. This value will be ignored if ` + "``image_http_proxy`` and ``image_https_proxy``" + ` are not specified. Optional.",
  "ipmi_address": "IP address or hostname of the node. Required.",
  "ipmi_bridging": "bridging_type; default is \"no\". One of \"single\", \"dual\", \"no\". Optional.",
  "ipmi_disable_boot_timeout": "By default ironic will send a raw IPMI command to disable the 60 second timeout for booting.  Setting this option to False will NOT send that command; default value is True. Optional.",
  "ipmi_force_boot_device": "Whether Ironic should specify the boot device to the BMC each time the server is turned on, eg. because the BMC is not capable of remembering the selected boot device across power cycles; default value is False. Optional.",
  "ipmi_local_address": "local IPMB address for bridged requests. Used only if ipmi_bridging is set to \"single\" or \"dual\". Optional.",
  "ipmi_password": "password. Optional.",
  "ipmi_port": "remote IPMI RMCP port. Optional.",
  "ipmi_priv_level": "privilege level; default is ADMINISTRATOR. One of ADMINISTRATOR, CALLBACK, OPERATOR, USER. Optional.",
  "ipmi_protocol_version": "the version of the IPMI protocol; default is \"2.0\". One of \"1.5\", \"2.0\". Optional.",
  "ipmi_target_address": "destination address for bridged request. Required only if ipmi_bridging is set to \"single\" or \"dual\".",
  "ipmi_target_channel": "destination channel for bridged request. Required only if ipmi_bridging is set to \"single\" or \"dual\".",
  "ipmi_terminal_port": "node's UDP port to connect to. Only required for console access.",
  "ipmi_transit_address": "transit address for bridged request. Required only if ipmi_bridging is set to \"dual\".",
  "ipmi_transit_channel": "transit channel for bridged request. Required only if ipmi_bridging is set to \"dual\".",
  "ipmi_username": "username; default is NULL user. Optional."
}
`

const SingleDriverDiskProperties = `
{
  "controller": "Controller to use for this logical disk. If not specified, the driver will choose a suitable RAID controller on the bare metal node. Optional.",
  "disk_type": "The type of disk preferred. Valid values are 'hdd' and 'ssd'. If this is not specified, disk type will not be a selection criterion for choosing backing physical disks. Optional.",
  "interface_type": "The interface type of disk. Valid values are 'sata', 'scsi' and 'sas'. If this is not specified, interface type will not be a selection criterion for choosing backing physical disks. Optional.",
  "is_root_volume": "Specifies whether this disk is a root volume. By default, this is False. Optional.",
  "number_of_physical_disks": "Number of physical disks to use for this logical disk. By default, the driver uses the minimum number of disks required for that RAID level. Optional.",
  "physical_disks": "The physical disks to use for this logical disk. If not specified, the driver will choose suitable physical disks to use. Optional.",
  "raid_level": "RAID level for the logical disk. Valid values are 'JBOD', '0', '1', '2', '5', '6', '1+0', '5+0' and '6+0'. Required.",
  "share_physical_disks": "Specifies whether other logical disks can share physical disks with this logical disk. By default, this is False. Optional.",
  "size_gb": "Size in GiB (Integer) for the logical disk. Use 'MAX' as size_gb if this logical disk is supposed to use the rest of the space available. Required.",
  "volume_name": "Name of the volume to be created. If this is not specified, it will be auto-generated. Optional."
}
`

var (
	DriverAgentIpmitool = drivers.Driver{
		Name:  "agent_ipmitool",
		Type:  "classic",
		Hosts: []string{"897ab1dad809"},
		Links: []interface{}{
			map[string]interface{}{
				"href": "http://127.0.0.1:6385/v1/drivers/agent_ipmitool",
				"rel":  "self",
			},
			map[string]interface{}{
				"href": "http://127.0.0.1:6385/drivers/agent_ipmitool",
				"rel":  "bookmark",
			},
		},
		Properties: []interface{}{
			map[string]interface{}{
				"href": "http://127.0.0.1:6385/v1/drivers/agent_ipmitool/properties",
				"rel":  "self",
			},
			map[string]interface{}{
				"href": "http://127.0.0.1:6385/drivers/agent_ipmitool/properties",
				"rel":  "bookmark",
			},
		},
	}

	DriverFake = drivers.Driver{
		Name:  "fake",
		Type:  "classic",
		Hosts: []string{"897ab1dad809"},
		Links: []interface{}{
			map[string]interface{}{
				"href": "http://127.0.0.1:6385/v1/drivers/fake",
				"rel":  "self",
			},
			map[string]interface{}{
				"href": "http://127.0.0.1:6385/drivers/fake",
				"rel":  "bookmark",
			},
		},
		Properties: []interface{}{
			map[string]interface{}{
				"href": "http://127.0.0.1:6385/v1/drivers/fake/properties",
				"rel":  "self",
			},
			map[string]interface{}{
				"href": "http://127.0.0.1:6385/drivers/fake/properties",
				"rel":  "bookmark",
			},
		},
	}

	DriverIpmi = drivers.Driver{
		Name:                        "ipmi",
		Type:                        "dynamic",
		Hosts:                       []string{"897ab1dad809"},
		DefaultBiosInterface:        "no-bios",
		DefaultBootInterface:        "pxe",
		DefaultConsoleInterface:     "no-console",
		DefaultDeployInterface:      "iscsi",
		DefaultInspectInterface:     "no-inspect",
		DefaultManagementInterface:  "ipmitool",
		DefaultNetworkInterface:     "flat",
		DefaultPowerInterface:       "ipmitool",
		DefaultRaidInterface:        "no-raid",
		DefaultRescueInterface:      "no-rescue",
		DefaultStorageInterface:     "noop",
		DefaultVendorInterface:      "no-vendor",
		EnabledBiosInterfaces:       []string{"no-bios"},
		EnabledBootInterfaces:       []string{"pxe"},
		EnabledConsoleInterface:     []string{"no-console"},
		EnabledDeployInterfaces:     []string{"iscsi", "direct"},
		EnabledInspectInterfaces:    []string{"no-inspect"},
		EnabledManagementInterfaces: []string{"ipmitool"},
		EnabledNetworkInterfaces:    []string{"flat", "noop"},
		EnabledPowerInterfaces:      []string{"ipmitool"},
		EnabledRescueInterfaces:     []string{"no-rescue"},
		EnabledRaidInterfaces:       []string{"no-raid", "agent"},
		EnabledStorageInterfaces:    []string{"noop"},
		EnabledVendorInterfaces:     []string{"no-vendor"},
		Links: []interface{}{
			map[string]interface{}{
				"href": "http://127.0.0.1:6385/v1/drivers/ipmi",
				"rel":  "self",
			},
			map[string]interface{}{
				"href": "http://127.0.0.1:6385/drivers/ipmi",
				"rel":  "bookmark",
			},
		},
		Properties: []interface{}{
			map[string]interface{}{
				"href": "http://127.0.0.1:6385/v1/drivers/ipmi/properties",
				"rel":  "self",
			},
			map[string]interface{}{
				"href": "http://127.0.0.1:6385/drivers/ipmi/properties",
				"rel":  "bookmark",
			},
		},
	}

	DriverIpmiToolProperties = drivers.DriverProperties{
		"deploy_forces_oob_reboot":  "Whether Ironic should force a reboot of the Node via the out-of-band channel after deployment is complete. Provides compatibility with older deploy ramdisks. Defaults to False. Optional.",
		"deploy_kernel":             "UUID (from Glance) of the deployment kernel. Required.",
		"deploy_ramdisk":            "UUID (from Glance) of the ramdisk that is mounted at boot time. Required.",
		"image_http_proxy":          "URL of a proxy server for HTTP connections. Optional.",
		"image_https_proxy":         "URL of a proxy server for HTTPS connections. Optional.",
		"image_no_proxy":            "A comma-separated list of host names, IP addresses and domain names (with optional :port) that will be excluded from proxying. To denote a domain name, use a dot to prefix the domain name. This value will be ignored if ``image_http_proxy`` and ``image_https_proxy`` are not specified. Optional.",
		"ipmi_address":              "IP address or hostname of the node. Required.",
		"ipmi_bridging":             "bridging_type; default is \"no\". One of \"single\", \"dual\", \"no\". Optional.",
		"ipmi_disable_boot_timeout": "By default ironic will send a raw IPMI command to disable the 60 second timeout for booting.  Setting this option to False will NOT send that command; default value is True. Optional.",
		"ipmi_force_boot_device":    "Whether Ironic should specify the boot device to the BMC each time the server is turned on, eg. because the BMC is not capable of remembering the selected boot device across power cycles; default value is False. Optional.",
		"ipmi_local_address":        "local IPMB address for bridged requests. Used only if ipmi_bridging is set to \"single\" or \"dual\". Optional.",
		"ipmi_password":             "password. Optional.",
		"ipmi_port":                 "remote IPMI RMCP port. Optional.",
		"ipmi_priv_level":           "privilege level; default is ADMINISTRATOR. One of ADMINISTRATOR, CALLBACK, OPERATOR, USER. Optional.",
		"ipmi_protocol_version":     "the version of the IPMI protocol; default is \"2.0\". One of \"1.5\", \"2.0\". Optional.",
		"ipmi_target_address":       "destination address for bridged request. Required only if ipmi_bridging is set to \"single\" or \"dual\".",
		"ipmi_target_channel":       "destination channel for bridged request. Required only if ipmi_bridging is set to \"single\" or \"dual\".",
		"ipmi_terminal_port":        "node's UDP port to connect to. Only required for console access.",
		"ipmi_transit_address":      "transit address for bridged request. Required only if ipmi_bridging is set to \"dual\".",
		"ipmi_transit_channel":      "transit channel for bridged request. Required only if ipmi_bridging is set to \"dual\".",
		"ipmi_username":             "username; default is NULL user. Optional.",
	}

	DriverIpmiToolDisk = drivers.DiskProperties{
		"controller":               "Controller to use for this logical disk. If not specified, the driver will choose a suitable RAID controller on the bare metal node. Optional.",
		"disk_type":                "The type of disk preferred. Valid values are 'hdd' and 'ssd'. If this is not specified, disk type will not be a selection criterion for choosing backing physical disks. Optional.",
		"interface_type":           "The interface type of disk. Valid values are 'sata', 'scsi' and 'sas'. If this is not specified, interface type will not be a selection criterion for choosing backing physical disks. Optional.",
		"is_root_volume":           "Specifies whether this disk is a root volume. By default, this is False. Optional.",
		"number_of_physical_disks": "Number of physical disks to use for this logical disk. By default, the driver uses the minimum number of disks required for that RAID level. Optional.",
		"physical_disks":           "The physical disks to use for this logical disk. If not specified, the driver will choose suitable physical disks to use. Optional.",
		"raid_level":               "RAID level for the logical disk. Valid values are 'JBOD', '0', '1', '2', '5', '6', '1+0', '5+0' and '6+0'. Required.",
		"share_physical_disks":     "Specifies whether other logical disks can share physical disks with this logical disk. By default, this is False. Optional.",
		"size_gb":                  "Size in GiB (Integer) for the logical disk. Use 'MAX' as size_gb if this logical disk is supposed to use the rest of the space available. Required.",
		"volume_name":              "Name of the volume to be created. If this is not specified, it will be auto-generated. Optional.",
	}
)

// HandleListDriversSuccessfully sets up the test server to respond to a drivers ListDrivers request.
func HandleListDriversSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/drivers", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.Header().Add("Content-Type", "application/json")
		r.ParseForm()

		fmt.Fprintf(w, ListDriversBody)
	})
}

// HandleGetDriverDetailsSuccessfully sets up the test server to respond to a drivers GetDriverDetails request.
func HandleGetDriverDetailsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/drivers/ipmi", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, SingleDriverDetails)
	})
}

// HandleGetDriverPropertiesSuccessfully sets up the test server to respond to a drivers GetDriverProperties request.
func HandleGetDriverPropertiesSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/drivers/agent_ipmitool/properties", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, SingleDriverProperties)
	})
}

// HandleGetDriverDiskPropertiesSuccessfully sets up the test server to respond to a drivers GetDriverDiskProperties request.
func HandleGetDriverDiskPropertiesSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/drivers/agent_ipmitool/raid/logical_disk_properties", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, SingleDriverDiskProperties)
	})
}
