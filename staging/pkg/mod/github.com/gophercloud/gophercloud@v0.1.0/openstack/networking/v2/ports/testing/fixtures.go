package testing

const ListResponse = `
{
    "ports": [
        {
            "status": "ACTIVE",
            "binding:host_id": "devstack",
            "name": "",
            "admin_state_up": true,
            "network_id": "70c1db1f-b701-45bd-96e0-a313ee3430b3",
            "tenant_id": "",
            "device_owner": "network:router_gateway",
            "mac_address": "fa:16:3e:58:42:ed",
            "binding:vnic_type": "normal",
            "fixed_ips": [
                {
                    "subnet_id": "008ba151-0b8c-4a67-98b5-0d2b87666062",
                    "ip_address": "172.24.4.2"
                }
            ],
            "id": "d80b1a3b-4fc1-49f3-952e-1e2ab7081d8b",
            "security_groups": [],
            "dns_name": "test-port",
            "dns_assignment": [
              {
                "hostname": "test-port",
                "ip_address": "172.24.4.2",
                "fqdn": "test-port.openstack.local."
              }
            ],
            "device_id": "9ae135f4-b6e0-4dad-9e91-3c223e385824",
            "port_security_enabled": false
        }
    ]
}
`

const GetResponse = `
{
    "port": {
        "status": "ACTIVE",
        "binding:host_id": "devstack",
        "name": "",
        "allowed_address_pairs": [],
        "admin_state_up": true,
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "7e02058126cc4950b75f9970368ba177",
        "extra_dhcp_opts": [],
        "binding:vif_details": {
            "port_filter": true,
            "ovs_hybrid_plug": true
         },
        "binding:vif_type": "ovs",
        "device_owner": "network:router_interface",
        "port_security_enabled": false,
        "mac_address": "fa:16:3e:23:fd:d7",
        "binding:profile": {},
        "binding:vnic_type": "normal",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.1"
            }
        ],
        "id": "46d4bfb9-b26e-41f3-bd2e-e6dcc1ccedb2",
        "security_groups": [],
        "dns_name": "test-port",
        "dns_assignment": [
          {
            "hostname": "test-port",
            "ip_address": "172.24.4.2",
            "fqdn": "test-port.openstack.local."
          }
        ],
        "device_id": "5e3898d7-11be-483e-9732-b2f5eccd2b2e"
    }
}
`

const CreateRequest = `
{
    "port": {
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "name": "private-port",
        "admin_state_up": true,
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.2"
            }
        ],
        "security_groups": ["foo"],
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ]
    }
}
`

const CreateResponse = `
{
    "port": {
        "status": "DOWN",
        "name": "private-port",
        "admin_state_up": true,
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.2"
            }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "security_groups": [
            "f0ac4394-7e4a-4409-9701-ba8be283dbc3"
        ],
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ],
        "device_id": ""
    }
}
`

const CreateOmitSecurityGroupsRequest = `
{
    "port": {
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "name": "private-port",
        "admin_state_up": true,
        "fixed_ips": [
          {
            "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
            "ip_address": "10.0.0.2"
          }
        ],
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ]
    }
}
`

const CreateWithNoSecurityGroupsRequest = `
{
    "port": {
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "name": "private-port",
        "admin_state_up": true,
        "fixed_ips": [
          {
            "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
            "ip_address": "10.0.0.2"
          }
        ],
        "security_groups": [],
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ]
    }
}
`

const CreateWithNoSecurityGroupsResponse = `
{
    "port": {
        "status": "DOWN",
        "name": "private-port",
        "admin_state_up": true,
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.2"
            }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ],
        "device_id": ""
    }
}
`

const CreateOmitSecurityGroupsResponse = `
{
    "port": {
        "status": "DOWN",
        "name": "private-port",
        "admin_state_up": true,
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.2"
            }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "security_groups": [
            "f0ac4394-7e4a-4409-9701-ba8be283dbc3"
        ],
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ],
        "device_id": ""
    }
}
`

const CreatePortSecurityRequest = `
{
    "port": {
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "name": "private-port",
        "admin_state_up": true,
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.2"
            }
        ],
        "security_groups": ["foo"],
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ],
        "port_security_enabled": false
    }
}
`

const CreatePortSecurityResponse = `
{
    "port": {
        "status": "DOWN",
        "name": "private-port",
        "admin_state_up": true,
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.2"
            }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "security_groups": [
            "f0ac4394-7e4a-4409-9701-ba8be283dbc3"
        ],
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ],
        "device_id": "",
        "port_security_enabled": false
    }
}
`

const UpdateRequest = `
{
    "port": {
        "name": "new_port_name",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.3"
            }
        ],
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ],
        "security_groups": [
            "f0ac4394-7e4a-4409-9701-ba8be283dbc3"
        ]
    }
}
`

const UpdateResponse = `
{
    "port": {
        "status": "DOWN",
        "name": "new_port_name",
        "admin_state_up": true,
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.3"
            }
        ],
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "security_groups": [
            "f0ac4394-7e4a-4409-9701-ba8be283dbc3"
        ],
        "device_id": ""
    }
}
`

const UpdateOmitSecurityGroupsRequest = `
{
  "port": {
      "name": "new_port_name",
      "fixed_ips": [
          {
            "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
            "ip_address": "10.0.0.3"
          }
        ],
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ]
    }
}
`

const UpdateOmitSecurityGroupsResponse = `
{
    "port": {
        "status": "DOWN",
        "name": "new_port_name",
        "admin_state_up": true,
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.3"
            }
        ],
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "security_groups": [
            "f0ac4394-7e4a-4409-9701-ba8be283dbc3"
        ],
        "device_id": ""
    }
}
`

const UpdatePortSecurityRequest = `
{
    "port": {
        "port_security_enabled": false
    }
}
`

const UpdatePortSecurityResponse = `
{
    "port": {
        "status": "DOWN",
        "name": "private-port",
        "admin_state_up": true,
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.2"
            }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "security_groups": [
            "f0ac4394-7e4a-4409-9701-ba8be283dbc3"
        ],
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ],
        "device_id": "",
        "port_security_enabled": false
    }
}
`

const RemoveSecurityGroupRequest = `
{
    "port": {
      "name": "new_port_name",
      "fixed_ips": [
        {
          "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
          "ip_address": "10.0.0.3"
        }
      ],
      "allowed_address_pairs": [
        {
          "ip_address": "10.0.0.4",
          "mac_address": "fa:16:3e:c9:cb:f0"
        }
      ],
      "security_groups": []
    }
}
`

const RemoveSecurityGroupResponse = `
{
    "port": {
        "status": "DOWN",
        "name": "new_port_name",
        "admin_state_up": true,
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.3"
            }
        ],
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "device_id": ""
    }
}
`

const RemoveAllowedAddressPairsRequest = `
{
    "port": {
      "name": "new_port_name",
      "fixed_ips": [
        {
          "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
          "ip_address": "10.0.0.3"
        }
      ],
      "allowed_address_pairs": [],
      "security_groups": [
        "f0ac4394-7e4a-4409-9701-ba8be283dbc3"
      ]
    }
}
`

const RemoveAllowedAddressPairsResponse = `
{
    "port": {
        "status": "DOWN",
        "name": "new_port_name",
        "admin_state_up": true,
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.3"
            }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "security_groups": [
            "f0ac4394-7e4a-4409-9701-ba8be283dbc3"
        ],
        "device_id": ""
    }
}
`

const DontUpdateAllowedAddressPairsRequest = `
{
    "port": {
        "name": "new_port_name",
        "fixed_ips": [
          {
            "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
            "ip_address": "10.0.0.3"
          }
        ],
        "security_groups": [
            "f0ac4394-7e4a-4409-9701-ba8be283dbc3"
        ]
    }
}
`

const DontUpdateAllowedAddressPairsResponse = `
{
    "port": {
        "status": "DOWN",
        "name": "new_port_name",
        "admin_state_up": true,
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.3"
            }
        ],
        "allowed_address_pairs": [
          {
            "ip_address": "10.0.0.4",
            "mac_address": "fa:16:3e:c9:cb:f0"
          }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "security_groups": [
            "f0ac4394-7e4a-4409-9701-ba8be283dbc3"
        ],
        "device_id": ""
    }
}
`

// GetWithExtraDHCPOptsResponse represents a raw port response with extra
// DHCP options.
const GetWithExtraDHCPOptsResponse = `
{
    "port": {
        "status": "ACTIVE",
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "extra_dhcp_opts": [
            {
                "opt_name": "option1",
                "opt_value": "value1",
                "ip_version": 4
            },
            {
                "opt_name": "option2",
                "opt_value": "value2",
                "ip_version": 4
            }
        ],
        "admin_state_up": true,
        "name": "port-with-extra-dhcp-opts",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.4"
            }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "device_id": ""
    }
}
`

// CreateWithExtraDHCPOptsRequest represents a raw port creation request
// with extra DHCP options.
const CreateWithExtraDHCPOptsRequest = `
{
    "port": {
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "name": "port-with-extra-dhcp-opts",
        "admin_state_up": true,
        "fixed_ips": [
            {
                "ip_address": "10.0.0.2",
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2"
            }
        ],
        "extra_dhcp_opts": [
            {
                "opt_name": "option1",
                "opt_value": "value1"
            }
        ]
    }
}
`

// CreateWithExtraDHCPOptsResponse represents a raw port creation response
// with extra DHCP options.
const CreateWithExtraDHCPOptsResponse = `
{
    "port": {
        "status": "DOWN",
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "extra_dhcp_opts": [
            {
                "opt_name": "option1",
                "opt_value": "value1",
                "ip_version": 4
            }
        ],
        "admin_state_up": true,
        "name": "port-with-extra-dhcp-opts",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.2"
            }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "device_id": ""
    }
}
`

// UpdateWithExtraDHCPOptsRequest represents a raw port update request with
// extra DHCP options.
const UpdateWithExtraDHCPOptsRequest = `
{
    "port": {
        "name": "updated-port-with-dhcp-opts",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.3"
            }
        ],
        "extra_dhcp_opts": [
            {
                "opt_name": "option1",
                "opt_value": null
            },
            {
                "opt_name": "option2",
                "opt_value": "value2"
            }
        ]
    }
}
`

// UpdateWithExtraDHCPOptsResponse represents a raw port update response with
// extra DHCP options.
const UpdateWithExtraDHCPOptsResponse = `
{
    "port": {
        "status": "DOWN",
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "extra_dhcp_opts": [
            {
                "opt_name": "option2",
                "opt_value": "value2",
                "ip_version": 4
            }
        ],
        "admin_state_up": true,
        "name": "updated-port-with-dhcp-opts",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.3"
            }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "device_id": ""
    }
}
`
