package testing

import (
	"github.com/gophercloud/gophercloud/openstack/networking/v2/subnets"
)

const SubnetListResult = `
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
`

var Subnet1 = subnets.Subnet{
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
}

var Subnet2 = subnets.Subnet{
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
}

var Subnet3 = subnets.Subnet{
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
}

const SubnetGetResult = `
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
`

const SubnetCreateRequest = `
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
`

const SubnetCreateResult = `
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
`

const SubnetCreateWithNoGatewayRequest = `
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
`

const SubnetCreateWithNoGatewayResponse = `
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
`

const SubnetCreateWithDefaultGatewayRequest = `
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
`

const SubnetCreateWithDefaultGatewayResponse = `
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
`

const SubnetUpdateRequest = `
{
    "subnet": {
        "name": "my_new_subnet",
        "dns_nameservers": ["foo"],
        "host_routes": [{"destination":"","nexthop": "bar"}]
    }
}
`

const SubnetUpdateResponse = `
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
`

const SubnetUpdateGatewayRequest = `
{
    "subnet": {
        "name": "my_new_subnet",
        "gateway_ip": "10.0.0.1"
    }
}
`

const SubnetUpdateGatewayResponse = `
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
`

const SubnetUpdateRemoveGatewayRequest = `
{
    "subnet": {
        "name": "my_new_subnet",
        "gateway_ip": null
    }
}
`

const SubnetUpdateRemoveGatewayResponse = `
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
`

const SubnetUpdateAllocationPoolRequest = `
{
    "subnet": {
        "name": "my_new_subnet",
        "allocation_pools": [
            {
                "start": "10.1.0.2",
                "end": "10.1.0.254"
            }
        ]
    }
}
`

const SubnetUpdateAllocationPoolResponse = `
{
    "subnet": {
        "name": "my_new_subnet",
        "enable_dhcp": true,
        "network_id": "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
        "tenant_id": "26a7980765d0414dbc1fc1f88cdb7e6e",
        "dns_nameservers": [],
        "allocation_pools": [
            {
                "start": "10.1.0.2",
                "end": "10.1.0.254"
            }
        ],
        "host_routes": [],
        "ip_version": 4,
        "gateway_ip": "10.0.0.1",
        "cidr": "10.0.0.0/24",
        "id": "08eae331-0402-425a-923c-34f7cfe39c1b"
    }
}
`
