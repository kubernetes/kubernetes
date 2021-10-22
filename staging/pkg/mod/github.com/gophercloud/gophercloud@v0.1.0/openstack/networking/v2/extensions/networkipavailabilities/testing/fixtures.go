package testing

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/networkipavailabilities"
)

// NetworkIPAvailabilityListResult represents raw server response from a server to a list call.
const NetworkIPAvailabilityListResult = `
{
    "network_ip_availabilities": [
        {
            "network_id": "080ee064-036d-405a-a307-3bde4a213a1b",
            "network_name": "private",
            "project_id": "fb57277ef2f84a0e85b9018ec2dedbf7",
            "subnet_ip_availability": [
                {
                    "cidr": "fdbc:bf53:567e::/64",
                    "ip_version": 6,
                    "subnet_id": "497ac4d3-0b92-42cf-82de-71302ab2b656",
                    "subnet_name": "ipv6-private-subnet",
                    "total_ips": 18446744073709552000,
                    "used_ips": 2
                },
                {
                    "cidr": "10.0.0.0/26",
                    "ip_version": 4,
                    "subnet_id": "521f47e7-c4fb-452c-b71a-851da38cc571",
                    "subnet_name": "private-subnet",
                    "total_ips": 61,
                    "used_ips": 2
                }
            ],
            "tenant_id": "fb57277ef2f84a0e85b9018ec2dedbf7",
            "total_ips": 122,
            "used_ips": 14
        },
        {
            "network_id": "cf11ab78-2302-49fa-870f-851a08c7afb8",
            "network_name": "public",
            "project_id": "424e7cf0243c468ca61732ba45973b3e",
            "subnet_ip_availability": [
                {
                    "cidr": "203.0.113.0/24",
                    "ip_version": 4,
                    "subnet_id": "4afe6e5f-9649-40db-b18f-64c7ead942bd",
                    "subnet_name": "public-subnet",
                    "total_ips": 253,
                    "used_ips": 3
                }
            ],
            "tenant_id": "424e7cf0243c468ca61732ba45973b3e",
            "total_ips": 253,
            "used_ips": 3
        }
    ]
}
`

// NetworkIPAvailability1 is an expected representation of a first object from the ResourceListResult.
var NetworkIPAvailability1 = networkipavailabilities.NetworkIPAvailability{
	NetworkID:   "080ee064-036d-405a-a307-3bde4a213a1b",
	NetworkName: "private",
	ProjectID:   "fb57277ef2f84a0e85b9018ec2dedbf7",
	TenantID:    "fb57277ef2f84a0e85b9018ec2dedbf7",
	TotalIPs:    "122",
	UsedIPs:     "14",
	SubnetIPAvailabilities: []networkipavailabilities.SubnetIPAvailability{
		{
			SubnetID:   "497ac4d3-0b92-42cf-82de-71302ab2b656",
			SubnetName: "ipv6-private-subnet",
			CIDR:       "fdbc:bf53:567e::/64",
			IPVersion:  int(gophercloud.IPv6),
			TotalIPs:   "18446744073709552000",
			UsedIPs:    "2",
		},
		{
			SubnetID:   "521f47e7-c4fb-452c-b71a-851da38cc571",
			SubnetName: "private-subnet",
			CIDR:       "10.0.0.0/26",
			IPVersion:  int(gophercloud.IPv4),
			TotalIPs:   "61",
			UsedIPs:    "2",
		},
	},
}

// NetworkIPAvailability2 is an expected representation of a first object from the ResourceListResult.
var NetworkIPAvailability2 = networkipavailabilities.NetworkIPAvailability{
	NetworkID:   "cf11ab78-2302-49fa-870f-851a08c7afb8",
	NetworkName: "public",
	ProjectID:   "424e7cf0243c468ca61732ba45973b3e",
	TenantID:    "424e7cf0243c468ca61732ba45973b3e",
	TotalIPs:    "253",
	UsedIPs:     "3",
	SubnetIPAvailabilities: []networkipavailabilities.SubnetIPAvailability{
		{
			SubnetID:   "4afe6e5f-9649-40db-b18f-64c7ead942bd",
			SubnetName: "public-subnet",
			CIDR:       "203.0.113.0/24",
			IPVersion:  int(gophercloud.IPv4),
			TotalIPs:   "253",
			UsedIPs:    "3",
		},
	},
}

// NetworkIPAvailabilityGetResult represents raw server response from a server to a get call.
const NetworkIPAvailabilityGetResult = `
{
    "network_ip_availability": {
        "network_id": "cf11ab78-2302-49fa-870f-851a08c7afb8",
        "network_name": "public",
        "project_id": "424e7cf0243c468ca61732ba45973b3e",
        "subnet_ip_availability": [
            {
                "cidr": "203.0.113.0/24",
                "ip_version": 4,
                "subnet_id": "4afe6e5f-9649-40db-b18f-64c7ead942bd",
                "subnet_name": "public-subnet",
                "total_ips": 253,
                "used_ips": 3
            }
        ],
        "tenant_id": "424e7cf0243c468ca61732ba45973b3e",
        "total_ips": 253,
        "used_ips": 3
    }
}
`
