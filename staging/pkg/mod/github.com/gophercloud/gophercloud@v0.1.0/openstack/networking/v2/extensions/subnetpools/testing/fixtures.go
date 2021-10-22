package testing

import (
	"time"

	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/subnetpools"
)

const SubnetPoolsListResult = `
{
    "subnetpools": [
        {
            "address_scope_id": null,
            "created_at": "2017-12-28T07:21:41Z",
            "default_prefixlen": "8",
            "default_quota": null,
            "description": "IPv4",
            "id": "d43a57fe-3390-4608-b437-b1307b0adb40",
            "ip_version": 4,
            "is_default": false,
            "max_prefixlen": "32",
            "min_prefixlen": "8",
            "name": "MyPoolIpv4",
            "prefixes": [
                "10.10.10.0/24",
                "10.11.11.0/24"
            ],
            "project_id": "1e2b9857295a4a3e841809ef492812c5",
            "revision_number": 1,
            "shared": false,
            "tenant_id": "1e2b9857295a4a3e841809ef492812c5",
            "updated_at": "2017-12-28T07:21:41Z"
        },
        {
            "address_scope_id": "0bc38e22-be49-4e67-969e-fec3f36508bd",
            "created_at": "2017-12-28T07:21:34Z",
            "default_prefixlen": "64",
            "default_quota": null,
            "description": "IPv6",
            "id": "832cb7f3-59fe-40cf-8f64-8350ffc03272",
            "ip_version": 6,
            "is_default": true,
            "max_prefixlen": "128",
            "min_prefixlen": "64",
            "name": "MyPoolIpv6",
            "prefixes": [
                "fdf7:b13d:dead:beef::/64",
                "fd65:86cc:a334:39b7::/64"
            ],
            "project_id": "1e2b9857295a4a3e841809ef492812c5",
            "revision_number": 1,
            "shared": false,
            "tenant_id": "1e2b9857295a4a3e841809ef492812c5",
            "updated_at": "2017-12-28T07:21:34Z"
        },
        {
            "address_scope_id": null,
            "created_at": "2017-12-28T07:21:27Z",
            "default_prefixlen": "64",
            "default_quota": 4,
            "description": "PublicPool",
            "id": "2fe18ae6-58c2-4a85-8bfb-566d6426749b",
            "ip_version": 6,
            "is_default": false,
            "max_prefixlen": "128",
            "min_prefixlen": "64",
            "name": "PublicIPv6",
            "prefixes": [
                "2001:db8::a3/64"
            ],
            "project_id": "ceb366d50ad54fe39717df3af60f9945",
            "revision_number": 1,
            "shared": true,
            "tenant_id": "ceb366d50ad54fe39717df3af60f9945",
            "updated_at": "2017-12-28T07:21:27Z"
        }
    ]
}
`

var SubnetPool1 = subnetpools.SubnetPool{
	AddressScopeID:   "",
	CreatedAt:        time.Date(2017, 12, 28, 7, 21, 41, 0, time.UTC),
	DefaultPrefixLen: 8,
	DefaultQuota:     0,
	Description:      "IPv4",
	ID:               "d43a57fe-3390-4608-b437-b1307b0adb40",
	IPversion:        4,
	IsDefault:        false,
	MaxPrefixLen:     32,
	MinPrefixLen:     8,
	Name:             "MyPoolIpv4",
	Prefixes: []string{
		"10.10.10.0/24",
		"10.11.11.0/24",
	},
	ProjectID:      "1e2b9857295a4a3e841809ef492812c5",
	TenantID:       "1e2b9857295a4a3e841809ef492812c5",
	RevisionNumber: 1,
	Shared:         false,
	UpdatedAt:      time.Date(2017, 12, 28, 7, 21, 41, 0, time.UTC),
}

var SubnetPool2 = subnetpools.SubnetPool{
	AddressScopeID:   "0bc38e22-be49-4e67-969e-fec3f36508bd",
	CreatedAt:        time.Date(2017, 12, 28, 7, 21, 34, 0, time.UTC),
	DefaultPrefixLen: 64,
	DefaultQuota:     0,
	Description:      "IPv6",
	ID:               "832cb7f3-59fe-40cf-8f64-8350ffc03272",
	IPversion:        6,
	IsDefault:        true,
	MaxPrefixLen:     128,
	MinPrefixLen:     64,
	Name:             "MyPoolIpv6",
	Prefixes: []string{
		"fdf7:b13d:dead:beef::/64",
		"fd65:86cc:a334:39b7::/64",
	},
	ProjectID:      "1e2b9857295a4a3e841809ef492812c5",
	TenantID:       "1e2b9857295a4a3e841809ef492812c5",
	RevisionNumber: 1,
	Shared:         false,
	UpdatedAt:      time.Date(2017, 12, 28, 7, 21, 34, 0, time.UTC),
}

var SubnetPool3 = subnetpools.SubnetPool{
	AddressScopeID:   "",
	CreatedAt:        time.Date(2017, 12, 28, 7, 21, 27, 0, time.UTC),
	DefaultPrefixLen: 64,
	DefaultQuota:     4,
	Description:      "PublicPool",
	ID:               "2fe18ae6-58c2-4a85-8bfb-566d6426749b",
	IPversion:        6,
	IsDefault:        false,
	MaxPrefixLen:     128,
	MinPrefixLen:     64,
	Name:             "PublicIPv6",
	Prefixes: []string{
		"2001:db8::a3/64",
	},
	ProjectID:      "ceb366d50ad54fe39717df3af60f9945",
	TenantID:       "ceb366d50ad54fe39717df3af60f9945",
	RevisionNumber: 1,
	Shared:         true,
	UpdatedAt:      time.Date(2017, 12, 28, 7, 21, 27, 0, time.UTC),
}

const SubnetPoolGetResult = `
{
    "subnetpool": {
        "min_prefixlen": "64",
        "address_scope_id": null,
        "default_prefixlen": "64",
        "id": "0a738452-8057-4ad3-89c2-92f6a74afa76",
        "max_prefixlen": "128",
        "name": "my-ipv6-pool",
        "default_quota": 2,
        "is_default": true,
        "project_id": "1e2b9857295a4a3e841809ef492812c5",
        "tenant_id": "1e2b9857295a4a3e841809ef492812c5",
        "created_at": "2018-01-01T00:00:01Z",
        "prefixes": [
            "2001:db8::a3/64"
        ],
        "updated_at": "2018-01-01T00:10:10Z",
        "ip_version": 6,
        "shared": false,
        "description": "ipv6 prefixes",
        "revision_number": 2
    }
}
`

const SubnetPoolCreateRequest = `
{
    "subnetpool": {
        "name": "my_ipv4_pool",
        "prefixes": [
            "10.10.0.0/16",
            "10.11.11.0/24"
        ],
        "address_scope_id": "3d4e2e2a-552b-42ad-a16d-820bbf3edaf3",
        "min_prefixlen": 25,
        "max_prefixlen": 30,
        "description": "ipv4 prefixes"
    }
}
`

const SubnetPoolCreateResult = `
{
    "subnetpool": {
        "address_scope_id": "3d4e2e2a-552b-42ad-a16d-820bbf3edaf3",
        "created_at": "2018-01-01T00:00:15Z",
        "default_prefixlen": "25",
        "default_quota": null,
        "description": "ipv4 prefixes",
        "id": "55b5999c-c2fe-42cd-bce0-961a551b80f5",
        "ip_version": 4,
        "is_default": false,
        "max_prefixlen": "30",
        "min_prefixlen": "25",
        "name": "my_ipv4_pool",
        "prefixes": [
            "10.10.0.0/16",
            "10.11.11.0/24"
        ],
        "project_id": "1e2b9857295a4a3e841809ef492812c5",
        "revision_number": 1,
        "shared": false,
        "tenant_id": "1e2b9857295a4a3e841809ef492812c5",
        "updated_at": "2018-01-01T00:00:15Z"
    }
}
`

const SubnetPoolUpdateRequest = `
{
    "subnetpool": {
        "name": "new_subnetpool_name",
        "prefixes": [
            "10.11.12.0/24",
            "10.24.0.0/16"
        ],
        "max_prefixlen": 16,
        "address_scope_id": "",
        "default_quota": 0,
        "description": ""
    }
}
`

const SubnetPoolUpdateResponse = `
{
    "subnetpool": {
        "address_scope_id": null,
        "created_at": "2018-01-03T07:21:34Z",
        "default_prefixlen": 8,
        "default_quota": null,
        "description": null,
        "id": "099546ca-788d-41e5-a76d-17d8cd282d3e",
        "ip_version": 4,
        "is_default": true,
        "max_prefixlen": 16,
        "min_prefixlen": 8,
        "name": "new_subnetpool_name",
        "prefixes": [
            "10.8.0.0/16",
            "10.11.12.0/24",
            "10.24.0.0/16"
        ],
        "revision_number": 2,
        "shared": false,
        "tenant_id": "1e2b9857295a4a3e841809ef492812c5",
        "updated_at": "2018-01-05T09:56:56Z"
    }
}
`
