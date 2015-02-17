// +build fixtures

package servers

import (
	os "github.com/rackspace/gophercloud/openstack/compute/v2/servers"
)

// ListOutput is the recorded output of a Rackspace servers.List request.
const ListOutput = `
{
	"servers": [
		{
			"OS-DCF:diskConfig": "MANUAL",
			"OS-EXT-STS:power_state": 1,
			"OS-EXT-STS:task_state": null,
			"OS-EXT-STS:vm_state": "active",
			"accessIPv4": "1.2.3.4",
			"accessIPv6": "1111:4822:7818:121:2000:9b5e:7438:a2d0",
			"addresses": {
				"private": [
					{
						"addr": "10.208.230.113",
						"version": 4
					}
				],
				"public": [
					{
						"addr": "2001:4800:7818:101:2000:9b5e:7428:a2d0",
						"version": 6
					},
					{
						"addr": "104.130.131.164",
						"version": 4
					}
				]
			},
			"created": "2014-09-23T12:34:58Z",
			"flavor": {
				"id": "performance1-8",
				"links": [
					{
						"href": "https://dfw.servers.api.rackspacecloud.com/111111/flavors/performance1-8",
						"rel": "bookmark"
					}
				]
			},
			"hostId": "e8951a524bc465b0898aeac7674da6fe1495e253ae1ea17ddb2c2475",
			"id": "59818cee-bc8c-44eb-8073-673ee65105f7",
			"image": {
				"id": "255df5fb-e3d4-45a3-9a07-c976debf7c14",
				"links": [
					{
						"href": "https://dfw.servers.api.rackspacecloud.com/111111/images/255df5fb-e3d4-45a3-9a07-c976debf7c14",
						"rel": "bookmark"
					}
				]
			},
			"key_name": "mykey",
			"links": [
				{
					"href": "https://dfw.servers.api.rackspacecloud.com/v2/111111/servers/59818cee-bc8c-44eb-8073-673ee65105f7",
					"rel": "self"
				},
				{
					"href": "https://dfw.servers.api.rackspacecloud.com/111111/servers/59818cee-bc8c-44eb-8073-673ee65105f7",
					"rel": "bookmark"
				}
			],
			"metadata": {},
			"name": "devstack",
			"progress": 100,
			"status": "ACTIVE",
			"tenant_id": "111111",
			"updated": "2014-09-23T12:38:19Z",
			"user_id": "14ae7bb21d81422694655f3cc30f2930"
		},
		{
			"OS-DCF:diskConfig": "MANUAL",
			"OS-EXT-STS:power_state": 1,
			"OS-EXT-STS:task_state": null,
			"OS-EXT-STS:vm_state": "active",
			"accessIPv4": "1.1.2.3",
			"accessIPv6": "2222:4444:7817:101:be76:4eff:f0e5:9e02",
			"addresses": {
				"private": [
					{
						"addr": "10.10.20.30",
						"version": 4
					}
				],
				"public": [
					{
						"addr": "1.1.2.3",
						"version": 4
					},
					{
						"addr": "2222:4444:7817:101:be76:4eff:f0e5:9e02",
						"version": 6
					}
				]
			},
			"created": "2014-07-21T19:32:55Z",
			"flavor": {
				"id": "performance1-2",
				"links": [
					{
						"href": "https://dfw.servers.api.rackspacecloud.com/111111/flavors/performance1-2",
						"rel": "bookmark"
					}
				]
			},
			"hostId": "f859679906d6b1a38c1bd516b78f4dcc7d5fcf012578fa3ce460716c",
			"id": "25f1c7f5-e00a-4715-b354-16e24b2f4630",
			"image": {
				"id": "bb02b1a3-bc77-4d17-ab5b-421d89850fca",
				"links": [
					{
						"href": "https://dfw.servers.api.rackspacecloud.com/111111/images/bb02b1a3-bc77-4d17-ab5b-421d89850fca",
						"rel": "bookmark"
					}
				]
			},
			"key_name": "otherkey",
			"links": [
				{
					"href": "https://dfw.servers.api.rackspacecloud.com/v2/111111/servers/25f1c7f5-e00a-4715-b355-16e24b2f4630",
					"rel": "self"
				},
				{
					"href": "https://dfw.servers.api.rackspacecloud.com/111111/servers/25f1c7f5-e00a-4715-b355-16e24b2f4630",
					"rel": "bookmark"
				}
			],
			"metadata": {},
			"name": "peril-dfw",
			"progress": 100,
			"status": "ACTIVE",
			"tenant_id": "111111",
			"updated": "2014-07-21T19:34:24Z",
			"user_id": "14ae7bb21d81422694655f3cc30f2930"
		}
	]
}
`

// GetOutput is the recorded output of a Rackspace servers.Get request.
const GetOutput = `
{
	"server": {
		"OS-DCF:diskConfig": "AUTO",
		"OS-EXT-STS:power_state": 1,
		"OS-EXT-STS:task_state": null,
		"OS-EXT-STS:vm_state": "active",
		"accessIPv4": "1.2.4.8",
		"accessIPv6": "2001:4800:6666:105:2a0f:c056:f594:7777",
		"addresses": {
			"private": [
				{
					"addr": "10.20.40.80",
					"version": 4
				}
			],
			"public": [
				{
					"addr": "1.2.4.8",
					"version": 4
				},
				{
					"addr": "2001:4800:6666:105:2a0f:c056:f594:7777",
					"version": 6
				}
			]
		},
		"created": "2014-10-21T14:42:16Z",
		"flavor": {
			"id": "performance1-1",
			"links": [
				{
					"href": "https://dfw.servers.api.rackspacecloud.com/111111/flavors/performance1-1",
					"rel": "bookmark"
				}
			]
		},
		"hostId": "430d2ae02de0a7af77012c94778145eccf67e75b1fac0528aa10d4a7",
		"id": "8c65cb68-0681-4c30-bc88-6b83a8a26aee",
		"image": {
			"id": "e19a734c-c7e6-443a-830c-242209c4d65d",
			"links": [
				{
					"href": "https://dfw.servers.api.rackspacecloud.com/111111/images/e19a734c-c7e6-443a-830c-242209c4d65d",
					"rel": "bookmark"
				}
			]
		},
		"key_name": null,
		"links": [
			{
				"href": "https://dfw.servers.api.rackspacecloud.com/v2/111111/servers/8c65cb68-0681-4c30-bc88-6b83a8a26aee",
				"rel": "self"
			},
			{
				"href": "https://dfw.servers.api.rackspacecloud.com/111111/servers/8c65cb68-0681-4c30-bc88-6b83a8a26aee",
				"rel": "bookmark"
			}
		],
		"metadata": {},
		"name": "Gophercloud-pxpGGuey",
		"progress": 100,
		"status": "ACTIVE",
		"tenant_id": "111111",
		"updated": "2014-10-21T14:42:57Z",
		"user_id": "14ae7bb21d81423694655f4dd30f2930"
	}
}
`

// UpdateOutput is the recorded output of a Rackspace servers.Update request.
const UpdateOutput = `
{
	"server": {
		"OS-DCF:diskConfig": "AUTO",
		"OS-EXT-STS:power_state": 1,
		"OS-EXT-STS:task_state": null,
		"OS-EXT-STS:vm_state": "active",
		"accessIPv4": "1.2.4.8",
		"accessIPv6": "2001:4800:6666:105:2a0f:c056:f594:7777",
		"addresses": {
			"private": [
			{
				"addr": "10.20.40.80",
				"version": 4
			}
			],
			"public": [
			{
				"addr": "1.2.4.8",
				"version": 4
				},
				{
					"addr": "2001:4800:6666:105:2a0f:c056:f594:7777",
					"version": 6
				}
			]
		},
		"created": "2014-10-21T14:42:16Z",
		"flavor": {
			"id": "performance1-1",
			"links": [
			{
				"href": "https://dfw.servers.api.rackspacecloud.com/111111/flavors/performance1-1",
				"rel": "bookmark"
			}
			]
		},
		"hostId": "430d2ae02de0a7af77012c94778145eccf67e75b1fac0528aa10d4a7",
		"id": "8c65cb68-0681-4c30-bc88-6b83a8a26aee",
		"image": {
			"id": "e19a734c-c7e6-443a-830c-242209c4d65d",
			"links": [
			{
				"href": "https://dfw.servers.api.rackspacecloud.com/111111/images/e19a734c-c7e6-443a-830c-242209c4d65d",
				"rel": "bookmark"
			}
			]
		},
		"key_name": null,
		"links": [
		{
			"href": "https://dfw.servers.api.rackspacecloud.com/v2/111111/servers/8c65cb68-0681-4c30-bc88-6b83a8a26aee",
			"rel": "self"
		},
		{
			"href": "https://dfw.servers.api.rackspacecloud.com/111111/servers/8c65cb68-0681-4c30-bc88-6b83a8a26aee",
			"rel": "bookmark"
		}
		],
		"metadata": {},
		"name": "test-server-updated",
		"progress": 100,
		"status": "ACTIVE",
		"tenant_id": "111111",
		"updated": "2014-10-21T14:42:57Z",
		"user_id": "14ae7bb21d81423694655f4dd30f2930"
	}
}
`

// CreateOutput contains a sample of Rackspace's response to a Create call.
const CreateOutput = `
{
	"server": {
		"OS-DCF:diskConfig": "AUTO",
		"adminPass": "v7tADqbE5pr9",
		"id": "bb63327b-6a2f-34bc-b0ef-4b6d97ea637e",
		"links": [
			{
				"href": "https://dfw.servers.api.rackspacecloud.com/v2/111111/servers/bb63327b-6a2f-34bc-b0ef-4b6d97ea637e",
				"rel": "self"
			},
			{
				"href": "https://dfw.servers.api.rackspacecloud.com/111111/servers/bb63327b-6a2f-34bc-b0ef-4b6d97ea637e",
				"rel": "bookmark"
			}
		]
	}
}
`

// DevstackServer is the expected first result from parsing ListOutput.
var DevstackServer = os.Server{
	ID:         "59818cee-bc8c-44eb-8073-673ee65105f7",
	Name:       "devstack",
	TenantID:   "111111",
	UserID:     "14ae7bb21d81422694655f3cc30f2930",
	HostID:     "e8951a524bc465b0898aeac7674da6fe1495e253ae1ea17ddb2c2475",
	Updated:    "2014-09-23T12:38:19Z",
	Created:    "2014-09-23T12:34:58Z",
	AccessIPv4: "1.2.3.4",
	AccessIPv6: "1111:4822:7818:121:2000:9b5e:7438:a2d0",
	Progress:   100,
	Status:     "ACTIVE",
	Image: map[string]interface{}{
		"id": "255df5fb-e3d4-45a3-9a07-c976debf7c14",
		"links": []interface{}{
			map[string]interface{}{
				"href": "https://dfw.servers.api.rackspacecloud.com/111111/images/255df5fb-e3d4-45a3-9a07-c976debf7c14",
				"rel":  "bookmark",
			},
		},
	},
	Flavor: map[string]interface{}{
		"id": "performance1-8",
		"links": []interface{}{
			map[string]interface{}{
				"href": "https://dfw.servers.api.rackspacecloud.com/111111/flavors/performance1-8",
				"rel":  "bookmark",
			},
		},
	},
	Addresses: map[string]interface{}{
		"private": []interface{}{
			map[string]interface{}{
				"addr":    "10.20.30.40",
				"version": float64(4.0),
			},
		},
		"public": []interface{}{
			map[string]interface{}{
				"addr":    "1111:4822:7818:121:2000:9b5e:7438:a2d0",
				"version": float64(6.0),
			},
			map[string]interface{}{
				"addr":    "1.2.3.4",
				"version": float64(4.0),
			},
		},
	},
	Metadata: map[string]interface{}{},
	Links: []interface{}{
		map[string]interface{}{
			"href": "https://dfw.servers.api.rackspacecloud.com/v2/111111/servers/59918cee-bd9d-44eb-8173-673ee75105f7",
			"rel":  "self",
		},
		map[string]interface{}{
			"href": "https://dfw.servers.api.rackspacecloud.com/v2/111111/servers/59818cee-bc8c-44eb-8073-673ee65105f7",
			"rel":  "bookmark",
		},
	},
	KeyName:   "mykey",
	AdminPass: "",
}

// PerilServer is the expected second result from parsing ListOutput.
var PerilServer = os.Server{
	ID:         "25f1c7f5-e00a-4715-b354-16e24b2f4630",
	Name:       "peril-dfw",
	TenantID:   "111111",
	UserID:     "14ae7bb21d81422694655f3cc30f2930",
	HostID:     "f859679906d6b1a38c1bd516b78f4dcc7d5fcf012578fa3ce460716c",
	Updated:    "2014-07-21T19:34:24Z",
	Created:    "2014-07-21T19:32:55Z",
	AccessIPv4: "1.1.2.3",
	AccessIPv6: "2222:4444:7817:101:be76:4eff:f0e5:9e02",
	Progress:   100,
	Status:     "ACTIVE",
	Image: map[string]interface{}{
		"id": "bb02b1a3-bc77-4d17-ab5b-421d89850fca",
		"links": []interface{}{
			map[string]interface{}{
				"href": "https://dfw.servers.api.rackspacecloud.com/111111/images/bb02b1a3-bc77-4d17-ab5b-421d89850fca",
				"rel":  "bookmark",
			},
		},
	},
	Flavor: map[string]interface{}{
		"id": "performance1-2",
		"links": []interface{}{
			map[string]interface{}{
				"href": "https://dfw.servers.api.rackspacecloud.com/111111/flavors/performance1-2",
				"rel":  "bookmark",
			},
		},
	},
	Addresses: map[string]interface{}{
		"private": []interface{}{
			map[string]interface{}{
				"addr":    "10.10.20.30",
				"version": float64(4.0),
			},
		},
		"public": []interface{}{
			map[string]interface{}{
				"addr":    "2222:4444:7817:101:be76:4eff:f0e5:9e02",
				"version": float64(6.0),
			},
			map[string]interface{}{
				"addr":    "1.1.2.3",
				"version": float64(4.0),
			},
		},
	},
	Metadata: map[string]interface{}{},
	Links: []interface{}{
		map[string]interface{}{
			"href": "https://dfw.servers.api.rackspacecloud.com/v2/111111/servers/25f1c7f5-e00a-4715-b355-16e24b2f4630",
			"rel":  "self",
		},
		map[string]interface{}{
			"href": "https://dfw.servers.api.rackspacecloud.com/v2/111111/servers/25f1c7f5-e00a-4715-b355-16e24b2f4630",
			"rel":  "bookmark",
		},
	},
	KeyName:   "otherkey",
	AdminPass: "",
}

// GophercloudServer is the expected result from parsing GetOutput.
var GophercloudServer = os.Server{
	ID:         "8c65cb68-0681-4c30-bc88-6b83a8a26aee",
	Name:       "Gophercloud-pxpGGuey",
	TenantID:   "111111",
	UserID:     "14ae7bb21d81423694655f4dd30f2930",
	HostID:     "430d2ae02de0a7af77012c94778145eccf67e75b1fac0528aa10d4a7",
	Updated:    "2014-10-21T14:42:57Z",
	Created:    "2014-10-21T14:42:16Z",
	AccessIPv4: "1.2.4.8",
	AccessIPv6: "2001:4800:6666:105:2a0f:c056:f594:7777",
	Progress:   100,
	Status:     "ACTIVE",
	Image: map[string]interface{}{
		"id": "e19a734c-c7e6-443a-830c-242209c4d65d",
		"links": []interface{}{
			map[string]interface{}{
				"href": "https://dfw.servers.api.rackspacecloud.com/111111/images/e19a734c-c7e6-443a-830c-242209c4d65d",
				"rel":  "bookmark",
			},
		},
	},
	Flavor: map[string]interface{}{
		"id": "performance1-1",
		"links": []interface{}{
			map[string]interface{}{
				"href": "https://dfw.servers.api.rackspacecloud.com/111111/flavors/performance1-1",
				"rel":  "bookmark",
			},
		},
	},
	Addresses: map[string]interface{}{
		"private": []interface{}{
			map[string]interface{}{
				"addr":    "10.20.40.80",
				"version": float64(4.0),
			},
		},
		"public": []interface{}{
			map[string]interface{}{
				"addr":    "2001:4800:6666:105:2a0f:c056:f594:7777",
				"version": float64(6.0),
			},
			map[string]interface{}{
				"addr":    "1.2.4.8",
				"version": float64(4.0),
			},
		},
	},
	Metadata: map[string]interface{}{},
	Links: []interface{}{
		map[string]interface{}{
			"href": "https://dfw.servers.api.rackspacecloud.com/v2/111111/servers/8c65cb68-0681-4c30-bc88-6b83a8a26aee",
			"rel":  "self",
		},
		map[string]interface{}{
			"href": "https://dfw.servers.api.rackspacecloud.com/111111/servers/8c65cb68-0681-4c30-bc88-6b83a8a26aee",
			"rel":  "bookmark",
		},
	},
	KeyName:   "",
	AdminPass: "",
}

// GophercloudUpdatedServer is the expected result from parsing UpdateOutput.
var GophercloudUpdatedServer = os.Server{
	ID:         "8c65cb68-0681-4c30-bc88-6b83a8a26aee",
	Name:       "test-server-updated",
	TenantID:   "111111",
	UserID:     "14ae7bb21d81423694655f4dd30f2930",
	HostID:     "430d2ae02de0a7af77012c94778145eccf67e75b1fac0528aa10d4a7",
	Updated:    "2014-10-21T14:42:57Z",
	Created:    "2014-10-21T14:42:16Z",
	AccessIPv4: "1.2.4.8",
	AccessIPv6: "2001:4800:6666:105:2a0f:c056:f594:7777",
	Progress:   100,
	Status:     "ACTIVE",
	Image: map[string]interface{}{
		"id": "e19a734c-c7e6-443a-830c-242209c4d65d",
		"links": []interface{}{
			map[string]interface{}{
				"href": "https://dfw.servers.api.rackspacecloud.com/111111/images/e19a734c-c7e6-443a-830c-242209c4d65d",
				"rel":  "bookmark",
			},
		},
	},
	Flavor: map[string]interface{}{
		"id": "performance1-1",
		"links": []interface{}{
			map[string]interface{}{
				"href": "https://dfw.servers.api.rackspacecloud.com/111111/flavors/performance1-1",
				"rel":  "bookmark",
			},
		},
	},
	Addresses: map[string]interface{}{
		"private": []interface{}{
			map[string]interface{}{
				"addr":    "10.20.40.80",
				"version": float64(4.0),
			},
		},
		"public": []interface{}{
			map[string]interface{}{
				"addr":    "2001:4800:6666:105:2a0f:c056:f594:7777",
				"version": float64(6.0),
			},
			map[string]interface{}{
				"addr":    "1.2.4.8",
				"version": float64(4.0),
			},
		},
	},
	Metadata: map[string]interface{}{},
	Links: []interface{}{
		map[string]interface{}{
			"href": "https://dfw.servers.api.rackspacecloud.com/v2/111111/servers/8c65cb68-0681-4c30-bc88-6b83a8a26aee",
			"rel":  "self",
		},
		map[string]interface{}{
			"href": "https://dfw.servers.api.rackspacecloud.com/111111/servers/8c65cb68-0681-4c30-bc88-6b83a8a26aee",
			"rel":  "bookmark",
		},
	},
	KeyName:   "",
	AdminPass: "",
}

// CreatedServer is the partial Server struct that can be parsed from CreateOutput.
var CreatedServer = os.Server{
	ID:        "bb63327b-6a2f-34bc-b0ef-4b6d97ea637e",
	AdminPass: "v7tADqbE5pr9",
	Links:     []interface{}{},
}

// ExpectedServerSlice is the collection of servers, in order, that should be parsed from ListOutput.
var ExpectedServerSlice = []os.Server{DevstackServer, PerilServer}
