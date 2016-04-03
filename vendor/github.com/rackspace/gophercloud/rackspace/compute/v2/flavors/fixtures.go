// +build fixtures

package flavors

// ListOutput is a sample response of a flavor List request.
const ListOutput = `
{
  "flavors": [
    {
      "OS-FLV-EXT-DATA:ephemeral": 0,
      "OS-FLV-WITH-EXT-SPECS:extra_specs": {
        "class": "performance1",
        "disk_io_index": "40",
        "number_of_data_disks": "0",
        "policy_class": "performance_flavor",
        "resize_policy_class": "performance_flavor"
      },
      "disk": 20,
      "id": "performance1-1",
      "links": [
        {
          "href": "https://iad.servers.api.rackspacecloud.com/v2/864477/flavors/performance1-1",
          "rel": "self"
        },
        {
          "href": "https://iad.servers.api.rackspacecloud.com/864477/flavors/performance1-1",
          "rel": "bookmark"
        }
      ],
      "name": "1 GB Performance",
      "ram": 1024,
      "rxtx_factor": 200,
      "swap": "",
      "vcpus": 1
    },
    {
      "OS-FLV-EXT-DATA:ephemeral": 20,
      "OS-FLV-WITH-EXT-SPECS:extra_specs": {
        "class": "performance1",
        "disk_io_index": "40",
        "number_of_data_disks": "1",
        "policy_class": "performance_flavor",
        "resize_policy_class": "performance_flavor"
      },
      "disk": 40,
      "id": "performance1-2",
      "links": [
        {
          "href": "https://iad.servers.api.rackspacecloud.com/v2/864477/flavors/performance1-2",
          "rel": "self"
        },
        {
          "href": "https://iad.servers.api.rackspacecloud.com/864477/flavors/performance1-2",
          "rel": "bookmark"
        }
      ],
      "name": "2 GB Performance",
      "ram": 2048,
      "rxtx_factor": 400,
      "swap": "",
      "vcpus": 2
    }
  ]
}`

// GetOutput is a sample response from a flavor Get request. Its contents correspond to the
// Performance1Flavor struct.
const GetOutput = `
{
  "flavor": {
    "OS-FLV-EXT-DATA:ephemeral": 0,
    "OS-FLV-WITH-EXT-SPECS:extra_specs": {
      "class": "performance1",
      "disk_io_index": "40",
      "number_of_data_disks": "0",
      "policy_class": "performance_flavor",
      "resize_policy_class": "performance_flavor"
    },
    "disk": 20,
    "id": "performance1-1",
    "links": [
      {
        "href": "https://iad.servers.api.rackspacecloud.com/v2/864477/flavors/performance1-1",
        "rel": "self"
      },
      {
        "href": "https://iad.servers.api.rackspacecloud.com/864477/flavors/performance1-1",
        "rel": "bookmark"
      }
    ],
    "name": "1 GB Performance",
    "ram": 1024,
    "rxtx_factor": 200,
    "swap": "",
    "vcpus": 1
  }
}
`

// Performance1Flavor is the expected result of parsing GetOutput, or the first element of
// ListOutput.
var Performance1Flavor = Flavor{
	ID:         "performance1-1",
	Disk:       20,
	RAM:        1024,
	Name:       "1 GB Performance",
	RxTxFactor: 200.0,
	Swap:       0,
	VCPUs:      1,
	ExtraSpecs: ExtraSpecs{
		NumDataDisks: 0,
		Class:        "performance1",
		DiskIOIndex:  0,
		PolicyClass:  "performance_flavor",
	},
}

// Performance2Flavor is the second result expected from parsing ListOutput.
var Performance2Flavor = Flavor{
	ID:         "performance1-2",
	Disk:       40,
	RAM:        2048,
	Name:       "2 GB Performance",
	RxTxFactor: 400.0,
	Swap:       0,
	VCPUs:      2,
	ExtraSpecs: ExtraSpecs{
		NumDataDisks: 0,
		Class:        "performance1",
		DiskIOIndex:  0,
		PolicyClass:  "performance_flavor",
	},
}

// ExpectedFlavorSlice is the slice of Flavor structs that are expected to be parsed from
// ListOutput.
var ExpectedFlavorSlice = []Flavor{Performance1Flavor, Performance2Flavor}
