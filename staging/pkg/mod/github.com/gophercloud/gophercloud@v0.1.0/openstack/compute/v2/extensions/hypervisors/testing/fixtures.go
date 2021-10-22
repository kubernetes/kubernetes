package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/hypervisors"
	"github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// HypervisorListBodyPre253 represents a raw hypervisor list from the Compute
// API with microversion older than 2.53.
// The first hypervisor represents what the specification says (~Newton)
// The second is exactly the same, but what you can get off a real system (~Kilo)
const HypervisorListBodyPre253 = `
{
    "hypervisors": [
        {
            "cpu_info": {
                "arch": "x86_64",
                "model": "Nehalem",
                "vendor": "Intel",
                "features": [
                    "pge",
                    "clflush"
                ],
                "topology": {
                    "cores": 1,
                    "threads": 1,
                    "sockets": 4
                }
            },
            "current_workload": 0,
            "status": "enabled",
            "state": "up",
            "disk_available_least": 0,
            "host_ip": "1.1.1.1",
            "free_disk_gb": 1028,
            "free_ram_mb": 7680,
            "hypervisor_hostname": "fake-mini",
            "hypervisor_type": "fake",
            "hypervisor_version": 2002000,
            "id": 1,
            "local_gb": 1028,
            "local_gb_used": 0,
            "memory_mb": 8192,
            "memory_mb_used": 512,
            "running_vms": 0,
            "service": {
                "host": "e6a37ee802d74863ab8b91ade8f12a67",
                "id": 2,
                "disabled_reason": null
            },
            "vcpus": 1,
            "vcpus_used": 0
        },
        {
            "cpu_info": "{\"arch\": \"x86_64\", \"model\": \"Nehalem\", \"vendor\": \"Intel\", \"features\": [\"pge\", \"clflush\"], \"topology\": {\"cores\": 1, \"threads\": 1, \"sockets\": 4}}",
            "current_workload": 0,
            "status": "enabled",
            "state": "up",
            "disk_available_least": 0,
            "host_ip": "1.1.1.1",
            "free_disk_gb": 1028,
            "free_ram_mb": 7680,
            "hypervisor_hostname": "fake-mini",
            "hypervisor_type": "fake",
            "hypervisor_version": 2.002e+06,
            "id": 1,
            "local_gb": 1028,
            "local_gb_used": 0,
            "memory_mb": 8192,
            "memory_mb_used": 512,
            "running_vms": 0,
            "service": {
                "host": "e6a37ee802d74863ab8b91ade8f12a67",
                "id": 2,
                "disabled_reason": null
            },
            "vcpus": 1,
            "vcpus_used": 0
        }
    ]
}`

// HypervisorListBody represents a raw hypervisor list result with Pike+ release.
const HypervisorListBody = `
{
    "hypervisors": [
        {
            "cpu_info": {
                "arch": "x86_64",
                "model": "Nehalem",
                "vendor": "Intel",
                "features": [
                    "pge",
                    "clflush"
                ],
                "topology": {
                    "cores": 1,
                    "threads": 1,
                    "sockets": 4
                }
            },
            "current_workload": 0,
            "status": "enabled",
            "state": "up",
            "disk_available_least": 0,
            "host_ip": "1.1.1.1",
            "free_disk_gb": 1028,
            "free_ram_mb": 7680,
            "hypervisor_hostname": "fake-mini",
            "hypervisor_type": "fake",
            "hypervisor_version": 2002000,
            "id": "c48f6247-abe4-4a24-824e-ea39e108874f",
            "local_gb": 1028,
            "local_gb_used": 0,
            "memory_mb": 8192,
            "memory_mb_used": 512,
            "running_vms": 0,
            "service": {
                "host": "e6a37ee802d74863ab8b91ade8f12a67",
                "id": "9c2566e7-7a54-4777-a1ae-c2662f0c407c",
                "disabled_reason": null
            },
            "vcpus": 1,
            "vcpus_used": 0
        },
        {
            "cpu_info": "{\"arch\": \"x86_64\", \"model\": \"Nehalem\", \"vendor\": \"Intel\", \"features\": [\"pge\", \"clflush\"], \"topology\": {\"cores\": 1, \"threads\": 1, \"sockets\": 4}}",
            "current_workload": 0,
            "status": "enabled",
            "state": "up",
            "disk_available_least": 0,
            "host_ip": "1.1.1.1",
            "free_disk_gb": 1028,
            "free_ram_mb": 7680,
            "hypervisor_hostname": "fake-mini",
            "hypervisor_type": "fake",
            "hypervisor_version": 2.002e+06,
            "id": "c48f6247-abe4-4a24-824e-ea39e108874f",
            "local_gb": 1028,
            "local_gb_used": 0,
            "memory_mb": 8192,
            "memory_mb_used": 512,
            "running_vms": 0,
            "service": {
                "host": "e6a37ee802d74863ab8b91ade8f12a67",
                "id": "9c2566e7-7a54-4777-a1ae-c2662f0c407c",
                "disabled_reason": null
            },
            "vcpus": 1,
            "vcpus_used": 0
        }
    ]
}`

const HypervisorsStatisticsBody = `
{
    "hypervisor_statistics": {
        "count": 1,
        "current_workload": 0,
        "disk_available_least": 0,
        "free_disk_gb": 1028,
        "free_ram_mb": 7680,
        "local_gb": 1028,
        "local_gb_used": 0,
        "memory_mb": 8192,
        "memory_mb_used": 512,
        "running_vms": 0,
        "vcpus": 2,
        "vcpus_used": 0
    }
}
`

// HypervisorGetBody represents a raw hypervisor GET result with Pike+ release.
const HypervisorGetBody = `
{
    "hypervisor":{
        "cpu_info":{
            "arch":"x86_64",
            "model":"Nehalem",
            "vendor":"Intel",
            "features":[
                "pge",
                "clflush"
            ],
            "topology":{
                "cores":1,
                "threads":1,
                "sockets":4
            }
        },
        "current_workload":0,
        "status":"enabled",
        "state":"up",
        "disk_available_least":0,
        "host_ip":"1.1.1.1",
        "free_disk_gb":1028,
        "free_ram_mb":7680,
        "hypervisor_hostname":"fake-mini",
        "hypervisor_type":"fake",
        "hypervisor_version":2002000,
        "id":"c48f6247-abe4-4a24-824e-ea39e108874f",
        "local_gb":1028,
        "local_gb_used":0,
        "memory_mb":8192,
        "memory_mb_used":512,
        "running_vms":0,
        "service":{
            "host":"e6a37ee802d74863ab8b91ade8f12a67",
            "id":"9c2566e7-7a54-4777-a1ae-c2662f0c407c",
            "disabled_reason":null
        },
        "vcpus":1,
        "vcpus_used":0
    }
}
`

// HypervisorGetEmptyCPUInfoBody represents a raw hypervisor GET result with
// no cpu_info
const HypervisorGetEmptyCPUInfoBody = `
{
    "hypervisor":{
        "cpu_info": "",
        "current_workload":0,
        "status":"enabled",
        "state":"up",
        "disk_available_least":0,
        "host_ip":"1.1.1.1",
        "free_disk_gb":1028,
        "free_ram_mb":7680,
        "hypervisor_hostname":"fake-mini",
        "hypervisor_type":"fake",
        "hypervisor_version":2002000,
        "id":"c48f6247-abe4-4a24-824e-ea39e108874f",
        "local_gb":1028,
        "local_gb_used":0,
        "memory_mb":8192,
        "memory_mb_used":512,
        "running_vms":0,
        "service":{
            "host":"e6a37ee802d74863ab8b91ade8f12a67",
            "id":"9c2566e7-7a54-4777-a1ae-c2662f0c407c",
            "disabled_reason":null
        },
        "vcpus":1,
        "vcpus_used":0
    }
}
`

// HypervisorUptimeBody represents a raw hypervisor uptime request result with
// Pike+ release.
const HypervisorUptimeBody = `
{
    "hypervisor": {
        "hypervisor_hostname": "fake-mini",
        "id": "c48f6247-abe4-4a24-824e-ea39e108874f",
        "state": "up",
        "status": "enabled",
        "uptime": " 08:32:11 up 93 days, 18:25, 12 users,  load average: 0.20, 0.12, 0.14"
    }
}
`

var (
	HypervisorFakePre253 = hypervisors.Hypervisor{
		CPUInfo: hypervisors.CPUInfo{
			Arch:   "x86_64",
			Model:  "Nehalem",
			Vendor: "Intel",
			Features: []string{
				"pge",
				"clflush",
			},
			Topology: hypervisors.Topology{
				Cores:   1,
				Threads: 1,
				Sockets: 4,
			},
		},
		CurrentWorkload:    0,
		Status:             "enabled",
		State:              "up",
		DiskAvailableLeast: 0,
		HostIP:             "1.1.1.1",
		FreeDiskGB:         1028,
		FreeRamMB:          7680,
		HypervisorHostname: "fake-mini",
		HypervisorType:     "fake",
		HypervisorVersion:  2002000,
		ID:                 "1",
		LocalGB:            1028,
		LocalGBUsed:        0,
		MemoryMB:           8192,
		MemoryMBUsed:       512,
		RunningVMs:         0,
		Service: hypervisors.Service{
			Host:           "e6a37ee802d74863ab8b91ade8f12a67",
			ID:             "2",
			DisabledReason: "",
		},
		VCPUs:     1,
		VCPUsUsed: 0,
	}

	HypervisorFake = hypervisors.Hypervisor{
		CPUInfo: hypervisors.CPUInfo{
			Arch:   "x86_64",
			Model:  "Nehalem",
			Vendor: "Intel",
			Features: []string{
				"pge",
				"clflush",
			},
			Topology: hypervisors.Topology{
				Cores:   1,
				Threads: 1,
				Sockets: 4,
			},
		},
		CurrentWorkload:    0,
		Status:             "enabled",
		State:              "up",
		DiskAvailableLeast: 0,
		HostIP:             "1.1.1.1",
		FreeDiskGB:         1028,
		FreeRamMB:          7680,
		HypervisorHostname: "fake-mini",
		HypervisorType:     "fake",
		HypervisorVersion:  2002000,
		ID:                 "c48f6247-abe4-4a24-824e-ea39e108874f",
		LocalGB:            1028,
		LocalGBUsed:        0,
		MemoryMB:           8192,
		MemoryMBUsed:       512,
		RunningVMs:         0,
		Service: hypervisors.Service{
			Host:           "e6a37ee802d74863ab8b91ade8f12a67",
			ID:             "9c2566e7-7a54-4777-a1ae-c2662f0c407c",
			DisabledReason: "",
		},
		VCPUs:     1,
		VCPUsUsed: 0,
	}

	HypervisorEmptyCPUInfo = hypervisors.Hypervisor{
		CurrentWorkload:    0,
		Status:             "enabled",
		State:              "up",
		DiskAvailableLeast: 0,
		HostIP:             "1.1.1.1",
		FreeDiskGB:         1028,
		FreeRamMB:          7680,
		HypervisorHostname: "fake-mini",
		HypervisorType:     "fake",
		HypervisorVersion:  2002000,
		ID:                 "c48f6247-abe4-4a24-824e-ea39e108874f",
		LocalGB:            1028,
		LocalGBUsed:        0,
		MemoryMB:           8192,
		MemoryMBUsed:       512,
		RunningVMs:         0,
		Service: hypervisors.Service{
			Host:           "e6a37ee802d74863ab8b91ade8f12a67",
			ID:             "9c2566e7-7a54-4777-a1ae-c2662f0c407c",
			DisabledReason: "",
		},
		VCPUs:     1,
		VCPUsUsed: 0,
	}

	HypervisorsStatisticsExpected = hypervisors.Statistics{
		Count:              1,
		CurrentWorkload:    0,
		DiskAvailableLeast: 0,
		FreeDiskGB:         1028,
		FreeRamMB:          7680,
		LocalGB:            1028,
		LocalGBUsed:        0,
		MemoryMB:           8192,
		MemoryMBUsed:       512,
		RunningVMs:         0,
		VCPUs:              2,
		VCPUsUsed:          0,
	}

	HypervisorUptimeExpected = hypervisors.Uptime{
		HypervisorHostname: "fake-mini",
		ID:                 "c48f6247-abe4-4a24-824e-ea39e108874f",
		State:              "up",
		Status:             "enabled",
		Uptime:             " 08:32:11 up 93 days, 18:25, 12 users,  load average: 0.20, 0.12, 0.14",
	}
)

func HandleHypervisorsStatisticsSuccessfully(t *testing.T) {
	testhelper.Mux.HandleFunc("/os-hypervisors/statistics", func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "GET")
		testhelper.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, HypervisorsStatisticsBody)
	})
}

func HandleHypervisorListPre253Successfully(t *testing.T) {
	testhelper.Mux.HandleFunc("/os-hypervisors/detail", func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "GET")
		testhelper.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, HypervisorListBodyPre253)
	})
}

func HandleHypervisorListSuccessfully(t *testing.T) {
	testhelper.Mux.HandleFunc("/os-hypervisors/detail", func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "GET")
		testhelper.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, HypervisorListBody)
	})
}

func HandleHypervisorGetSuccessfully(t *testing.T) {
	testhelper.Mux.HandleFunc("/os-hypervisors/"+HypervisorFake.ID, func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "GET")
		testhelper.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, HypervisorGetBody)
	})
}

func HandleHypervisorGetEmptyCPUInfoSuccessfully(t *testing.T) {
	testhelper.Mux.HandleFunc("/os-hypervisors/"+HypervisorFake.ID, func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "GET")
		testhelper.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, HypervisorGetEmptyCPUInfoBody)
	})
}

func HandleHypervisorUptimeSuccessfully(t *testing.T) {
	testhelper.Mux.HandleFunc("/os-hypervisors/"+HypervisorFake.ID+"/uptime", func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "GET")
		testhelper.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, HypervisorUptimeBody)
	})
}
