package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/usage"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

const FirstTenantID = "aabbccddeeff112233445566"
const SecondTenantID = "665544332211ffeeddccbbaa"

// GetSingleTenant holds the fixtures for the content of the request for a
// single tenant.
const GetSingleTenant = `{
    "tenant_usage": {
        "server_usages": [
            {
                "ended_at": null,
                "flavor": "m1.tiny",
                "hours": 0.021675453333333334,
                "instance_id": "a70096fd-8196-406b-86c4-045840f53ad7",
                "local_gb": 1,
                "memory_mb": 512,
                "name": "jttest",
                "started_at": "2017-11-30T03:23:43.000000",
                "state": "active",
                "tenant_id": "aabbccddeeff112233445566",
                "uptime": 78,
                "vcpus": 1
            },
            {
                "ended_at": "2017-11-21T04:10:11.000000",
                "flavor": "m1.acctest",
                "hours": 0.33444444444444443,
                "instance_id": "c04e38f2-dcee-4ca8-9466-7708d0a9b6dd",
                "local_gb": 15,
                "memory_mb": 512,
                "name": "basic",
                "started_at": "2017-11-21T03:50:07.000000",
                "state": "terminated",
                "tenant_id": "aabbccddeeff112233445566",
                "uptime": 1204,
                "vcpus": 1
            },
            {
                "ended_at": "2017-11-30T03:21:21.000000",
                "flavor": "m1.acctest",
                "hours": 0.004166666666666667,
                "instance_id": "ceb654fa-e0e8-44fb-8942-e4d0bfad3941",
                "local_gb": 15,
                "memory_mb": 512,
                "name": "ACPTTESTJSxbPQAC34lTnBE1",
                "started_at": "2017-11-30T03:21:06.000000",
                "state": "terminated",
                "tenant_id": "aabbccddeeff112233445566",
                "uptime": 15,
                "vcpus": 1
            }
        ],
        "start": "2017-11-02T03:25:01.000000",
        "stop": "2017-11-30T03:25:01.000000",
        "tenant_id": "aabbccddeeff112233445566",
        "total_hours": 1.25834212,
        "total_local_gb_usage": 18.571675453333334,
        "total_memory_mb_usage": 644.27116544,
        "total_vcpus_usage": 1.25834212
    }
}`

// HandleGetSingleTenantSuccessfully configures the test server to respond to a
// Get request for a single tenant
func HandleGetSingleTenantSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-simple-tenant-usage/"+FirstTenantID, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.Header().Add("Content-Type", "application/json")
		fmt.Fprint(w, GetSingleTenant)
	})
}

// SingleTenantUsageResults is the code fixture for GetSingleTenant.
var SingleTenantUsageResults = usage.TenantUsage{
	ServerUsages: []usage.ServerUsage{
		{
			Flavor:     "m1.tiny",
			Hours:      0.021675453333333334,
			InstanceID: "a70096fd-8196-406b-86c4-045840f53ad7",
			LocalGB:    1,
			MemoryMB:   512,
			Name:       "jttest",
			StartedAt:  time.Date(2017, 11, 30, 3, 23, 43, 0, time.UTC),
			State:      "active",
			TenantID:   "aabbccddeeff112233445566",
			Uptime:     78,
			VCPUs:      1,
		},
		{
			Flavor:     "m1.acctest",
			Hours:      0.33444444444444443,
			InstanceID: "c04e38f2-dcee-4ca8-9466-7708d0a9b6dd",
			LocalGB:    15,
			MemoryMB:   512,
			Name:       "basic",
			StartedAt:  time.Date(2017, 11, 21, 3, 50, 7, 0, time.UTC),
			EndedAt:    time.Date(2017, 11, 21, 4, 10, 11, 0, time.UTC),
			State:      "terminated",
			TenantID:   "aabbccddeeff112233445566",
			Uptime:     1204,
			VCPUs:      1,
		},
		{
			Flavor:     "m1.acctest",
			Hours:      0.004166666666666667,
			InstanceID: "ceb654fa-e0e8-44fb-8942-e4d0bfad3941",
			LocalGB:    15,
			MemoryMB:   512,
			Name:       "ACPTTESTJSxbPQAC34lTnBE1",
			StartedAt:  time.Date(2017, 11, 30, 3, 21, 6, 0, time.UTC),
			EndedAt:    time.Date(2017, 11, 30, 3, 21, 21, 0, time.UTC),
			State:      "terminated",
			TenantID:   "aabbccddeeff112233445566",
			Uptime:     15,
			VCPUs:      1,
		},
	},
	Start:              time.Date(2017, 11, 2, 3, 25, 1, 0, time.UTC),
	Stop:               time.Date(2017, 11, 30, 3, 25, 1, 0, time.UTC),
	TenantID:           "aabbccddeeff112233445566",
	TotalHours:         1.25834212,
	TotalLocalGBUsage:  18.571675453333334,
	TotalMemoryMBUsage: 644.27116544,
	TotalVCPUsUsage:    1.25834212,
}

// GetAllTenants holds the fixtures for the content of the request for
// all tenants.
const GetAllTenants = `{
    "tenant_usages": [
				{
					"server_usages": [
							{
									"ended_at": null,
									"flavor": "m1.tiny",
									"hours": 0.021675453333333334,
									"instance_id": "a70096fd-8196-406b-86c4-045840f53ad7",
									"local_gb": 1,
									"memory_mb": 512,
									"name": "jttest",
									"started_at": "2017-11-30T03:23:43.000000",
									"state": "active",
									"tenant_id": "aabbccddeeff112233445566",
									"uptime": 78,
									"vcpus": 1
							},
							{
									"ended_at": "2017-11-21T04:10:11.000000",
									"flavor": "m1.acctest",
									"hours": 0.33444444444444443,
									"instance_id": "c04e38f2-dcee-4ca8-9466-7708d0a9b6dd",
									"local_gb": 15,
									"memory_mb": 512,
									"name": "basic",
									"started_at": "2017-11-21T03:50:07.000000",
									"state": "terminated",
									"tenant_id": "aabbccddeeff112233445566",
									"uptime": 1204,
									"vcpus": 1
							},
							{
									"ended_at": "2017-11-30T03:21:21.000000",
									"flavor": "m1.acctest",
									"hours": 0.004166666666666667,
									"instance_id": "ceb654fa-e0e8-44fb-8942-e4d0bfad3941",
									"local_gb": 15,
									"memory_mb": 512,
									"name": "ACPTTESTJSxbPQAC34lTnBE1",
									"started_at": "2017-11-30T03:21:06.000000",
									"state": "terminated",
									"tenant_id": "aabbccddeeff112233445566",
									"uptime": 15,
									"vcpus": 1
							}
					],
					"start": "2017-11-02T03:25:01.000000",
					"stop": "2017-11-30T03:25:01.000000",
					"tenant_id": "aabbccddeeff112233445566",
					"total_hours": 1.25834212,
					"total_local_gb_usage": 18.571675453333334,
					"total_memory_mb_usage": 644.27116544,
					"total_vcpus_usage": 1.25834212
				},
				{
					"server_usages": [
							{
									"ended_at": null,
									"flavor": "m1.tiny",
									"hours": 0.021675453333333334,
									"instance_id": "a70096fd-8196-406b-86c4-045840f53ad7",
									"local_gb": 1,
									"memory_mb": 512,
									"name": "test",
									"started_at": "2017-11-30T03:23:43.000000",
									"state": "active",
									"tenant_id": "665544332211ffeeddccbbaa",
									"uptime": 78,
									"vcpus": 1
							}
					],
					"start": "2017-11-02T03:25:01.000000",
					"stop": "2017-11-30T03:25:01.000000",
					"tenant_id": "665544332211ffeeddccbbaa",
					"total_hours": 0.021675453333333334,
					"total_local_gb_usage": 18.571675453333334,
					"total_memory_mb_usage": 644.27116544,
					"total_vcpus_usage": 1.25834212
				}
		]
}`

// HandleGetAllTenantsSuccessfully configures the test server to respond to a
// Get request for all tenants.
func HandleGetAllTenantsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-simple-tenant-usage", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.Header().Add("Content-Type", "application/json")
		fmt.Fprint(w, GetAllTenants)
	})
}

// AllTenantsUsageResult is the code fixture for GetAllTenants.
var AllTenantsUsageResult = []usage.TenantUsage{
	{
		ServerUsages: []usage.ServerUsage{
			{
				Flavor:     "m1.tiny",
				Hours:      0.021675453333333334,
				InstanceID: "a70096fd-8196-406b-86c4-045840f53ad7",
				LocalGB:    1,
				MemoryMB:   512,
				Name:       "jttest",
				StartedAt:  time.Date(2017, 11, 30, 3, 23, 43, 0, time.UTC),
				State:      "active",
				TenantID:   "aabbccddeeff112233445566",
				Uptime:     78,
				VCPUs:      1,
			},
			{
				Flavor:     "m1.acctest",
				Hours:      0.33444444444444443,
				InstanceID: "c04e38f2-dcee-4ca8-9466-7708d0a9b6dd",
				LocalGB:    15,
				MemoryMB:   512,
				Name:       "basic",
				StartedAt:  time.Date(2017, 11, 21, 3, 50, 7, 0, time.UTC),
				EndedAt:    time.Date(2017, 11, 21, 4, 10, 11, 0, time.UTC),
				State:      "terminated",
				TenantID:   "aabbccddeeff112233445566",
				Uptime:     1204,
				VCPUs:      1,
			},
			{
				Flavor:     "m1.acctest",
				Hours:      0.004166666666666667,
				InstanceID: "ceb654fa-e0e8-44fb-8942-e4d0bfad3941",
				LocalGB:    15,
				MemoryMB:   512,
				Name:       "ACPTTESTJSxbPQAC34lTnBE1",
				StartedAt:  time.Date(2017, 11, 30, 3, 21, 6, 0, time.UTC),
				EndedAt:    time.Date(2017, 11, 30, 3, 21, 21, 0, time.UTC),
				State:      "terminated",
				TenantID:   "aabbccddeeff112233445566",
				Uptime:     15,
				VCPUs:      1,
			},
		},
		Start:              time.Date(2017, 11, 2, 3, 25, 1, 0, time.UTC),
		Stop:               time.Date(2017, 11, 30, 3, 25, 1, 0, time.UTC),
		TenantID:           "aabbccddeeff112233445566",
		TotalHours:         1.25834212,
		TotalLocalGBUsage:  18.571675453333334,
		TotalMemoryMBUsage: 644.27116544,
		TotalVCPUsUsage:    1.25834212,
	},
	{
		ServerUsages: []usage.ServerUsage{
			{
				Flavor:     "m1.tiny",
				Hours:      0.021675453333333334,
				InstanceID: "a70096fd-8196-406b-86c4-045840f53ad7",
				LocalGB:    1,
				MemoryMB:   512,
				Name:       "test",
				StartedAt:  time.Date(2017, 11, 30, 3, 23, 43, 0, time.UTC),
				State:      "active",
				TenantID:   "665544332211ffeeddccbbaa",
				Uptime:     78,
				VCPUs:      1,
			},
		},
		Start:              time.Date(2017, 11, 2, 3, 25, 1, 0, time.UTC),
		Stop:               time.Date(2017, 11, 30, 3, 25, 1, 0, time.UTC),
		TenantID:           "665544332211ffeeddccbbaa",
		TotalHours:         0.021675453333333334,
		TotalLocalGBUsage:  18.571675453333334,
		TotalMemoryMBUsage: 644.27116544,
		TotalVCPUsUsage:    1.25834212,
	},
}
