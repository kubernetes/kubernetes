package testing

import (
	"fmt"
	"math"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/blockstorage/extensions/schedulerstats"
	"github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

const StoragePoolsListBody = `
{
    "pools": [
        {
            "name": "rbd:cinder.volumes.ssd@cinder.volumes.ssd#cinder.volumes.ssd"
        },
        {
            "name": "rbd:cinder.volumes.hdd@cinder.volumes#cinder.volumes.hdd"
        }
    ]
}
`

const StoragePoolsListBodyDetail = `
{
    "pools": [
        {
            "capabilities": {
                "driver_version": "1.2.0",
                "filter_function": null,
                "free_capacity_gb": 64765,
                "goodness_function": null,
                "multiattach": false,
                "reserved_percentage": 0,
                "storage_protocol": "ceph",
                "timestamp": "2016-11-24T10:33:51.248360",
                "total_capacity_gb": 787947.93,
                "vendor_name": "Open Source",
                "volume_backend_name": "cinder.volumes.ssd"
            },
            "name": "rbd:cinder.volumes.ssd@cinder.volumes.ssd#cinder.volumes.ssd"
        },
        {
            "capabilities": {
                "driver_version": "1.2.0",
                "filter_function": null,
                "free_capacity_gb": "unknown",
                "goodness_function": null,
                "multiattach": false,
                "reserved_percentage": 0,
                "storage_protocol": "ceph",
                "timestamp": "2016-11-24T10:33:43.138628",
                "total_capacity_gb": "infinite",
                "vendor_name": "Open Source",
                "volume_backend_name": "cinder.volumes.hdd"
            },
            "name": "rbd:cinder.volumes.hdd@cinder.volumes.hdd#cinder.volumes.hdd"
        }
    ]
}
`

var (
	StoragePoolFake1 = schedulerstats.StoragePool{
		Name: "rbd:cinder.volumes.ssd@cinder.volumes.ssd#cinder.volumes.ssd",
		Capabilities: schedulerstats.Capabilities{
			DriverVersion:     "1.2.0",
			FreeCapacityGB:    64765,
			StorageProtocol:   "ceph",
			TotalCapacityGB:   787947.93,
			VendorName:        "Open Source",
			VolumeBackendName: "cinder.volumes.ssd",
		},
	}

	StoragePoolFake2 = schedulerstats.StoragePool{
		Name: "rbd:cinder.volumes.hdd@cinder.volumes.hdd#cinder.volumes.hdd",
		Capabilities: schedulerstats.Capabilities{
			DriverVersion:     "1.2.0",
			FreeCapacityGB:    0.0,
			StorageProtocol:   "ceph",
			TotalCapacityGB:   math.Inf(1),
			VendorName:        "Open Source",
			VolumeBackendName: "cinder.volumes.hdd",
		},
	}
)

func HandleStoragePoolsListSuccessfully(t *testing.T) {
	testhelper.Mux.HandleFunc("/scheduler-stats/get_pools", func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "GET")
		testhelper.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")

		r.ParseForm()
		if r.FormValue("detail") == "true" {
			fmt.Fprintf(w, StoragePoolsListBodyDetail)
		} else {
			fmt.Fprintf(w, StoragePoolsListBody)
		}
	})
}
