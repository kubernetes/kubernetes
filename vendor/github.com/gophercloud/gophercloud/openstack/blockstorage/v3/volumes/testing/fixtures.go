package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func MockListResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/detail", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
  {
  "volumes": [
    {
      "volume_type": "lvmdriver-1",
      "created_at": "2015-09-17T03:35:03.000000",
      "bootable": "false",
      "name": "vol-001",
      "os-vol-mig-status-attr:name_id": null,
      "consistencygroup_id": null,
      "source_volid": null,
      "os-volume-replication:driver_data": null,
      "multiattach": false,
      "snapshot_id": null,
      "replication_status": "disabled",
      "os-volume-replication:extended_status": null,
      "encrypted": false,
      "os-vol-host-attr:host": null,
      "availability_zone": "nova",
      "attachments": [{
        "server_id": "83ec2e3b-4321-422b-8706-a84185f52a0a",
        "attachment_id": "05551600-a936-4d4a-ba42-79a037c1-c91a",
        "attached_at": "2016-08-06T14:48:20.000000",
        "host_name": "foobar",
        "volume_id": "d6cacb1a-8b59-4c88-ad90-d70ebb82bb75",
        "device": "/dev/vdc",
        "id": "d6cacb1a-8b59-4c88-ad90-d70ebb82bb75"
      }],
      "id": "289da7f8-6440-407c-9fb4-7db01ec49164",
      "size": 75,
      "user_id": "ff1ce52c03ab433aaba9108c2e3ef541",
      "os-vol-tenant-attr:tenant_id": "304dc00909ac4d0da6c62d816bcb3459",
      "os-vol-mig-status-attr:migstat": null,
      "metadata": {"foo": "bar"},
      "status": "available",
      "description": null
    },
    {
      "volume_type": "lvmdriver-1",
      "created_at": "2015-09-17T03:32:29.000000",
      "bootable": "false",
      "name": "vol-002",
      "os-vol-mig-status-attr:name_id": null,
      "consistencygroup_id": null,
      "source_volid": null,
      "os-volume-replication:driver_data": null,
      "multiattach": false,
      "snapshot_id": null,
      "replication_status": "disabled",
      "os-volume-replication:extended_status": null,
      "encrypted": false,
      "os-vol-host-attr:host": null,
      "availability_zone": "nova",
      "attachments": [],
      "id": "96c3bda7-c82a-4f50-be73-ca7621794835",
      "size": 75,
      "user_id": "ff1ce52c03ab433aaba9108c2e3ef541",
      "os-vol-tenant-attr:tenant_id": "304dc00909ac4d0da6c62d816bcb3459",
      "os-vol-mig-status-attr:migstat": null,
      "metadata": {},
      "status": "available",
      "description": null
    }
  ]
}
  `)
	})
}

func MockGetResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
{
  "volume": {
    "volume_type": "lvmdriver-1",
    "created_at": "2015-09-17T03:32:29.000000",
    "bootable": "false",
    "name": "vol-001",
    "os-vol-mig-status-attr:name_id": null,
    "consistencygroup_id": null,
    "source_volid": null,
    "os-volume-replication:driver_data": null,
    "multiattach": false,
    "snapshot_id": null,
    "replication_status": "disabled",
    "os-volume-replication:extended_status": null,
    "encrypted": false,
    "os-vol-host-attr:host": null,
    "availability_zone": "nova",
    "attachments": [{
      "server_id": "83ec2e3b-4321-422b-8706-a84185f52a0a",
      "attachment_id": "05551600-a936-4d4a-ba42-79a037c1-c91a",
      "attached_at": "2016-08-06T14:48:20.000000",
      "host_name": "foobar",
      "volume_id": "d6cacb1a-8b59-4c88-ad90-d70ebb82bb75",
      "device": "/dev/vdc",
      "id": "d6cacb1a-8b59-4c88-ad90-d70ebb82bb75"
    }],
    "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
    "size": 75,
    "user_id": "ff1ce52c03ab433aaba9108c2e3ef541",
    "os-vol-tenant-attr:tenant_id": "304dc00909ac4d0da6c62d816bcb3459",
    "os-vol-mig-status-attr:migstat": null,
    "metadata": {},
    "status": "available",
    "description": null
  }
}
      `)
	})
}

func MockCreateResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "volume": {
    	"name": "vol-001",
        "size": 75
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)

		fmt.Fprintf(w, `
{
  "volume": {
    "size": 75,
    "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
    "metadata": {},
    "created_at": "2015-09-17T03:32:29.044216",
    "encrypted": false,
    "bootable": "false",
    "availability_zone": "nova",
    "attachments": [],
    "user_id": "ff1ce52c03ab433aaba9108c2e3ef541",
    "status": "creating",
    "description": null,
    "volume_type": "lvmdriver-1",
    "name": "vol-001",
    "replication_status": "disabled",
    "consistencygroup_id": null,
    "source_volid": null,
    "snapshot_id": null,
    "multiattach": false
  }
}
    `)
	})
}

func MockDeleteResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}

func MockUpdateResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
{
  "volume": {
    "name": "vol-002"
  }
}
        `)
	})
}
