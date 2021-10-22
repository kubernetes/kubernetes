package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func MockAttachResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/cd281d77-8217-4830-be95-9528227c105c/action",
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "POST")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
			th.TestHeader(t, r, "Content-Type", "application/json")
			th.TestHeader(t, r, "Accept", "application/json")
			th.TestJSONRequest(t, r, `
{
    "os-attach":
    {
        "mountpoint": "/mnt",
        "mode": "rw",
        "instance_uuid": "50902f4f-a974-46a0-85e9-7efc5e22dfdd"
    }
}
          `)

			w.Header().Add("Content-Type", "application/json")
			w.WriteHeader(http.StatusAccepted)

			fmt.Fprintf(w, `{}`)
		})
}

func MockBeginDetachingResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/cd281d77-8217-4830-be95-9528227c105c/action",
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "POST")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
			th.TestHeader(t, r, "Content-Type", "application/json")
			th.TestHeader(t, r, "Accept", "application/json")
			th.TestJSONRequest(t, r, `
{
    "os-begin_detaching": {}
}
          `)

			w.Header().Add("Content-Type", "application/json")
			w.WriteHeader(http.StatusAccepted)

			fmt.Fprintf(w, `{}`)
		})
}

func MockDetachResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/cd281d77-8217-4830-be95-9528227c105c/action",
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "POST")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
			th.TestHeader(t, r, "Content-Type", "application/json")
			th.TestHeader(t, r, "Accept", "application/json")
			th.TestJSONRequest(t, r, `
{
    "os-detach": {}
}
          `)

			w.Header().Add("Content-Type", "application/json")
			w.WriteHeader(http.StatusAccepted)

			fmt.Fprintf(w, `{}`)
		})
}

func MockUploadImageResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/cd281d77-8217-4830-be95-9528227c105c/action",
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "POST")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
			th.TestHeader(t, r, "Content-Type", "application/json")
			th.TestHeader(t, r, "Accept", "application/json")
			th.TestJSONRequest(t, r, `
{
    "os-volume_upload_image": {
        "container_format": "bare",
        "force": true,
        "image_name": "test",
        "disk_format": "raw"
    }
}
          `)

			w.Header().Add("Content-Type", "application/json")
			w.WriteHeader(http.StatusAccepted)

			fmt.Fprintf(w, `
{
    "os-volume_upload_image": {
        "container_format": "bare",
        "display_description": null,
        "id": "cd281d77-8217-4830-be95-9528227c105c",
        "image_id": "ecb92d98-de08-45db-8235-bbafe317269c",
        "image_name": "test",
        "disk_format": "raw",
        "size": 5,
        "status": "uploading",
        "updated_at": "2017-07-17T09:29:22.000000",
        "volume_type": {
            "created_at": "2016-05-04T08:54:14.000000",
            "deleted": false,
            "deleted_at": null,
            "description": null,
            "extra_specs": {
                "volume_backend_name": "basic.ru-2a"
            },
            "id": "b7133444-62f6-4433-8da3-70ac332229b7",
            "is_public": true,
            "name": "basic.ru-2a",
            "updated_at": "2016-05-04T09:15:33.000000"
        }
    }
}
          `)
		})
}

func MockReserveResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/cd281d77-8217-4830-be95-9528227c105c/action",
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "POST")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
			th.TestHeader(t, r, "Content-Type", "application/json")
			th.TestHeader(t, r, "Accept", "application/json")
			th.TestJSONRequest(t, r, `
{
    "os-reserve": {}
}
          `)

			w.Header().Add("Content-Type", "application/json")
			w.WriteHeader(http.StatusAccepted)

			fmt.Fprintf(w, `{}`)
		})
}

func MockUnreserveResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/cd281d77-8217-4830-be95-9528227c105c/action",
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "POST")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
			th.TestHeader(t, r, "Content-Type", "application/json")
			th.TestHeader(t, r, "Accept", "application/json")
			th.TestJSONRequest(t, r, `
{
    "os-unreserve": {}
}
          `)

			w.Header().Add("Content-Type", "application/json")
			w.WriteHeader(http.StatusAccepted)

			fmt.Fprintf(w, `{}`)
		})
}

func MockInitializeConnectionResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/cd281d77-8217-4830-be95-9528227c105c/action",
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "POST")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
			th.TestHeader(t, r, "Content-Type", "application/json")
			th.TestHeader(t, r, "Accept", "application/json")
			th.TestJSONRequest(t, r, `
{
    "os-initialize_connection":
    {
        "connector":
        {
        "ip":"127.0.0.1",
        "host":"stack",
        "initiator":"iqn.1994-05.com.redhat:17cf566367d2",
        "multipath": false,
        "platform": "x86_64",
        "os_type": "linux2"
        }
    }
}
          `)

			w.Header().Add("Content-Type", "application/json")
			w.WriteHeader(http.StatusAccepted)

			fmt.Fprintf(w, `{
"connection_info": {
    "data": {
      "target_portals": [
        "172.31.17.48:3260"
      ],
      "auth_method": "CHAP",
      "auth_username": "5MLtcsTEmNN5jFVcT6ui",
      "access_mode": "rw",
      "target_lun": 0,
      "volume_id": "cd281d77-8217-4830-be95-9528227c105c",
      "target_luns": [
        0
      ],
      "target_iqns": [
        "iqn.2010-10.org.openstack:volume-cd281d77-8217-4830-be95-9528227c105c"
      ],
      "auth_password": "x854ZY5Re3aCkdNL",
      "target_discovered": false,
      "encrypted": false,
      "qos_specs": null,
      "target_iqn": "iqn.2010-10.org.openstack:volume-cd281d77-8217-4830-be95-9528227c105c",
      "target_portal": "172.31.17.48:3260"
    },
    "driver_volume_type": "iscsi"
  }
            }`)
		})
}

func MockTerminateConnectionResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/cd281d77-8217-4830-be95-9528227c105c/action",
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "POST")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
			th.TestHeader(t, r, "Content-Type", "application/json")
			th.TestHeader(t, r, "Accept", "application/json")
			th.TestJSONRequest(t, r, `
{
    "os-terminate_connection":
    {
        "connector":
        {
        "ip":"127.0.0.1",
        "host":"stack",
        "initiator":"iqn.1994-05.com.redhat:17cf566367d2",
        "multipath": true,
        "platform": "x86_64",
        "os_type": "linux2"
        }
    }
}
          `)

			w.Header().Add("Content-Type", "application/json")
			w.WriteHeader(http.StatusAccepted)

			fmt.Fprintf(w, `{}`)
		})
}

func MockExtendSizeResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/cd281d77-8217-4830-be95-9528227c105c/action",
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "POST")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
			th.TestHeader(t, r, "Content-Type", "application/json")
			th.TestHeader(t, r, "Accept", "application/json")
			th.TestJSONRequest(t, r, `
{
    "os-extend":
    {
        "new_size": 3
    }
}
          `)

			w.Header().Add("Content-Type", "application/json")
			w.WriteHeader(http.StatusAccepted)

			fmt.Fprintf(w, `{}`)
		})
}

func MockForceDeleteResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/d32019d3-bc6e-4319-9c1d-6722fc136a22/action", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestBody(t, r, `{"os-force_delete":""}`)
		w.WriteHeader(http.StatusAccepted)
	})
}
