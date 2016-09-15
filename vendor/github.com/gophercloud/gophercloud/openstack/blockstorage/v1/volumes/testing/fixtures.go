package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func MockListResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
  {
    "volumes": [
      {
        "id": "289da7f8-6440-407c-9fb4-7db01ec49164",
        "display_name": "vol-001"
      },
      {
        "id": "96c3bda7-c82a-4f50-be73-ca7621794835",
        "display_name": "vol-002"
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
			        "id": "521752a6-acf6-4b2d-bc7a-119f9148cd8c",
			        "display_name": "vol-001",
			        "display_description": "Another volume.",
			        "status": "active",
			        "size": 30,
			        "volume_type": "289da7f8-6440-407c-9fb4-7db01ec49164",
			        "metadata": {
			            "contents": "junk"
			        },
			        "availability_zone": "us-east1",
			        "bootable": "false",
			        "snapshot_id": null,
			        "attachments": [
			            {
			                "attachment_id": "03987cd1-0ad5-40d1-9b2a-7cc48295d4fa",
			                "id": "47e9ecc5-4045-4ee3-9a4b-d859d546a0cf",
			                "volume_id": "6c80f8ac-e3e2-480c-8e6e-f1db92fe4bfe",
			                "server_id": "d1c4788b-9435-42e2-9b81-29f3be1cd01f",
			                "host_name": "mitaka",
			                "device": "/"
			            }
			        ],
			        "created_at": "2012-02-14T20:53:07"
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
        "size": 75
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "volume": {
        "size": 4,
        "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22"
    }
}
    `)
	})
}

func MockDeleteResponse(t *testing.T) {
	th.Mux.HandleFunc("/volumes/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
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
        "display_name": "vol-002",
        "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22"
        }
    }
    `)
	})
}
