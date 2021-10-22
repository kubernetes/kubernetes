package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func MockListResponse(t *testing.T) {
	th.Mux.HandleFunc("/snapshots", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, `
    {
      "snapshots": [
        {
          "id": "289da7f8-6440-407c-9fb4-7db01ec49164",
          "name": "snapshot-001",
          "volume_id": "521752a6-acf6-4b2d-bc7a-119f9148cd8c",
          "description": "Daily Backup",
          "status": "available",
          "size": 30,
		  "created_at": "2017-05-30T03:35:03.000000"
        },
        {
          "id": "96c3bda7-c82a-4f50-be73-ca7621794835",
          "name": "snapshot-002",
          "volume_id": "76b8950a-8594-4e5b-8dce-0dfa9c696358",
          "description": "Weekly Backup",
          "status": "available",
          "size": 25,
		  "created_at": "2017-05-30T03:35:03.000000"
        }
      ],
      "snapshots_links": [
        {
            "href": "%s/snapshots?marker=1",
            "rel": "next"
        }]
    }
    `, th.Server.URL)
		case "1":
			fmt.Fprintf(w, `{"snapshots": []}`)
		default:
			t.Fatalf("Unexpected marker: [%s]", marker)
		}
	})
}

func MockGetResponse(t *testing.T) {
	th.Mux.HandleFunc("/snapshots/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
{
    "snapshot": {
        "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
        "name": "snapshot-001",
        "description": "Daily backup",
        "volume_id": "521752a6-acf6-4b2d-bc7a-119f9148cd8c",
        "status": "available",
        "size": 30,
		"created_at": "2017-05-30T03:35:03.000000"
    }
}
      `)
	})
}

func MockCreateResponse(t *testing.T) {
	th.Mux.HandleFunc("/snapshots", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "snapshot": {
        "volume_id": "1234",
        "name": "snapshot-001"
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)

		fmt.Fprintf(w, `
{
    "snapshot": {
        "volume_id": "1234",
        "name": "snapshot-001",
        "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
        "description": "Daily backup",
        "volume_id": "1234",
        "status": "available",
        "size": 30,
		"created_at": "2017-05-30T03:35:03.000000"
  }
}
    `)
	})
}

func MockUpdateMetadataResponse(t *testing.T) {
	th.Mux.HandleFunc("/snapshots/123/metadata", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestJSONRequest(t, r, `
    {
      "metadata": {
        "key": "v1"
      }
    }
    `)

		fmt.Fprintf(w, `
      {
        "metadata": {
          "key": "v1"
        }
      }
    `)
	})
}

func MockDeleteResponse(t *testing.T) {
	th.Mux.HandleFunc("/snapshots/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})
}
