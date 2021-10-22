package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const ListResponse = `
{
  "backups": [
    {
      "id": "289da7f8-6440-407c-9fb4-7db01ec49164",
      "name": "backup-001",
      "volume_id": "521752a6-acf6-4b2d-bc7a-119f9148cd8c",
      "description": "Daily Backup",
      "status": "available",
      "size": 30,
      "created_at": "2017-05-30T03:35:03.000000"
    },
    {
      "id": "96c3bda7-c82a-4f50-be73-ca7621794835",
      "name": "backup-002",
      "volume_id": "76b8950a-8594-4e5b-8dce-0dfa9c696358",
      "description": "Weekly Backup",
      "status": "available",
      "size": 25,
      "created_at": "2017-05-30T03:35:03.000000"
    }
  ],
  "backups_links": [
    {
      "href": "%s/backups?marker=1",
      "rel": "next"
    }
  ]
}
`

const GetResponse = `
{
  "backup": {
    "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
    "name": "backup-001",
    "description": "Daily backup",
    "volume_id": "521752a6-acf6-4b2d-bc7a-119f9148cd8c",
    "status": "available",
    "size": 30,
    "created_at": "2017-05-30T03:35:03.000000"
  }
}
`
const CreateRequest = `
{
  "backup": {
    "volume_id": "1234",
    "name": "backup-001"
  }
}
`

const CreateResponse = `
{
  "backup": {
    "volume_id": "1234",
    "name": "backup-001",
    "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
    "description": "Daily backup",
    "volume_id": "1234",
    "status": "available",
    "size": 30,
    "created_at": "2017-05-30T03:35:03.000000"
  }
}
`

func MockListResponse(t *testing.T) {
	th.Mux.HandleFunc("/backups", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, ListResponse, th.Server.URL)
		case "1":
			fmt.Fprintf(w, `{"backups": []}`)
		default:
			t.Fatalf("Unexpected marker: [%s]", marker)
		}
	})
}

func MockGetResponse(t *testing.T) {
	th.Mux.HandleFunc("/backups/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, GetResponse)
	})
}

func MockCreateResponse(t *testing.T) {
	th.Mux.HandleFunc("/backups", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, CreateRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)

		fmt.Fprintf(w, CreateResponse)
	})
}

func MockDeleteResponse(t *testing.T) {
	th.Mux.HandleFunc("/backups/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})
}
