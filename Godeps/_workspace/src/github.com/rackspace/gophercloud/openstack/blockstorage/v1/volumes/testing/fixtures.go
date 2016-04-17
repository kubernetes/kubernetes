// +build fixtures

package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
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
        "display_name": "vol-001",
        "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
 	"attachments": [
	  {
            "device": "/dev/vde",
            "server_id": "a740d24b-dc5b-4d59-ac75-53971c2920ba",
            "id": "d6da11e5-2ed3-413e-88d8-b772ba62193d",
            "volume_id": "d6da11e5-2ed3-413e-88d8-b772ba62193d"
          }
        ]
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
