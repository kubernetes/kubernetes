// +build fixtures

package volumetypes

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func MockListResponse(t *testing.T) {
	th.Mux.HandleFunc("/types", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
    {
      "volume_types": [
        {
          "id": "289da7f8-6440-407c-9fb4-7db01ec49164",
          "name": "vol-type-001",
          "extra_specs": {
            "capabilities": "gpu"
            }
        },
        {
          "id": "96c3bda7-c82a-4f50-be73-ca7621794835",
          "name": "vol-type-002",
          "extra_specs": {}
        }
      ]
    }
    `)
	})
}

func MockGetResponse(t *testing.T) {
	th.Mux.HandleFunc("/types/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
{
    "volume_type": {
        "name": "vol-type-001",
        "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
    "extra_specs": {
      "serverNumber": "2"
    }
    }
}
      `)
	})
}
