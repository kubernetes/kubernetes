package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/baremetal/v1/allocations"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

const AllocationListBody = `
{
  "allocations": [
    {
      "candidate_nodes": [],
      "created_at": "2019-02-20T09:43:58+00:00",
      "extra": {},
      "last_error": null,
      "links": [
        {
          "href": "http://127.0.0.1:6385/v1/allocations/5344a3e2-978a-444e-990a-cbf47c62ef88",
          "rel": "self"
        },
        {
          "href": "http://127.0.0.1:6385/allocations/5344a3e2-978a-444e-990a-cbf47c62ef88",
          "rel": "bookmark"
        }
      ],
      "name": "allocation-1",
      "node_uuid": "6d85703a-565d-469a-96ce-30b6de53079d",
      "resource_class": "bm-large",
      "state": "active",
      "traits": [],
      "updated_at": "2019-02-20T09:43:58+00:00",
      "uuid": "5344a3e2-978a-444e-990a-cbf47c62ef88"
    },
    {
      "candidate_nodes": [],
      "created_at": "2019-02-20T09:43:58+00:00",
      "extra": {},
      "last_error": "Failed to process allocation eff80f47-75f0-4d41-b1aa-cf07c201adac: no available nodes match the resource class bm-large.",
      "links": [
        {
          "href": "http://127.0.0.1:6385/v1/allocations/eff80f47-75f0-4d41-b1aa-cf07c201adac",
          "rel": "self"
        },
        {
          "href": "http://127.0.0.1:6385/allocations/eff80f47-75f0-4d41-b1aa-cf07c201adac",
          "rel": "bookmark"
        }
      ],
      "name": "allocation-2",
      "node_uuid": null,
      "resource_class": "bm-large",
      "state": "error",
      "traits": [
        "CUSTOM_GOLD"
      ],
      "updated_at": "2019-02-20T09:43:58+00:00",
      "uuid": "eff80f47-75f0-4d41-b1aa-cf07c201adac"
    }
  ]
}
`

const SingleAllocationBody = `
{
  "candidate_nodes": ["344a3e2-978a-444e-990a-cbf47c62ef88"],
  "created_at": "2019-02-20T09:43:58+00:00",
  "extra": {},
  "last_error": null,
  "links": [
    {
      "href": "http://127.0.0.1:6385/v1/allocations/5344a3e2-978a-444e-990a-cbf47c62ef88",
      "rel": "self"
    },
    {
      "href": "http://127.0.0.1:6385/allocations/5344a3e2-978a-444e-990a-cbf47c62ef88",
      "rel": "bookmark"
    }
  ],
  "name": "allocation-1",
  "node_uuid": null,
  "resource_class": "baremetal",
  "state": "allocating",
  "traits": ["foo"],
  "updated_at": null,
  "uuid": "5344a3e2-978a-444e-990a-cbf47c62ef88"
}`

var (
	createdAt, _ = time.Parse(time.RFC3339, "2019-02-20T09:43:58+00:00")

	Allocation1 = allocations.Allocation{
		UUID:           "5344a3e2-978a-444e-990a-cbf47c62ef88",
		CandidateNodes: []string{"344a3e2-978a-444e-990a-cbf47c62ef88"},
		Name:           "allocation-1",
		State:          "allocating",
		ResourceClass:  "baremetal",
		Traits:         []string{"foo"},
		Extra:          map[string]string{},
		CreatedAt:      createdAt,
		Links:          []interface{}{map[string]interface{}{"href": "http://127.0.0.1:6385/v1/allocations/5344a3e2-978a-444e-990a-cbf47c62ef88", "rel": "self"}, map[string]interface{}{"href": "http://127.0.0.1:6385/allocations/5344a3e2-978a-444e-990a-cbf47c62ef88", "rel": "bookmark"}},
	}
)

// HandleAllocationListSuccessfully sets up the test server to respond to a allocation List request.
func HandleAllocationListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/allocations", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		w.Header().Add("Content-Type", "application/json")
		r.ParseForm()

		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, AllocationListBody)

		case "eff80f47-75f0-4d41-b1aa-cf07c201adac":
			fmt.Fprintf(w, `{ "allocations": [] }`)
		default:
			t.Fatalf("/allocations invoked with unexpected marker=[%s]", marker)
		}
	})
}

// HandleAllocationCreationSuccessfully sets up the test server to respond to a allocation creation request
// with a given response.
func HandleAllocationCreationSuccessfully(t *testing.T, response string) {
	th.Mux.HandleFunc("/allocations", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{
    		"name": "allocation-1",
    		"resource_class": "baremetal",
			"candidate_nodes": ["344a3e2-978a-444e-990a-cbf47c62ef88"],
		 	"traits": ["foo"]
        }`)

		w.WriteHeader(http.StatusAccepted)
		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, response)
	})
}

// HandleAllocationDeletionSuccessfully sets up the test server to respond to a allocation deletion request.
func HandleAllocationDeletionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/allocations/344a3e2-978a-444e-990a-cbf47c62ef88", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

func HandleAllocationGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/allocations/344a3e2-978a-444e-990a-cbf47c62ef88", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, SingleAllocationBody)
	})
}
