package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/policies"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

// ListOutput provides a single page of Policy results.
const ListOutput = `
{
    "links": {
        "next": null,
        "previous": null,
        "self": "http://example.com/identity/v3/policies"
    },
    "policies": [
        {
            "type": "text/plain",
            "id": "2844b2a08be147a08ef58317d6471f1f",
            "links": {
                "self": "http://example.com/identity/v3/policies/2844b2a08be147a08ef58317d6471f1f"
            },
            "blob": "'foo_user': 'role:compute-user'"
        },
        {
            "type": "application/json",
            "id": "b49884da9d31494ea02aff38d4b4e701",
            "links": {
                "self": "http://example.com/identity/v3/policies/b49884da9d31494ea02aff38d4b4e701"
            },
            "blob": "{'bar_user': 'role:network-user'}",
            "description": "policy for bar_user"
        }
    ]
}
`

// ListWithFilterOutput provides a single page of filtered Policy results.
const ListWithFilterOutput = `
{
    "links": {
        "next": null,
        "previous": null,
        "self": "http://example.com/identity/v3/policies"
    },
    "policies": [
        {
            "type": "application/json",
            "id": "b49884da9d31494ea02aff38d4b4e701",
            "links": {
                "self": "http://example.com/identity/v3/policies/b49884da9d31494ea02aff38d4b4e701"
            },
            "blob": "{'bar_user': 'role:network-user'}",
            "description": "policy for bar_user"
        }
    ]
}
`

// GetOutput provides a Get result.
const GetOutput = `
{
    "policy": {
        "type": "application/json",
        "id": "b49884da9d31494ea02aff38d4b4e701",
        "links": {
            "self": "http://example.com/identity/v3/policies/b49884da9d31494ea02aff38d4b4e701"
        },
        "blob": "{'bar_user': 'role:network-user'}",
        "description": "policy for bar_user"
    }
}
`

// CreateRequest provides the input to a Create request.
const CreateRequest = `
{
    "policy": {
        "blob": "{'bar_user': 'role:network-user'}",
        "description": "policy for bar_user",
        "type": "application/json"
    }
}
`

// UpdateRequest provides the input to as Update request.
const UpdateRequest = `
{
    "policy": {
        "description": "updated policy for bar_user"
    }
}
`

// UpdateOutput provides an update result.
const UpdateOutput = `
{
    "policy": {
        "type": "application/json",
        "id": "b49884da9d31494ea02aff38d4b4e701",
        "links": {
            "self": "http://example.com/identity/v3/policies/b49884da9d31494ea02aff38d4b4e701"
        },
        "blob": "{'bar_user': 'role:network-user'}",
        "description": "updated policy for bar_user"
    }
}
`

// FirstPolicy is the first policy in the List request.
var FirstPolicy = policies.Policy{
	ID:   "2844b2a08be147a08ef58317d6471f1f",
	Blob: "'foo_user': 'role:compute-user'",
	Type: "text/plain",
	Links: map[string]interface{}{
		"self": "http://example.com/identity/v3/policies/2844b2a08be147a08ef58317d6471f1f",
	},
	Extra: map[string]interface{}{},
}

// SecondPolicy is the second policy in the List request.
var SecondPolicy = policies.Policy{
	ID:   "b49884da9d31494ea02aff38d4b4e701",
	Blob: "{'bar_user': 'role:network-user'}",
	Type: "application/json",
	Links: map[string]interface{}{
		"self": "http://example.com/identity/v3/policies/b49884da9d31494ea02aff38d4b4e701",
	},
	Extra: map[string]interface{}{
		"description": "policy for bar_user",
	},
}

// SecondPolicyUpdated is the policy in the Update request.
var SecondPolicyUpdated = policies.Policy{
	ID:   "b49884da9d31494ea02aff38d4b4e701",
	Blob: "{'bar_user': 'role:network-user'}",
	Type: "application/json",
	Links: map[string]interface{}{
		"self": "http://example.com/identity/v3/policies/b49884da9d31494ea02aff38d4b4e701",
	},
	Extra: map[string]interface{}{
		"description": "updated policy for bar_user",
	},
}

// ExpectedPoliciesSlice is the slice of policies expected to be returned from ListOutput.
var ExpectedPoliciesSlice = []policies.Policy{FirstPolicy, SecondPolicy}

// HandleListPoliciesSuccessfully creates an HTTP handler at `/policies` on the
// test handler mux that responds with a list of two policies.
func HandleListPoliciesSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/policies", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		switch r.URL.Query().Get("type") {
		case "":
			fmt.Fprintf(w, ListOutput)
		case "application/json":
			fmt.Fprintf(w, ListWithFilterOutput)
		default:
			w.WriteHeader(http.StatusBadRequest)
		}
	})
}

// HandleCreatePolicySuccessfully creates an HTTP handler at `/policies` on the
// test handler mux that tests policy creation.
func HandleCreatePolicySuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/policies", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestJSONRequest(t, r, CreateRequest)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleGetPolicySuccessfully creates an HTTP handler at `/policies` on the
// test handler mux that responds with a single policy.
func HandleGetPolicySuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/policies/b49884da9d31494ea02aff38d4b4e701",
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "GET")
			th.TestHeader(t, r, "Accept", "application/json")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, GetOutput)
		},
	)
}

// HandleUpdatePolicySuccessfully creates an HTTP handler at `/policies` on the
// test handler mux that tests role update.
func HandleUpdatePolicySuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/policies/b49884da9d31494ea02aff38d4b4e701",
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "PATCH")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
			th.TestJSONRequest(t, r, UpdateRequest)

			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, UpdateOutput)
		},
	)
}

// HandleDeletePolicySuccessfully creates an HTTP handler at `/policies` on the
// test handler mux that tests policy deletion.
func HandleDeletePolicySuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/policies/9fe1d3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}
