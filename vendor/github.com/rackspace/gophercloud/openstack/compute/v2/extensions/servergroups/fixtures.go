// +build fixtures

package servergroups

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

// ListOutput is a sample response to a List call.
const ListOutput = `
{
    "server_groups": [
        {
            "id": "616fb98f-46ca-475e-917e-2563e5a8cd19",
            "name": "test",
            "policies": [
                "anti-affinity"
            ],
            "members": [],
            "metadata": {}
        },
        {
            "id": "4d8c3732-a248-40ed-bebc-539a6ffd25c0",
            "name": "test2",
            "policies": [
                "affinity"
            ],
            "members": [],
            "metadata": {}
        }
    ]
}
`

// GetOutput is a sample response to a Get call.
const GetOutput = `
{
    "server_group": {
        "id": "616fb98f-46ca-475e-917e-2563e5a8cd19",
        "name": "test",
        "policies": [
            "anti-affinity"
        ],
        "members": [],
        "metadata": {}
    }
}
`

// CreateOutput is a sample response to a Post call
const CreateOutput = `
{
    "server_group": {
        "id": "616fb98f-46ca-475e-917e-2563e5a8cd19",
        "name": "test",
        "policies": [
            "anti-affinity"
        ],
        "members": [],
        "metadata": {}
    }
}
`

// FirstServerGroup is the first result in ListOutput.
var FirstServerGroup = ServerGroup{
	ID:   "616fb98f-46ca-475e-917e-2563e5a8cd19",
	Name: "test",
	Policies: []string{
		"anti-affinity",
	},
	Members:  []string{},
	Metadata: map[string]interface{}{},
}

// SecondServerGroup is the second result in ListOutput.
var SecondServerGroup = ServerGroup{
	ID:   "4d8c3732-a248-40ed-bebc-539a6ffd25c0",
	Name: "test2",
	Policies: []string{
		"affinity",
	},
	Members:  []string{},
	Metadata: map[string]interface{}{},
}

// ExpectedServerGroupSlice is the slice of results that should be parsed
// from ListOutput, in the expected order.
var ExpectedServerGroupSlice = []ServerGroup{FirstServerGroup, SecondServerGroup}

// CreatedServerGroup is the parsed result from CreateOutput.
var CreatedServerGroup = ServerGroup{
	ID:   "616fb98f-46ca-475e-917e-2563e5a8cd19",
	Name: "test",
	Policies: []string{
		"anti-affinity",
	},
	Members:  []string{},
	Metadata: map[string]interface{}{},
}

// HandleListSuccessfully configures the test server to respond to a List request.
func HandleListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-server-groups", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, ListOutput)
	})
}

// HandleGetSuccessfully configures the test server to respond to a Get request
// for an existing server group
func HandleGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-server-groups/4d8c3732-a248-40ed-bebc-539a6ffd25c0", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleCreateSuccessfully configures the test server to respond to a Create request
// for a new server group
func HandleCreateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-server-groups", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `
{
    "server_group": {
        "name": "test",
        "policies": [
            "anti-affinity"
        ]
    }
}
`)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, CreateOutput)
	})
}

// HandleDeleteSuccessfully configures the test server to respond to a Delete request for a
// an existing server group
func HandleDeleteSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-server-groups/616fb98f-46ca-475e-917e-2563e5a8cd19", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusAccepted)
	})
}
