// +build fixtures

package floatingip

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
    "floating_ips": [
        {
            "fixed_ip": null,
            "id": 1,
            "instance_id": null,
            "ip": "10.10.10.1",
            "pool": "nova"
        },
        {
            "fixed_ip": "166.78.185.201",
            "id": 2,
            "instance_id": "4d8c3732-a248-40ed-bebc-539a6ffd25c0",
            "ip": "10.10.10.2",
            "pool": "nova"
        }
    ]
}
`

// GetOutput is a sample response to a Get call.
const GetOutput = `
{
    "floating_ip": {
        "fixed_ip": "166.78.185.201",
        "id": 2,
        "instance_id": "4d8c3732-a248-40ed-bebc-539a6ffd25c0",
        "ip": "10.10.10.2",
        "pool": "nova"
    }
}
`

// CreateOutput is a sample response to a Post call
const CreateOutput = `
{
    "floating_ip": {
        "fixed_ip": null,
        "id": 1,
        "instance_id": null,
        "ip": "10.10.10.1",
        "pool": "nova"
    }
}
`

// FirstFloatingIP is the first result in ListOutput.
var FirstFloatingIP = FloatingIP{
	ID:   "1",
	IP:   "10.10.10.1",
	Pool: "nova",
}

// SecondFloatingIP is the first result in ListOutput.
var SecondFloatingIP = FloatingIP{
	FixedIP:    "166.78.185.201",
	ID:         "2",
	InstanceID: "4d8c3732-a248-40ed-bebc-539a6ffd25c0",
	IP:         "10.10.10.2",
	Pool:       "nova",
}

// ExpectedFloatingIPsSlice is the slice of results that should be parsed
// from ListOutput, in the expected order.
var ExpectedFloatingIPsSlice = []FloatingIP{FirstFloatingIP, SecondFloatingIP}

// CreatedFloatingIP is the parsed result from CreateOutput.
var CreatedFloatingIP = FloatingIP{
	ID:   "1",
	IP:   "10.10.10.1",
	Pool: "nova",
}

// HandleListSuccessfully configures the test server to respond to a List request.
func HandleListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-floating-ips", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, ListOutput)
	})
}

// HandleGetSuccessfully configures the test server to respond to a Get request
// for an existing floating ip
func HandleGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-floating-ips/2", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleCreateSuccessfully configures the test server to respond to a Create request
// for a new floating ip
func HandleCreateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-floating-ips", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `
{
	"pool": "nova"
}
`)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, CreateOutput)
	})
}

// HandleDeleteSuccessfully configures the test server to respond to a Delete request for a
// an existing floating ip
func HandleDeleteSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-floating-ips/1", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusAccepted)
	})
}

// HandleAssociateSuccessfully configures the test server to respond to a Post request
// to associate an allocated floating IP
func HandleAssociateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/servers/4d8c3732-a248-40ed-bebc-539a6ffd25c0/action", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `
{
	"addFloatingIp": {
		"address": "10.10.10.2"
	}
}
`)

		w.WriteHeader(http.StatusAccepted)
	})
}

// HandleFixedAssociateSucessfully configures the test server to respond to a Post request
// to associate an allocated floating IP with a specific fixed IP address
func HandleAssociateFixedSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/servers/4d8c3732-a248-40ed-bebc-539a6ffd25c0/action", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `
{
	"addFloatingIp": {
		"address": "10.10.10.2",
		"fixed_address": "166.78.185.201"
	}
}
`)

		w.WriteHeader(http.StatusAccepted)
	})
}

// HandleDisassociateSuccessfully configures the test server to respond to a Post request
// to disassociate an allocated floating IP
func HandleDisassociateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/servers/4d8c3732-a248-40ed-bebc-539a6ffd25c0/action", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `
{
	"removeFloatingIp": {
		"address": "10.10.10.2"
	}
}
`)

		w.WriteHeader(http.StatusAccepted)
	})
}
