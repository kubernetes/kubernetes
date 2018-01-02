package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/tenantnetworks"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ListOutput is a sample response to a List call.
const ListOutput = `
{
    "networks": [
        {
            "cidr": "10.0.0.0/29",
            "id": "20c8acc0-f747-4d71-a389-46d078ebf047",
            "label": "mynet_0"
        },
        {
            "cidr": "10.0.0.10/29",
            "id": "20c8acc0-f747-4d71-a389-46d078ebf000",
            "label": "mynet_1"
        }
    ]
}
`

// GetOutput is a sample response to a Get call.
const GetOutput = `
{
    "network": {
			"cidr": "10.0.0.10/29",
			"id": "20c8acc0-f747-4d71-a389-46d078ebf000",
			"label": "mynet_1"
		}
}
`

// FirstNetwork is the first result in ListOutput.
var nilTime time.Time
var FirstNetwork = tenantnetworks.Network{
	CIDR: "10.0.0.0/29",
	ID:   "20c8acc0-f747-4d71-a389-46d078ebf047",
	Name: "mynet_0",
}

// SecondNetwork is the second result in ListOutput.
var SecondNetwork = tenantnetworks.Network{
	CIDR: "10.0.0.10/29",
	ID:   "20c8acc0-f747-4d71-a389-46d078ebf000",
	Name: "mynet_1",
}

// ExpectedNetworkSlice is the slice of results that should be parsed
// from ListOutput, in the expected order.
var ExpectedNetworkSlice = []tenantnetworks.Network{FirstNetwork, SecondNetwork}

// HandleListSuccessfully configures the test server to respond to a List request.
func HandleListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-tenant-networks", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, ListOutput)
	})
}

// HandleGetSuccessfully configures the test server to respond to a Get request
// for an existing network.
func HandleGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-tenant-networks/20c8acc0-f747-4d71-a389-46d078ebf000", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, GetOutput)
	})
}
