// +build fixtures

package testing

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
  "volumeAttachments": [
    {
      "device": "/dev/vdd",
      "id": "a26887c6-c47b-4654-abb5-dfadf7d3f803",
      "serverId": "4d8c3732-a248-40ed-bebc-539a6ffd25c0",
      "volumeId": "a26887c6-c47b-4654-abb5-dfadf7d3f803"
    },
    {
      "device": "/dev/vdc",
      "id": "a26887c6-c47b-4654-abb5-dfadf7d3f804",
      "serverId": "4d8c3732-a248-40ed-bebc-539a6ffd25c0",
      "volumeId": "a26887c6-c47b-4654-abb5-dfadf7d3f804"
    }
  ]
}
`

// GetOutput is a sample response to a Get call.
const GetOutput = `
{
  "volumeAttachment": {
    "device": "/dev/vdc",
    "id": "a26887c6-c47b-4654-abb5-dfadf7d3f804",
    "serverId": "4d8c3732-a248-40ed-bebc-539a6ffd25c0",
    "volumeId": "a26887c6-c47b-4654-abb5-dfadf7d3f804"
  }
}
`

// CreateOutput is a sample response to a Create call.
const CreateOutput = `
{
  "volumeAttachment": {
    "device": "/dev/vdc",
    "id": "a26887c6-c47b-4654-abb5-dfadf7d3f804",
    "serverId": "4d8c3732-a248-40ed-bebc-539a6ffd25c0",
    "volumeId": "a26887c6-c47b-4654-abb5-dfadf7d3f804"
  }
}
`

// HandleListSuccessfully configures the test server to respond to a List request.
func HandleListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/servers/4d8c3732-a248-40ed-bebc-539a6ffd25c0/os-volume_attachments", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, ListOutput)
	})
}

// HandleGetSuccessfully configures the test server to respond to a Get request
// for an existing attachment
func HandleGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/servers/4d8c3732-a248-40ed-bebc-539a6ffd25c0/os-volume_attachments/a26887c6-c47b-4654-abb5-dfadf7d3f804", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleCreateSuccessfully configures the test server to respond to a Create request
// for a new attachment
func HandleCreateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/servers/4d8c3732-a248-40ed-bebc-539a6ffd25c0/os-volume_attachments", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `
{
  "volumeAttachment": {
    "volumeId": "a26887c6-c47b-4654-abb5-dfadf7d3f804",
    "device": "/dev/vdc"
  }
}
`)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, CreateOutput)
	})
}

// HandleDeleteSuccessfully configures the test server to respond to a Delete request for a
// an existing attachment
func HandleDeleteSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/servers/4d8c3732-a248-40ed-bebc-539a6ffd25c0/os-volume_attachments/a26887c6-c47b-4654-abb5-dfadf7d3f804", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusAccepted)
	})
}
