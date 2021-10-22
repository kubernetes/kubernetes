package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/keymanager/v1/orders"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ListResponse provides a single page of RESOURCE results.
const ListResponse = `
{
  "orders": [
    {
      "created": "2018-06-22T05:05:43",
      "creator_id": "5c70d99f4a8641c38f8084b32b5e5c0e",
      "meta": {
        "algorithm": "aes",
        "bit_length": 256,
        "expiration": null,
        "mode": "cbc",
        "name": null,
        "payload_content_type": "application/octet-stream"
      },
      "order_ref": "http://barbican:9311/v1/orders/46f73695-82bb-447a-bf96-6635f0fb0ce7",
      "secret_ref": "http://barbican:9311/v1/secrets/22dfef44-1046-4549-a86d-95af462e8fa0",
      "status": "ACTIVE",
      "sub_status": "Unknown",
      "sub_status_message": "Unknown",
      "type": "key",
      "updated": "2018-06-22T05:05:43"
  	},
  	{
    	"created": "2018-06-22T05:08:15",
    	"creator_id": "5c70d99f4a8641c38f8084b32b5e5c0e",
    	"meta": {
      	"algorithm": "aes",
        "bit_length": 256,
        "expiration": null,
        "mode": "cbc",
        "name": null,
        "payload_content_type": "application/octet-stream"
      },
      "order_ref": "http://barbican:9311/v1/orders/07fba88b-3dcf-44e3-a4a3-0bad7f56f01c",
      "secret_ref": "http://barbican:9311/v1/secrets/a31ad551-1aa5-4ba0-810e-0865163e0fa9",
      "status": "ACTIVE",
      "sub_status": "Unknown",
      "sub_status_message": "Unknown",
      "type": "key",
      "updated": "2018-06-22T05:08:15"
    }
  ],
  "total": 2
}`

// GetResponse provides a Get result.
const GetResponse = `
{
  "created": "2018-06-22T05:08:15",
  "creator_id": "5c70d99f4a8641c38f8084b32b5e5c0e",
  "meta": {
    "algorithm": "aes",
    "bit_length": 256,
    "expiration": null,
    "mode": "cbc",
    "name": null,
    "payload_content_type": "application/octet-stream"
  },
  "order_ref": "http://barbican:9311/v1/orders/07fba88b-3dcf-44e3-a4a3-0bad7f56f01c",
  "secret_ref": "http://barbican:9311/v1/secrets/a31ad551-1aa5-4ba0-810e-0865163e0fa9",
  "status": "ACTIVE",
  "sub_status": "Unknown",
  "sub_status_message": "Unknown",
  "type": "key",
  "updated": "2018-06-22T05:08:15"
}
`

// CreateRequest provides the input to a Create request.
const CreateRequest = `
{
  "meta": {
    "algorithm": "aes",
    "bit_length": 256,
    "mode": "cbc",
    "payload_content_type": "application/octet-stream"
  },
  "type": "key"
}`

// FirstOrder is the first resource in the List request.
var FirstOrder = orders.Order{
	Created:   time.Date(2018, 6, 22, 5, 5, 43, 0, time.UTC),
	CreatorID: "5c70d99f4a8641c38f8084b32b5e5c0e",
	Meta: orders.Meta{
		Algorithm:          "aes",
		BitLength:          256,
		Mode:               "cbc",
		PayloadContentType: "application/octet-stream",
	},
	OrderRef:         "http://barbican:9311/v1/orders/46f73695-82bb-447a-bf96-6635f0fb0ce7",
	SecretRef:        "http://barbican:9311/v1/secrets/22dfef44-1046-4549-a86d-95af462e8fa0",
	Status:           "ACTIVE",
	SubStatus:        "Unknown",
	SubStatusMessage: "Unknown",
	Type:             "key",
	Updated:          time.Date(2018, 6, 22, 5, 5, 43, 0, time.UTC),
}

// SecondOrder is the second resource in the List request.
var SecondOrder = orders.Order{
	Created:   time.Date(2018, 6, 22, 5, 8, 15, 0, time.UTC),
	CreatorID: "5c70d99f4a8641c38f8084b32b5e5c0e",
	Meta: orders.Meta{
		Algorithm:          "aes",
		BitLength:          256,
		Mode:               "cbc",
		PayloadContentType: "application/octet-stream",
	},
	OrderRef:         "http://barbican:9311/v1/orders/07fba88b-3dcf-44e3-a4a3-0bad7f56f01c",
	SecretRef:        "http://barbican:9311/v1/secrets/a31ad551-1aa5-4ba0-810e-0865163e0fa9",
	Status:           "ACTIVE",
	SubStatus:        "Unknown",
	SubStatusMessage: "Unknown",
	Type:             "key",
	Updated:          time.Date(2018, 6, 22, 5, 8, 15, 0, time.UTC),
}

// ExpectedOrdersSlice is the slice of orders expected to be returned from ListResponse.
var ExpectedOrdersSlice = []orders.Order{FirstOrder, SecondOrder}

// HandleListOrdersSuccessfully creates an HTTP handler at `/orders` on the
// test handler mux that responds with a list of two orders.
func HandleListOrdersSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/orders", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListResponse)
	})
}

// HandleGetOrderSuccessfully creates an HTTP handler at `/orders` on the
// test handler mux that responds with a single resource.
func HandleGetOrderSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/orders/46f73695-82bb-447a-bf96-6635f0fb0ce7", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, GetResponse)
	})
}

// HandleCreateOrderSuccessfully creates an HTTP handler at `/orders` on the
// test handler mux that tests resource creation.
func HandleCreateOrderSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/orders", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, CreateRequest)

		w.WriteHeader(http.StatusAccepted)
		fmt.Fprintf(w, GetResponse)
	})
}

// HandleDeleteOrderSuccessfully creates an HTTP handler at `/orders` on the
// test handler mux that tests resource deletion.
func HandleDeleteOrderSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/orders/46f73695-82bb-447a-bf96-6635f0fb0ce7", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}
