package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/clustering/v1/events"

	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const ListResponse = `
{
	"events": [
		{
			"action": "CLUSTER_CREATE",
			"cluster": null,
			"cluster_id": null,
			"id": "edce3528-864f-41fb-8759-f4707925cc09",
			"level": "INFO",
			"meta_data": {},
			"oid": "0df0931b-e251-4f2e-8719-4ebfda3627ba",
			"oname": "cluster001",
			"otype": "CLUSTER",
			"project": "f1fe61dcda2f4618a14c10dc7abc214d",
			"status": "start",
			"status_reason": "Initializing",
			"timestamp": "2015-03-05T08:53:15Z",
			"user": "8bcd2cdca7684c02afc9e4f2fc0f0c79"
		},
		{
			"action": "NODE_DELETE",
			"cluster": null,
			"cluster_id": null,
			"id": "abcd1234-864f-41fb-8759-f4707925dd10",
			"level": "INFO",
			"meta_data": {},
			"oid": "0df0931b-e251-4f2e-8719-4ebfda3627ba",
			"oname": "node119",
			"otype": "node",
			"project": "f1fe61dcda2f4618a14c10dc7abc214d",
			"status": "start",
			"status_reason": "84492c96",
			"timestamp": "2015-03-06T18:53:15Z",
			"user": "8bcd2cdca7684c02afc9e4f2fc0f0c79"
		}
	]
}
`

const GetResponse = `
{
	"event": {
		"action": "CLUSTER_CREATE",
		"cluster_id": null,
		"id": "edce3528-864f-41fb-8759-f4707925cc09",
		"level": "INFO",
		"meta_data": {},
		"oid": "0df0931b-e251-4f2e-8719-4ebfda3627ba",
		"oname": "cluster001",
		"otype": "CLUSTER",
		"project": "f1fe61dcda2f4618a14c10dc7abc214d",
		"status": "start",
		"status_reason": "Initializing",
		"timestamp": "2015-03-05T08:53:15Z",
		"user": "8bcd2cdca7684c02afc9e4f2fc0f0c79"
	}
}
`

var ExpectedEvent1 = events.Event{
	Action:       "CLUSTER_CREATE",
	Cluster:      "",
	ClusterID:    "",
	ID:           "edce3528-864f-41fb-8759-f4707925cc09",
	Level:        "INFO",
	Metadata:     map[string]interface{}{},
	OID:          "0df0931b-e251-4f2e-8719-4ebfda3627ba",
	OName:        "cluster001",
	OType:        "CLUSTER",
	Project:      "f1fe61dcda2f4618a14c10dc7abc214d",
	Status:       "start",
	StatusReason: "Initializing",
	Timestamp:    time.Date(2015, 3, 5, 8, 53, 15, 0, time.UTC),
	User:         "8bcd2cdca7684c02afc9e4f2fc0f0c79",
}

var ExpectedEvent2 = events.Event{
	Action:       "NODE_DELETE",
	Cluster:      "",
	ClusterID:    "",
	ID:           "abcd1234-864f-41fb-8759-f4707925dd10",
	Level:        "INFO",
	Metadata:     map[string]interface{}{},
	OID:          "0df0931b-e251-4f2e-8719-4ebfda3627ba",
	OName:        "node119",
	OType:        "node",
	Project:      "f1fe61dcda2f4618a14c10dc7abc214d",
	Status:       "start",
	StatusReason: "84492c96",
	Timestamp:    time.Date(2015, 3, 6, 18, 53, 15, 0, time.UTC),
	User:         "8bcd2cdca7684c02afc9e4f2fc0f0c79",
}

var ExpectedEvents = []events.Event{ExpectedEvent1, ExpectedEvent2}

func HandleListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/events", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListResponse)
	})
}

func HandleGetSuccessfully(t *testing.T, id string) {
	th.Mux.HandleFunc("/v1/events/"+id, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, GetResponse)
	})
}
