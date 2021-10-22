package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/clustering/v1/actions"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const ListResponse = `
{
	"actions": [
		{
			"action": "NODE_DELETE",
			"cause": "RPC Request",
			"created_at": "2015-11-04T05:21:41Z",
			"data": {},
			"depended_by": ["ef67fe80-6547-40f2-ba1b-83e950aa38df"],
			"depends_on": ["ef67fe80-6547-40f2-ba1b-83e950aa38df"],
			"end_time": 1425550000.0,
			"id": "edce3528-864f-41fb-8759-f4707925cc09",
			"inputs": {},
			"interval": -1,
			"name": "node_delete_f0de9b9c",
			"outputs": {},
			"owner": null,
			"project": "f1fe61dcda2f4618a14c10dc7abc214d",
			"start_time": 1425550000.0,
			"status": "SUCCEEDED",
			"status_reason": "Action completed successfully.",
			"target": "f0de9b9c-6d48-4a46-af21-2ca8607777fe",
			"timeout": 3600,
			"updated_at": "2016-11-04T05:21:41Z",
			"user": "8bcd2cdca7684c02afc9e4f2fc0f0c79"
		},
		{
			"action": "NODE_DELETE",
			"cause": "RPC Request",
			"created_at": null,
			"data": {},
			"depended_by": ["ef67fe80-6547-40f2-ba1b-83e950aa38df"],
			"depends_on": ["ef67fe80-6547-40f2-ba1b-83e950aa38df"],
			"end_time": 1425550000.0,
			"id": "edce3528-864f-41fb-8759-f4707925cc09",
			"inputs": {},
			"interval": -1,
			"name": "node_delete_f0de9b9c",
			"outputs": {},
			"owner": null,
			"project": "f1fe61dcda2f4618a14c10dc7abc214d",
			"start_time": 1425550000.0,
			"status": "SUCCEEDED",
			"status_reason": "Action completed successfully.",
			"target": "f0de9b9c-6d48-4a46-af21-2ca8607777fe",
			"timeout": 3600,
			"updated_at": "",
			"user": "8bcd2cdca7684c02afc9e4f2fc0f0c79"
		}
	]
}
`

const GetResponse = `
{
	"action": {
		"action": "NODE_DELETE",
		"cause": "RPC Request",
		"created_at": "2015-11-04T05:21:41Z",
		"data": {},
		"depended_by": ["ef67fe80-6547-40f2-ba1b-83e950aa38df"],
		"depends_on": ["ef67fe80-6547-40f2-ba1b-83e950aa38df"],
		"end_time": 1425550000.0,
		"id": "edce3528-864f-41fb-8759-f4707925cc09",
		"inputs": {},
		"interval": -1,
		"name": "node_delete_f0de9b9c",
		"outputs": {},
		"owner": null,
		"project": "f1fe61dcda2f4618a14c10dc7abc214d",
		"start_time": 1425550000.0,
		"status": "SUCCEEDED",
		"status_reason": "Action completed successfully.",
		"target": "f0de9b9c-6d48-4a46-af21-2ca8607777fe",
		"timeout": 3600,
		"updated_at": "2016-11-04T05:21:41Z",
		"user": "8bcd2cdca7684c02afc9e4f2fc0f0c79"
	}
}
`

var ExpectedAction1 = actions.Action{
	Action:       "NODE_DELETE",
	Cause:        "RPC Request",
	CreatedAt:    time.Date(2015, 11, 4, 5, 21, 41, 0, time.UTC),
	Data:         map[string]interface{}{},
	DependedBy:   []string{"ef67fe80-6547-40f2-ba1b-83e950aa38df"},
	DependsOn:    []string{"ef67fe80-6547-40f2-ba1b-83e950aa38df"},
	EndTime:      1425550000.0,
	ID:           "edce3528-864f-41fb-8759-f4707925cc09",
	Inputs:       make(map[string]interface{}),
	Interval:     -1,
	Name:         "node_delete_f0de9b9c",
	Outputs:      make(map[string]interface{}),
	Owner:        "",
	Project:      "f1fe61dcda2f4618a14c10dc7abc214d",
	StartTime:    1425550000.0,
	Status:       "SUCCEEDED",
	StatusReason: "Action completed successfully.",
	Target:       "f0de9b9c-6d48-4a46-af21-2ca8607777fe",
	Timeout:      3600,
	UpdatedAt:    time.Date(2016, 11, 4, 5, 21, 41, 0, time.UTC),
	User:         "8bcd2cdca7684c02afc9e4f2fc0f0c79",
}

var ExpectedAction2 = actions.Action{
	Action:       "NODE_DELETE",
	Cause:        "RPC Request",
	CreatedAt:    time.Time{},
	Data:         map[string]interface{}{},
	DependedBy:   []string{"ef67fe80-6547-40f2-ba1b-83e950aa38df"},
	DependsOn:    []string{"ef67fe80-6547-40f2-ba1b-83e950aa38df"},
	EndTime:      1425550000.0,
	ID:           "edce3528-864f-41fb-8759-f4707925cc09",
	Inputs:       make(map[string]interface{}),
	Interval:     -1,
	Name:         "node_delete_f0de9b9c",
	Outputs:      make(map[string]interface{}),
	Owner:        "",
	Project:      "f1fe61dcda2f4618a14c10dc7abc214d",
	StartTime:    1425550000.0,
	Status:       "SUCCEEDED",
	StatusReason: "Action completed successfully.",
	Target:       "f0de9b9c-6d48-4a46-af21-2ca8607777fe",
	Timeout:      3600,
	UpdatedAt:    time.Time{},
	User:         "8bcd2cdca7684c02afc9e4f2fc0f0c79",
}

var ExpectedActions = []actions.Action{ExpectedAction1, ExpectedAction2}

func HandleListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListResponse)
	})
}

func HandleGetSuccessfully(t *testing.T, id string) {
	th.Mux.HandleFunc("/v1/actions/"+id, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, GetResponse)
	})
}
