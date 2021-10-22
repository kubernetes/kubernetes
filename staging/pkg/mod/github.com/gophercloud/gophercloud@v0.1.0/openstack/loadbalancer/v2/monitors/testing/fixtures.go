package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/monitors"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// HealthmonitorsListBody contains the canned body of a healthmonitor list response.
const HealthmonitorsListBody = `
{
	"healthmonitors":[
		{
			"admin_state_up":true,
			"project_id":"83657cfcdfe44cd5920adaf26c48ceea",
			"delay":10,
			"name":"web",
			"max_retries":1,
			"timeout":1,
			"type":"PING",
			"pools": [{"id": "84f1b61f-58c4-45bf-a8a9-2dafb9e5214d"}],
			"id":"466c8345-28d8-4f84-a246-e04380b0461d"
		},
		{
			"admin_state_up":true,
			"project_id":"83657cfcdfe44cd5920adaf26c48ceea",
			"delay":5,
			"name":"db",
			"expected_codes":"200",
			"max_retries":2,
			"http_method":"GET",
			"timeout":2,
			"url_path":"/",
			"type":"HTTP",
			"pools": [{"id": "d459f7d8-c6ee-439d-8713-d3fc08aeed8d"}],
			"id":"5d4b5228-33b0-4e60-b225-9b727c1a20e7"
		}
	]
}
`

// SingleHealthmonitorBody is the canned body of a Get request on an existing healthmonitor.
const SingleHealthmonitorBody = `
{
	"healthmonitor": {
		"admin_state_up":true,
		"project_id":"83657cfcdfe44cd5920adaf26c48ceea",
		"delay":5,
		"name":"db",
		"expected_codes":"200",
		"max_retries":2,
		"http_method":"GET",
		"timeout":2,
		"url_path":"/",
		"type":"HTTP",
		"pools": [{"id": "d459f7d8-c6ee-439d-8713-d3fc08aeed8d"}],
		"id":"5d4b5228-33b0-4e60-b225-9b727c1a20e7"
	}
}
`

// PostUpdateHealthmonitorBody is the canned response body of a Update request on an existing healthmonitor.
const PostUpdateHealthmonitorBody = `
{
	"healthmonitor": {
		"admin_state_up":true,
		"project_id":"83657cfcdfe44cd5920adaf26c48ceea",
		"delay":3,
		"name":"NewHealthmonitorName",
		"expected_codes":"301",
		"max_retries":10,
		"http_method":"GET",
		"timeout":20,
		"url_path":"/another_check",
		"type":"HTTP",
		"pools": [{"id": "d459f7d8-c6ee-439d-8713-d3fc08aeed8d"}],
		"id":"5d4b5228-33b0-4e60-b225-9b727c1a20e7"
	}
}
`

var (
	HealthmonitorWeb = monitors.Monitor{
		AdminStateUp: true,
		Name:         "web",
		ProjectID:    "83657cfcdfe44cd5920adaf26c48ceea",
		Delay:        10,
		MaxRetries:   1,
		Timeout:      1,
		Type:         "PING",
		ID:           "466c8345-28d8-4f84-a246-e04380b0461d",
		Pools:        []monitors.PoolID{{ID: "84f1b61f-58c4-45bf-a8a9-2dafb9e5214d"}},
	}
	HealthmonitorDb = monitors.Monitor{
		AdminStateUp:  true,
		Name:          "db",
		ProjectID:     "83657cfcdfe44cd5920adaf26c48ceea",
		Delay:         5,
		ExpectedCodes: "200",
		MaxRetries:    2,
		Timeout:       2,
		URLPath:       "/",
		Type:          "HTTP",
		HTTPMethod:    "GET",
		ID:            "5d4b5228-33b0-4e60-b225-9b727c1a20e7",
		Pools:         []monitors.PoolID{{ID: "d459f7d8-c6ee-439d-8713-d3fc08aeed8d"}},
	}
	HealthmonitorUpdated = monitors.Monitor{
		AdminStateUp:  true,
		Name:          "NewHealthmonitorName",
		ProjectID:     "83657cfcdfe44cd5920adaf26c48ceea",
		Delay:         3,
		ExpectedCodes: "301",
		MaxRetries:    10,
		Timeout:       20,
		URLPath:       "/another_check",
		Type:          "HTTP",
		HTTPMethod:    "GET",
		ID:            "5d4b5228-33b0-4e60-b225-9b727c1a20e7",
		Pools:         []monitors.PoolID{{ID: "d459f7d8-c6ee-439d-8713-d3fc08aeed8d"}},
	}
)

// HandleHealthmonitorListSuccessfully sets up the test server to respond to a healthmonitor List request.
func HandleHealthmonitorListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/healthmonitors", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, HealthmonitorsListBody)
		case "556c8345-28d8-4f84-a246-e04380b0461d":
			fmt.Fprintf(w, `{ "healthmonitors": [] }`)
		default:
			t.Fatalf("/v2.0/lbaas/healthmonitors invoked with unexpected marker=[%s]", marker)
		}
	})
}

// HandleHealthmonitorCreationSuccessfully sets up the test server to respond to a healthmonitor creation request
// with a given response.
func HandleHealthmonitorCreationSuccessfully(t *testing.T, response string) {
	th.Mux.HandleFunc("/v2.0/lbaas/healthmonitors", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{
			"healthmonitor": {
				"type":"HTTP",
				"pool_id":"84f1b61f-58c4-45bf-a8a9-2dafb9e5214d",
				"project_id":"453105b9-1754-413f-aab1-55f1af620750",
				"delay":20,
				"name":"db",
				"timeout":10,
				"max_retries":5,
				"url_path":"/check",
				"expected_codes":"200-299"
			}
		}`)

		w.WriteHeader(http.StatusAccepted)
		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, response)
	})
}

// HandleHealthmonitorGetSuccessfully sets up the test server to respond to a healthmonitor Get request.
func HandleHealthmonitorGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/healthmonitors/5d4b5228-33b0-4e60-b225-9b727c1a20e7", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, SingleHealthmonitorBody)
	})
}

// HandleHealthmonitorDeletionSuccessfully sets up the test server to respond to a healthmonitor deletion request.
func HandleHealthmonitorDeletionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/healthmonitors/5d4b5228-33b0-4e60-b225-9b727c1a20e7", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleHealthmonitorUpdateSuccessfully sets up the test server to respond to a healthmonitor Update request.
func HandleHealthmonitorUpdateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/healthmonitors/5d4b5228-33b0-4e60-b225-9b727c1a20e7", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestJSONRequest(t, r, `{
			"healthmonitor": {
				"name": "NewHealthmonitorName",
				"delay": 3,
				"timeout": 20,
				"max_retries": 10,
				"url_path": "/another_check",
				"expected_codes": "301"
			}
		}`)

		fmt.Fprintf(w, PostUpdateHealthmonitorBody)
	})
}
