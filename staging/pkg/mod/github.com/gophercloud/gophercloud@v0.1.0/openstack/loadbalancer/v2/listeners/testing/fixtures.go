package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/listeners"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ListenersListBody contains the canned body of a listeners list response.
const ListenersListBody = `
{
	"listeners":[
		{
			"id": "db902c0c-d5ff-4753-b465-668ad9656918",
			"project_id": "310df60f-2a10-4ee5-9554-98393092194c",
			"name": "web",
			"description": "listener config for the web tier",
			"loadbalancers": [{"id": "53306cda-815d-4354-9444-59e09da9c3c5"}],
			"protocol": "HTTP",
			"protocol_port": 80,
			"default_pool_id": "fad389a3-9a4a-4762-a365-8c7038508b5d",
			"admin_state_up": true,
			"default_tls_container_ref": "2c433435-20de-4411-84ae-9cc8917def76",
			"sni_container_refs": ["3d328d82-2547-4921-ac2f-61c3b452b5ff", "b3cfd7e3-8c19-455c-8ebb-d78dfd8f7e7d"]
		},
		{
			"id": "36e08a3e-a78f-4b40-a229-1e7e23eee1ab",
			"project_id": "310df60f-2a10-4ee5-9554-98393092194c",
			"name": "db",
			"description": "listener config for the db tier",
			"loadbalancers": [{"id": "79e05663-7f03-45d2-a092-8b94062f22ab"}],
			"protocol": "TCP",
			"protocol_port": 3306,
			"default_pool_id": "41efe233-7591-43c5-9cf7-923964759f9e",
			"connection_limit": 2000,
			"admin_state_up": true,
			"default_tls_container_ref": "2c433435-20de-4411-84ae-9cc8917def76",
			"sni_container_refs": ["3d328d82-2547-4921-ac2f-61c3b452b5ff", "b3cfd7e3-8c19-455c-8ebb-d78dfd8f7e7d"],
			"timeout_client_data": 50000,
			"timeout_member_data": 50000,
			"timeout_member_connect": 5000,
			"timeout_tcp_inspect": 0,
			"insert_headers": {
				"X-Forwarded-For": "true"
			}
		}
	]
}
`

// SingleServerBody is the canned body of a Get request on an existing listener.
const SingleListenerBody = `
{
	"listener": {
		"id": "36e08a3e-a78f-4b40-a229-1e7e23eee1ab",
		"project_id": "310df60f-2a10-4ee5-9554-98393092194c",
		"name": "db",
		"description": "listener config for the db tier",
		"loadbalancers": [{"id": "79e05663-7f03-45d2-a092-8b94062f22ab"}],
		"protocol": "TCP",
		"protocol_port": 3306,
		"default_pool_id": "41efe233-7591-43c5-9cf7-923964759f9e",
		"connection_limit": 2000,
		"admin_state_up": true,
		"default_tls_container_ref": "2c433435-20de-4411-84ae-9cc8917def76",
		"sni_container_refs": ["3d328d82-2547-4921-ac2f-61c3b452b5ff", "b3cfd7e3-8c19-455c-8ebb-d78dfd8f7e7d"],
		"timeout_client_data": 50000,
		"timeout_member_data": 50000,
		"timeout_member_connect": 5000,
		"timeout_tcp_inspect": 0,
        	"insert_headers": {
            		"X-Forwarded-For": "true"
        	}
	}
}
`

// PostUpdateListenerBody is the canned response body of a Update request on an existing listener.
const PostUpdateListenerBody = `
{
	"listener": {
		"id": "36e08a3e-a78f-4b40-a229-1e7e23eee1ab",
		"project_id": "310df60f-2a10-4ee5-9554-98393092194c",
		"name": "NewListenerName",
		"description": "listener config for the db tier",
		"loadbalancers": [{"id": "79e05663-7f03-45d2-a092-8b94062f22ab"}],
		"protocol": "TCP",
		"protocol_port": 3306,
		"default_pool_id": "41efe233-7591-43c5-9cf7-923964759f9e",
		"connection_limit": 1000,
		"admin_state_up": true,
		"default_tls_container_ref": "2c433435-20de-4411-84ae-9cc8917def76",
		"sni_container_refs": ["3d328d82-2547-4921-ac2f-61c3b452b5ff", "b3cfd7e3-8c19-455c-8ebb-d78dfd8f7e7d"],
		"timeout_client_data": 181000,
		"timeout_member_data": 181000,
		"timeout_member_connect": 181000,
		"timeout_tcp_inspect": 181000

	}
}
`

// GetListenerStatsBody is the canned request body of a Get request on listener's statistics.
const GetListenerStatsBody = `
{
    "stats": {
        "active_connections": 0,
        "bytes_in": 9532,
        "bytes_out": 22033,
        "request_errors": 46,
        "total_connections": 112
    }
}
`

var (
	ListenerWeb = listeners.Listener{
		ID:                     "db902c0c-d5ff-4753-b465-668ad9656918",
		ProjectID:              "310df60f-2a10-4ee5-9554-98393092194c",
		Name:                   "web",
		Description:            "listener config for the web tier",
		Loadbalancers:          []listeners.LoadBalancerID{{ID: "53306cda-815d-4354-9444-59e09da9c3c5"}},
		Protocol:               "HTTP",
		ProtocolPort:           80,
		DefaultPoolID:          "fad389a3-9a4a-4762-a365-8c7038508b5d",
		AdminStateUp:           true,
		DefaultTlsContainerRef: "2c433435-20de-4411-84ae-9cc8917def76",
		SniContainerRefs:       []string{"3d328d82-2547-4921-ac2f-61c3b452b5ff", "b3cfd7e3-8c19-455c-8ebb-d78dfd8f7e7d"},
	}
	ListenerDb = listeners.Listener{
		ID:                     "36e08a3e-a78f-4b40-a229-1e7e23eee1ab",
		ProjectID:              "310df60f-2a10-4ee5-9554-98393092194c",
		Name:                   "db",
		Description:            "listener config for the db tier",
		Loadbalancers:          []listeners.LoadBalancerID{{ID: "79e05663-7f03-45d2-a092-8b94062f22ab"}},
		Protocol:               "TCP",
		ProtocolPort:           3306,
		DefaultPoolID:          "41efe233-7591-43c5-9cf7-923964759f9e",
		ConnLimit:              2000,
		AdminStateUp:           true,
		DefaultTlsContainerRef: "2c433435-20de-4411-84ae-9cc8917def76",
		SniContainerRefs:       []string{"3d328d82-2547-4921-ac2f-61c3b452b5ff", "b3cfd7e3-8c19-455c-8ebb-d78dfd8f7e7d"},
		TimeoutClientData:      50000,
		TimeoutMemberData:      50000,
		TimeoutMemberConnect:   5000,
		TimeoutTCPInspect:      0,
		InsertHeaders:          map[string]string{"X-Forwarded-For": "true"},
	}
	ListenerUpdated = listeners.Listener{
		ID:                     "36e08a3e-a78f-4b40-a229-1e7e23eee1ab",
		ProjectID:              "310df60f-2a10-4ee5-9554-98393092194c",
		Name:                   "NewListenerName",
		Description:            "listener config for the db tier",
		Loadbalancers:          []listeners.LoadBalancerID{{ID: "79e05663-7f03-45d2-a092-8b94062f22ab"}},
		Protocol:               "TCP",
		ProtocolPort:           3306,
		DefaultPoolID:          "41efe233-7591-43c5-9cf7-923964759f9e",
		ConnLimit:              1000,
		AdminStateUp:           true,
		DefaultTlsContainerRef: "2c433435-20de-4411-84ae-9cc8917def76",
		SniContainerRefs:       []string{"3d328d82-2547-4921-ac2f-61c3b452b5ff", "b3cfd7e3-8c19-455c-8ebb-d78dfd8f7e7d"},
		TimeoutClientData:      181000,
		TimeoutMemberData:      181000,
		TimeoutMemberConnect:   181000,
		TimeoutTCPInspect:      181000,
	}
	ListenerStatsTree = listeners.Stats{
		ActiveConnections: 0,
		BytesIn:           9532,
		BytesOut:          22033,
		RequestErrors:     46,
		TotalConnections:  112,
	}
)

// HandleListenerListSuccessfully sets up the test server to respond to a listener List request.
func HandleListenerListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/listeners", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, ListenersListBody)
		case "45e08a3e-a78f-4b40-a229-1e7e23eee1ab":
			fmt.Fprintf(w, `{ "listeners": [] }`)
		default:
			t.Fatalf("/v2.0/lbaas/listeners invoked with unexpected marker=[%s]", marker)
		}
	})
}

// HandleListenerCreationSuccessfully sets up the test server to respond to a listener creation request
// with a given response.
func HandleListenerCreationSuccessfully(t *testing.T, response string) {
	th.Mux.HandleFunc("/v2.0/lbaas/listeners", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{
			    "listener": {
			        "loadbalancer_id": "79e05663-7f03-45d2-a092-8b94062f22ab",
			        "protocol": "TCP",
			        "name": "db",
			        "admin_state_up": true,
			        "default_tls_container_ref": "2c433435-20de-4411-84ae-9cc8917def76",
			        "default_pool_id": "41efe233-7591-43c5-9cf7-923964759f9e",
			        "protocol_port": 3306,
				"insert_headers": {
					"X-Forwarded-For": "true"
				}
			    }
		}`)

		w.WriteHeader(http.StatusAccepted)
		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, response)
	})
}

// HandleListenerGetSuccessfully sets up the test server to respond to a listener Get request.
func HandleListenerGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/listeners/4ec89087-d057-4e2c-911f-60a3b47ee304", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, SingleListenerBody)
	})
}

// HandleListenerDeletionSuccessfully sets up the test server to respond to a listener deletion request.
func HandleListenerDeletionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/listeners/4ec89087-d057-4e2c-911f-60a3b47ee304", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleListenerUpdateSuccessfully sets up the test server to respond to a listener Update request.
func HandleListenerUpdateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/listeners/4ec89087-d057-4e2c-911f-60a3b47ee304", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestJSONRequest(t, r, `{
			"listener": {
				"name": "NewListenerName",
				"default_pool_id": null,
				"connection_limit": 1001,
				"timeout_client_data": 181000,
				"timeout_member_data": 181000,
				"timeout_member_connect": 181000,
				"timeout_tcp_inspect": 181000
			}
		}`)

		fmt.Fprintf(w, PostUpdateListenerBody)
	})
}

// HandleListenerGetStatsTree sets up the test server to respond to a listener Get stats tree request.
func HandleListenerGetStatsTree(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/listeners/4ec89087-d057-4e2c-911f-60a3b47ee304/stats", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, GetListenerStatsBody)
	})
}
