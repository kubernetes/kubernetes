package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/pools"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// PoolsListBody contains the canned body of a pool list response.
const PoolsListBody = `
{
	"pools":[
	         {
			"lb_algorithm":"ROUND_ROBIN",
			"protocol":"HTTP",
			"description":"",
			"healthmonitor_id": "466c8345-28d8-4f84-a246-e04380b0461d",
			"members":[{"id": "53306cda-815d-4354-9fe4-59e09da9c3c5"}],
			"listeners":[{"id": "2a280670-c202-4b0b-a562-34077415aabf"}],
			"loadbalancers":[{"id": "79e05663-7f03-45d2-a092-8b94062f22ab"}],
			"id":"72741b06-df4d-4715-b142-276b6bce75ab",
			"name":"web",
			"admin_state_up":true,
			"tenant_id":"83657cfcdfe44cd5920adaf26c48ceea",
			"provider": "haproxy"
		},
		{
			"lb_algorithm":"LEAST_CONNECTION",
			"protocol":"HTTP",
			"description":"",
			"healthmonitor_id": "5f6c8345-28d8-4f84-a246-e04380b0461d",
			"members":[{"id": "67306cda-815d-4354-9fe4-59e09da9c3c5"}],
			"listeners":[{"id": "2a280670-c202-4b0b-a562-34077415aabf"}],
			"loadbalancers":[{"id": "79e05663-7f03-45d2-a092-8b94062f22ab"}],
			"id":"c3741b06-df4d-4715-b142-276b6bce75ab",
			"name":"db",
			"admin_state_up":true,
			"tenant_id":"83657cfcdfe44cd5920adaf26c48ceea",
			"provider": "haproxy"
		}
	]
}
`

// SinglePoolBody is the canned body of a Get request on an existing pool.
const SinglePoolBody = `
{
	"pool": {
		"lb_algorithm":"LEAST_CONNECTION",
		"protocol":"HTTP",
		"description":"",
		"healthmonitor_id": "5f6c8345-28d8-4f84-a246-e04380b0461d",
		"members":[{"id": "67306cda-815d-4354-9fe4-59e09da9c3c5"}],
		"listeners":[{"id": "2a280670-c202-4b0b-a562-34077415aabf"}],
		"loadbalancers":[{"id": "79e05663-7f03-45d2-a092-8b94062f22ab"}],
		"id":"c3741b06-df4d-4715-b142-276b6bce75ab",
		"name":"db",
		"admin_state_up":true,
		"tenant_id":"83657cfcdfe44cd5920adaf26c48ceea",
		"provider": "haproxy"
	}
}
`

// PostUpdatePoolBody is the canned response body of a Update request on an existing pool.
const PostUpdatePoolBody = `
{
	"pool": {
		"lb_algorithm":"LEAST_CONNECTION",
		"protocol":"HTTP",
		"description":"",
		"healthmonitor_id": "5f6c8345-28d8-4f84-a246-e04380b0461d",
		"members":[{"id": "67306cda-815d-4354-9fe4-59e09da9c3c5"}],
		"listeners":[{"id": "2a280670-c202-4b0b-a562-34077415aabf"}],
		"loadbalancers":[{"id": "79e05663-7f03-45d2-a092-8b94062f22ab"}],
		"id":"c3741b06-df4d-4715-b142-276b6bce75ab",
		"name":"db",
		"admin_state_up":true,
		"tenant_id":"83657cfcdfe44cd5920adaf26c48ceea",
		"provider": "haproxy"
	}
}
`

var (
	PoolWeb = pools.Pool{
		LBMethod:      "ROUND_ROBIN",
		Protocol:      "HTTP",
		Description:   "",
		MonitorID:     "466c8345-28d8-4f84-a246-e04380b0461d",
		TenantID:      "83657cfcdfe44cd5920adaf26c48ceea",
		AdminStateUp:  true,
		Name:          "web",
		Members:       []pools.Member{{ID: "53306cda-815d-4354-9fe4-59e09da9c3c5"}},
		ID:            "72741b06-df4d-4715-b142-276b6bce75ab",
		Loadbalancers: []pools.LoadBalancerID{{ID: "79e05663-7f03-45d2-a092-8b94062f22ab"}},
		Listeners:     []pools.ListenerID{{ID: "2a280670-c202-4b0b-a562-34077415aabf"}},
		Provider:      "haproxy",
	}
	PoolDb = pools.Pool{
		LBMethod:      "LEAST_CONNECTION",
		Protocol:      "HTTP",
		Description:   "",
		MonitorID:     "5f6c8345-28d8-4f84-a246-e04380b0461d",
		TenantID:      "83657cfcdfe44cd5920adaf26c48ceea",
		AdminStateUp:  true,
		Name:          "db",
		Members:       []pools.Member{{ID: "67306cda-815d-4354-9fe4-59e09da9c3c5"}},
		ID:            "c3741b06-df4d-4715-b142-276b6bce75ab",
		Loadbalancers: []pools.LoadBalancerID{{ID: "79e05663-7f03-45d2-a092-8b94062f22ab"}},
		Listeners:     []pools.ListenerID{{ID: "2a280670-c202-4b0b-a562-34077415aabf"}},
		Provider:      "haproxy",
	}
	PoolUpdated = pools.Pool{
		LBMethod:      "LEAST_CONNECTION",
		Protocol:      "HTTP",
		Description:   "",
		MonitorID:     "5f6c8345-28d8-4f84-a246-e04380b0461d",
		TenantID:      "83657cfcdfe44cd5920adaf26c48ceea",
		AdminStateUp:  true,
		Name:          "db",
		Members:       []pools.Member{{ID: "67306cda-815d-4354-9fe4-59e09da9c3c5"}},
		ID:            "c3741b06-df4d-4715-b142-276b6bce75ab",
		Loadbalancers: []pools.LoadBalancerID{{ID: "79e05663-7f03-45d2-a092-8b94062f22ab"}},
		Listeners:     []pools.ListenerID{{ID: "2a280670-c202-4b0b-a562-34077415aabf"}},
		Provider:      "haproxy",
	}
)

// HandlePoolListSuccessfully sets up the test server to respond to a pool List request.
func HandlePoolListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/pools", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, PoolsListBody)
		case "45e08a3e-a78f-4b40-a229-1e7e23eee1ab":
			fmt.Fprintf(w, `{ "pools": [] }`)
		default:
			t.Fatalf("/v2.0/lbaas/pools invoked with unexpected marker=[%s]", marker)
		}
	})
}

// HandlePoolCreationSuccessfully sets up the test server to respond to a pool creation request
// with a given response.
func HandlePoolCreationSuccessfully(t *testing.T, response string) {
	th.Mux.HandleFunc("/v2.0/lbaas/pools", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{
			"pool": {
			        "lb_algorithm": "ROUND_ROBIN",
			        "protocol": "HTTP",
			        "name": "Example pool",
			        "tenant_id": "2ffc6e22aae24e4795f87155d24c896f",
			        "loadbalancer_id": "79e05663-7f03-45d2-a092-8b94062f22ab"
			}
		}`)

		w.WriteHeader(http.StatusAccepted)
		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, response)
	})
}

// HandlePoolGetSuccessfully sets up the test server to respond to a pool Get request.
func HandlePoolGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/pools/c3741b06-df4d-4715-b142-276b6bce75ab", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, SinglePoolBody)
	})
}

// HandlePoolDeletionSuccessfully sets up the test server to respond to a pool deletion request.
func HandlePoolDeletionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/pools/c3741b06-df4d-4715-b142-276b6bce75ab", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandlePoolUpdateSuccessfully sets up the test server to respond to a pool Update request.
func HandlePoolUpdateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/pools/c3741b06-df4d-4715-b142-276b6bce75ab", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestJSONRequest(t, r, `{
			"pool": {
				"name": "NewPoolName",
                                "lb_algorithm": "LEAST_CONNECTIONS"
			}
		}`)

		fmt.Fprintf(w, PostUpdatePoolBody)
	})
}

// MembersListBody contains the canned body of a member list response.
const MembersListBody = `
{
	"members":[
		{
			"id": "2a280670-c202-4b0b-a562-34077415aabf",
			"address": "10.0.2.10",
			"weight": 5,
			"name": "web",
			"subnet_id": "1981f108-3c48-48d2-b908-30f7d28532c9",
			"tenant_id": "2ffc6e22aae24e4795f87155d24c896f",
			"admin_state_up":true,
			"protocol_port": 80
		},
		{
			"id": "fad389a3-9a4a-4762-a365-8c7038508b5d",
			"address": "10.0.2.11",
			"weight": 10,
			"name": "db",
			"subnet_id": "1981f108-3c48-48d2-b908-30f7d28532c9",
			"tenant_id": "2ffc6e22aae24e4795f87155d24c896f",
			"admin_state_up":false,
			"protocol_port": 80
		}
	]
}
`

// SingleMemberBody is the canned body of a Get request on an existing member.
const SingleMemberBody = `
{
	"member": {
		"id": "fad389a3-9a4a-4762-a365-8c7038508b5d",
		"address": "10.0.2.11",
		"weight": 10,
		"name": "db",
		"subnet_id": "1981f108-3c48-48d2-b908-30f7d28532c9",
		"tenant_id": "2ffc6e22aae24e4795f87155d24c896f",
		"admin_state_up":false,
		"protocol_port": 80
	}
}
`

// PostUpdateMemberBody is the canned response body of a Update request on an existing member.
const PostUpdateMemberBody = `
{
	"member": {
		"id": "fad389a3-9a4a-4762-a365-8c7038508b5d",
		"address": "10.0.2.11",
		"weight": 10,
		"name": "db",
		"subnet_id": "1981f108-3c48-48d2-b908-30f7d28532c9",
		"tenant_id": "2ffc6e22aae24e4795f87155d24c896f",
		"admin_state_up":false,
		"protocol_port": 80
	}
}
`

var (
	MemberWeb = pools.Member{
		SubnetID:     "1981f108-3c48-48d2-b908-30f7d28532c9",
		TenantID:     "2ffc6e22aae24e4795f87155d24c896f",
		AdminStateUp: true,
		Name:         "web",
		ID:           "2a280670-c202-4b0b-a562-34077415aabf",
		Address:      "10.0.2.10",
		Weight:       5,
		ProtocolPort: 80,
	}
	MemberDb = pools.Member{
		SubnetID:     "1981f108-3c48-48d2-b908-30f7d28532c9",
		TenantID:     "2ffc6e22aae24e4795f87155d24c896f",
		AdminStateUp: false,
		Name:         "db",
		ID:           "fad389a3-9a4a-4762-a365-8c7038508b5d",
		Address:      "10.0.2.11",
		Weight:       10,
		ProtocolPort: 80,
	}
	MemberUpdated = pools.Member{
		SubnetID:     "1981f108-3c48-48d2-b908-30f7d28532c9",
		TenantID:     "2ffc6e22aae24e4795f87155d24c896f",
		AdminStateUp: false,
		Name:         "db",
		ID:           "fad389a3-9a4a-4762-a365-8c7038508b5d",
		Address:      "10.0.2.11",
		Weight:       10,
		ProtocolPort: 80,
	}
)

// HandleMemberListSuccessfully sets up the test server to respond to a member List request.
func HandleMemberListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/pools/332abe93-f488-41ba-870b-2ac66be7f853/members", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, MembersListBody)
		case "45e08a3e-a78f-4b40-a229-1e7e23eee1ab":
			fmt.Fprintf(w, `{ "members": [] }`)
		default:
			t.Fatalf("/v2.0/lbaas/pools/332abe93-f488-41ba-870b-2ac66be7f853/members invoked with unexpected marker=[%s]", marker)
		}
	})
}

// HandleMemberCreationSuccessfully sets up the test server to respond to a member creation request
// with a given response.
func HandleMemberCreationSuccessfully(t *testing.T, response string) {
	th.Mux.HandleFunc("/v2.0/lbaas/pools/332abe93-f488-41ba-870b-2ac66be7f853/members", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{
			"member": {
			        "address": "10.0.2.11",
			        "weight": 10,
			        "name": "db",
			        "subnet_id": "1981f108-3c48-48d2-b908-30f7d28532c9",
			        "tenant_id": "2ffc6e22aae24e4795f87155d24c896f",
			        "protocol_port": 80
			}
		}`)

		w.WriteHeader(http.StatusAccepted)
		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, response)
	})
}

// HandleMemberGetSuccessfully sets up the test server to respond to a member Get request.
func HandleMemberGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/pools/332abe93-f488-41ba-870b-2ac66be7f853/members/2a280670-c202-4b0b-a562-34077415aabf", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, SingleMemberBody)
	})
}

// HandleMemberDeletionSuccessfully sets up the test server to respond to a member deletion request.
func HandleMemberDeletionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/pools/332abe93-f488-41ba-870b-2ac66be7f853/members/2a280670-c202-4b0b-a562-34077415aabf", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleMemberUpdateSuccessfully sets up the test server to respond to a member Update request.
func HandleMemberUpdateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/pools/332abe93-f488-41ba-870b-2ac66be7f853/members/2a280670-c202-4b0b-a562-34077415aabf", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestJSONRequest(t, r, `{
			"member": {
				"name": "newMemberName",
                                "weight": 4
			}
		}`)

		fmt.Fprintf(w, PostUpdateMemberBody)
	})
}
