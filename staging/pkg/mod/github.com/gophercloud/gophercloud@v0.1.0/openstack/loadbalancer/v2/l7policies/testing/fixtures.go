package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/l7policies"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// SingleL7PolicyBody is the canned body of a Get request on an existing l7policy.
const SingleL7PolicyBody = `
{
	"l7policy": {
		"listener_id": "023f2e34-7806-443b-bfae-16c324569a3d",
		"description": "",
		"admin_state_up": true,
		"redirect_pool_id": null,
		"redirect_url": "http://www.example.com",
		"action": "REDIRECT_TO_URL",
		"position": 1,
		"project_id": "e3cd678b11784734bc366148aa37580e",
		"id": "8a1412f0-4c32-4257-8b07-af4770b604fd",
		"name": "redirect-example.com",
		"rules": []
	}
}
`

var (
	L7PolicyToURL = l7policies.L7Policy{
		ID:             "8a1412f0-4c32-4257-8b07-af4770b604fd",
		Name:           "redirect-example.com",
		ListenerID:     "023f2e34-7806-443b-bfae-16c324569a3d",
		Action:         "REDIRECT_TO_URL",
		Position:       1,
		Description:    "",
		ProjectID:      "e3cd678b11784734bc366148aa37580e",
		RedirectPoolID: "",
		RedirectURL:    "http://www.example.com",
		AdminStateUp:   true,
		Rules:          []l7policies.Rule{},
	}
	L7PolicyToPool = l7policies.L7Policy{
		ID:             "964f4ba4-f6cd-405c-bebd-639460af7231",
		Name:           "redirect-pool",
		ListenerID:     "be3138a3-5cf7-4513-a4c2-bb137e668bab",
		Action:         "REDIRECT_TO_POOL",
		Position:       1,
		Description:    "",
		ProjectID:      "c1f7910086964990847dc6c8b128f63c",
		RedirectPoolID: "bac433c6-5bea-4311-80da-bd1cd90fbd25",
		RedirectURL:    "",
		AdminStateUp:   true,
		Rules:          []l7policies.Rule{},
	}
	L7PolicyUpdated = l7policies.L7Policy{
		ID:             "8a1412f0-4c32-4257-8b07-af4770b604fd",
		Name:           "NewL7PolicyName",
		ListenerID:     "023f2e34-7806-443b-bfae-16c324569a3d",
		Action:         "REDIRECT_TO_URL",
		Position:       1,
		Description:    "Redirect requests to example.com",
		ProjectID:      "e3cd678b11784734bc366148aa37580e",
		RedirectPoolID: "",
		RedirectURL:    "http://www.new-example.com",
		AdminStateUp:   true,
		Rules:          []l7policies.Rule{},
	}
	L7PolicyNullRedirectURLUpdated = l7policies.L7Policy{
		ID:             "8a1412f0-4c32-4257-8b07-af4770b604fd",
		Name:           "NewL7PolicyName",
		ListenerID:     "023f2e34-7806-443b-bfae-16c324569a3d",
		Action:         "REDIRECT_TO_URL",
		Position:       1,
		Description:    "Redirect requests to example.com",
		ProjectID:      "e3cd678b11784734bc366148aa37580e",
		RedirectPoolID: "",
		RedirectURL:    "",
		AdminStateUp:   true,
		Rules:          []l7policies.Rule{},
	}
	RulePath = l7policies.Rule{
		ID:           "16621dbb-a736-4888-a57a-3ecd53df784c",
		RuleType:     "PATH",
		CompareType:  "REGEX",
		Value:        "/images*",
		ProjectID:    "e3cd678b11784734bc366148aa37580e",
		Key:          "",
		Invert:       true,
		AdminStateUp: true,
	}
	RuleHostName = l7policies.Rule{
		ID:           "d24521a0-df84-4468-861a-a531af116d1e",
		RuleType:     "HOST_NAME",
		CompareType:  "EQUAL_TO",
		Value:        "www.example.com",
		ProjectID:    "e3cd678b11784734bc366148aa37580e",
		Key:          "",
		Invert:       false,
		AdminStateUp: true,
	}
	RuleUpdated = l7policies.Rule{
		ID:           "16621dbb-a736-4888-a57a-3ecd53df784c",
		RuleType:     "PATH",
		CompareType:  "REGEX",
		Value:        "/images/special*",
		ProjectID:    "e3cd678b11784734bc366148aa37580e",
		Key:          "",
		Invert:       false,
		AdminStateUp: true,
	}
)

// HandleL7PolicyCreationSuccessfully sets up the test server to respond to a l7policy creation request
// with a given response.
func HandleL7PolicyCreationSuccessfully(t *testing.T, response string) {
	th.Mux.HandleFunc("/v2.0/lbaas/l7policies", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{
			"l7policy": {
				"listener_id": "023f2e34-7806-443b-bfae-16c324569a3d",
				"redirect_url": "http://www.example.com",
				"name": "redirect-example.com",
				"action": "REDIRECT_TO_URL"
			}
		}`)

		w.WriteHeader(http.StatusAccepted)
		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, response)
	})
}

// L7PoliciesListBody contains the canned body of a l7policy list response.
const L7PoliciesListBody = `
{
    "l7policies": [
        {
            "redirect_pool_id": null,
            "description": "",
            "admin_state_up": true,
            "rules": [],
            "project_id": "e3cd678b11784734bc366148aa37580e",
            "listener_id": "023f2e34-7806-443b-bfae-16c324569a3d",
            "redirect_url": "http://www.example.com",
            "action": "REDIRECT_TO_URL",
            "position": 1,
            "id": "8a1412f0-4c32-4257-8b07-af4770b604fd",
            "name": "redirect-example.com"
        },
        {
            "redirect_pool_id": "bac433c6-5bea-4311-80da-bd1cd90fbd25",
            "description": "",
            "admin_state_up": true,
            "rules": [],
            "project_id": "c1f7910086964990847dc6c8b128f63c",
            "listener_id": "be3138a3-5cf7-4513-a4c2-bb137e668bab",
            "action": "REDIRECT_TO_POOL",
            "position": 1,
            "id": "964f4ba4-f6cd-405c-bebd-639460af7231",
            "name": "redirect-pool"
        }
    ]
}
`

// PostUpdateL7PolicyBody is the canned response body of a Update request on an existing l7policy.
const PostUpdateL7PolicyBody = `
{
	"l7policy": {
		"listener_id": "023f2e34-7806-443b-bfae-16c324569a3d",
		"description": "Redirect requests to example.com",
		"admin_state_up": true,
		"redirect_url": "http://www.new-example.com",
		"action": "REDIRECT_TO_URL",
		"position": 1,
		"project_id": "e3cd678b11784734bc366148aa37580e",
		"id": "8a1412f0-4c32-4257-8b07-af4770b604fd",
		"name": "NewL7PolicyName",
		"rules": []
	}
}
`

// PostUpdateL7PolicyNullRedirectURLBody is the canned response body of a Update request
// on an existing l7policy with a null redirect_url .
const PostUpdateL7PolicyNullRedirectURLBody = `
{
	"l7policy": {
		"listener_id": "023f2e34-7806-443b-bfae-16c324569a3d",
		"description": "Redirect requests to example.com",
		"admin_state_up": true,
		"redirect_url": null,
		"action": "REDIRECT_TO_URL",
		"position": 1,
		"project_id": "e3cd678b11784734bc366148aa37580e",
		"id": "8a1412f0-4c32-4257-8b07-af4770b604fd",
		"name": "NewL7PolicyName",
		"rules": []
	}
}
`

// HandleL7PolicyListSuccessfully sets up the test server to respond to a l7policy List request.
func HandleL7PolicyListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/l7policies", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, L7PoliciesListBody)
		case "45e08a3e-a78f-4b40-a229-1e7e23eee1ab":
			fmt.Fprintf(w, `{ "l7policies": [] }`)
		default:
			t.Fatalf("/v2.0/lbaas/l7policies invoked with unexpected marker=[%s]", marker)
		}
	})
}

// HandleL7PolicyGetSuccessfully sets up the test server to respond to a l7policy Get request.
func HandleL7PolicyGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/l7policies/8a1412f0-4c32-4257-8b07-af4770b604fd", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, SingleL7PolicyBody)
	})
}

// HandleL7PolicyDeletionSuccessfully sets up the test server to respond to a l7policy deletion request.
func HandleL7PolicyDeletionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/l7policies/8a1412f0-4c32-4257-8b07-af4770b604fd", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleL7PolicyUpdateSuccessfully sets up the test server to respond to a l7policy Update request.
func HandleL7PolicyUpdateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/l7policies/8a1412f0-4c32-4257-8b07-af4770b604fd", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestJSONRequest(t, r, `{
			"l7policy": {
				"name": "NewL7PolicyName",
				"action": "REDIRECT_TO_URL",
				"redirect_url": "http://www.new-example.com"
			}
		}`)

		fmt.Fprintf(w, PostUpdateL7PolicyBody)
	})
}

// HandleL7PolicyUpdateNullRedirectURLSuccessfully sets up the test server to respond to a l7policy Update request.
func HandleL7PolicyUpdateNullRedirectURLSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/l7policies/8a1412f0-4c32-4257-8b07-af4770b604fd", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestJSONRequest(t, r, `{
			"l7policy": {
				"name": "NewL7PolicyName",
				"redirect_url": null
			}
		}`)

		fmt.Fprintf(w, PostUpdateL7PolicyNullRedirectURLBody)
	})
}

// SingleRuleBody is the canned body of a Get request on an existing rule.
const SingleRuleBody = `
{
	"rule": {
		"compare_type": "REGEX",
		"invert": true,
		"admin_state_up": true,
		"value": "/images*",
		"key": null,
		"project_id": "e3cd678b11784734bc366148aa37580e",
		"type": "PATH",
		"id": "16621dbb-a736-4888-a57a-3ecd53df784c"
	}
}
`

// HandleRuleCreationSuccessfully sets up the test server to respond to a rule creation request
// with a given response.
func HandleRuleCreationSuccessfully(t *testing.T, response string) {
	th.Mux.HandleFunc("/v2.0/lbaas/l7policies/8a1412f0-4c32-4257-8b07-af4770b604fd/rules", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{
			"rule": {
				"compare_type": "REGEX",
				"type": "PATH",
				"value": "/images*"
			}
		}`)

		w.WriteHeader(http.StatusAccepted)
		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, response)
	})
}

// RulesListBody contains the canned body of a rule list response.
const RulesListBody = `
{
	"rules":[
		{
            "compare_type": "REGEX",
            "invert": true,
            "admin_state_up": true,
            "value": "/images*",
            "key": null,
            "project_id": "e3cd678b11784734bc366148aa37580e",
            "type": "PATH",
            "id": "16621dbb-a736-4888-a57a-3ecd53df784c"
		},
		{
            "compare_type": "EQUAL_TO",
            "invert": false,
            "admin_state_up": true,
            "value": "www.example.com",
            "key": null,
            "project_id": "e3cd678b11784734bc366148aa37580e",
            "type": "HOST_NAME",
            "id": "d24521a0-df84-4468-861a-a531af116d1e"
		}
	]
}
`

// HandleRuleListSuccessfully sets up the test server to respond to a rule List request.
func HandleRuleListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/l7policies/8a1412f0-4c32-4257-8b07-af4770b604fd/rules", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, RulesListBody)
		case "45e08a3e-a78f-4b40-a229-1e7e23eee1ab":
			fmt.Fprintf(w, `{ "rules": [] }`)
		default:
			t.Fatalf("/v2.0/lbaas/l7policies/8a1412f0-4c32-4257-8b07-af4770b604fd/rules invoked with unexpected marker=[%s]", marker)
		}
	})
}

// HandleRuleGetSuccessfully sets up the test server to respond to a rule Get request.
func HandleRuleGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/l7policies/8a1412f0-4c32-4257-8b07-af4770b604fd/rules/16621dbb-a736-4888-a57a-3ecd53df784c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, SingleRuleBody)
	})
}

// HandleRuleDeletionSuccessfully sets up the test server to respond to a rule deletion request.
func HandleRuleDeletionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/l7policies/8a1412f0-4c32-4257-8b07-af4770b604fd/rules/16621dbb-a736-4888-a57a-3ecd53df784c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// PostUpdateRuleBody is the canned response body of a Update request on an existing rule.
const PostUpdateRuleBody = `
{
	"rule": {
		"compare_type": "REGEX",
		"invert": false,
		"admin_state_up": true,
		"value": "/images/special*",
		"key": null,
		"project_id": "e3cd678b11784734bc366148aa37580e",
		"type": "PATH",
		"id": "16621dbb-a736-4888-a57a-3ecd53df784c"
	}
}
`

// HandleRuleUpdateSuccessfully sets up the test server to respond to a rule Update request.
func HandleRuleUpdateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/lbaas/l7policies/8a1412f0-4c32-4257-8b07-af4770b604fd/rules/16621dbb-a736-4888-a57a-3ecd53df784c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestJSONRequest(t, r, `{
			"rule": {
				"compare_type": "REGEX",
				"invert": false,
				"key": null,
				"type": "PATH",
				"value": "/images/special*"
			}
		}`)

		fmt.Fprintf(w, PostUpdateRuleBody)
	})
}
