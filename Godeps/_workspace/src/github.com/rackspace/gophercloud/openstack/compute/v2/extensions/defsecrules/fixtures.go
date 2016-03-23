package defsecrules

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

const rootPath = "/os-security-group-default-rules"

func mockListRulesResponse(t *testing.T) {
	th.Mux.HandleFunc(rootPath, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
  "security_group_default_rules": [
    {
      "from_port": 80,
      "id": "{ruleID}",
      "ip_protocol": "TCP",
      "ip_range": {
        "cidr": "10.10.10.0/24"
      },
      "to_port": 80
    }
  ]
}
      `)
	})
}

func mockCreateRuleResponse(t *testing.T) {
	th.Mux.HandleFunc(rootPath, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
  "security_group_default_rule": {
    "ip_protocol": "TCP",
    "from_port": 80,
    "to_port": 80,
    "cidr": "10.10.12.0/24"
  }
}
	`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
  "security_group_default_rule": {
    "from_port": 80,
    "id": "{ruleID}",
    "ip_protocol": "TCP",
    "ip_range": {
      "cidr": "10.10.12.0/24"
    },
    "to_port": 80
  }
}
`)
	})
}

func mockCreateRuleResponseICMPZero(t *testing.T) {
	th.Mux.HandleFunc(rootPath, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
  "security_group_default_rule": {
    "ip_protocol": "ICMP",
    "from_port": 0,
    "to_port": 0,
    "cidr": "10.10.12.0/24"
  }
}
	`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
  "security_group_default_rule": {
    "from_port": 0,
    "id": "{ruleID}",
    "ip_protocol": "ICMP",
    "ip_range": {
      "cidr": "10.10.12.0/24"
    },
    "to_port": 0
  }
}
`)
	})
}

func mockGetRuleResponse(t *testing.T, ruleID string) {
	url := rootPath + "/" + ruleID
	th.Mux.HandleFunc(url, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
  "security_group_default_rule": {
    "id": "{ruleID}",
    "from_port": 80,
    "to_port": 80,
    "ip_protocol": "TCP",
    "ip_range": {
      "cidr": "10.10.12.0/24"
    }
  }
}
			`)
	})
}

func mockDeleteRuleResponse(t *testing.T, ruleID string) {
	url := rootPath + "/" + ruleID
	th.Mux.HandleFunc(url, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusNoContent)
	})
}
