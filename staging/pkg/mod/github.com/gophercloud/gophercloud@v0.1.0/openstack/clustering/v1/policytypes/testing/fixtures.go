package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/clustering/v1/policytypes"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const FakePolicyTypetoGet = "fake-policytype"

const PolicyTypeBody = `
{
	"policy_types": [
		{
			"name": "senlin.policy.affinity",
			"version": "1.0",
			"support_status": {
				"1.0": [
					{
						"status": "SUPPORTED",
						"since": "2016.10"
					}
				]
			}
		},
		{
			"name": "senlin.policy.health",
			"version": "1.0",
			"support_status": {
				"1.0": [
					{
						"status": "EXPERIMENTAL",
						"since": "2016.10"
					}
				]
			}
		},
		{
			"name": "senlin.policy.scaling",
			"version": "1.0",
			"support_status": {
				"1.0": [
					{
						"status": "SUPPORTED",
						"since": "2016.04"
					}
				]
			}
		},
		{
			"name": "senlin.policy.region_placement",
			"version": "1.0",
			"support_status": {
				"1.0": [
					{
						"status": "EXPERIMENTAL",
						"since": "2016.04"
					},
					{
						"status": "SUPPORTED",
						"since": "2016.10"
					}
				]
			}
		}
	]
}
`

const PolicyTypeDetailBody = `
{
    "policy_type": {
    	"name": "senlin.policy.batch-1.0",
		"schema": {
		  "max_batch_size": {
			"default": -1,
			"description": "Maximum number of nodes that will be updated in parallel.",
			"required": false,
			"type": "Integer",
			"updatable": false
		  },
		  "min_in_service": {
			"default": 1,
			"description": "Minimum number of nodes in service when performing updates.",
			"required": false,
			"type": "Integer",
			"updatable": false
		  },
		  "pause_time": {
			"default": 60,
			"description": "Interval in seconds between update batches if any.",
			"required": false,
			"type": "Integer",
			"updatable": false
		  }
		},
		"support_status": {
			"1.0": [
			  {
				"status": "EXPERIMENTAL",
				"since": "2017.02"
			  }
			]
		}
    }
}
`

var ExpectedPolicyType1 = policytypes.PolicyType{
	Name:    "senlin.policy.affinity",
	Version: "1.0",
	SupportStatus: map[string][]policytypes.SupportStatusType{
		"1.0": {
			{
				Status: "SUPPORTED",
				Since:  "2016.10",
			},
		},
	},
}

var ExpectedPolicyType2 = policytypes.PolicyType{
	Name:    "senlin.policy.health",
	Version: "1.0",
	SupportStatus: map[string][]policytypes.SupportStatusType{
		"1.0": {
			{
				Status: "EXPERIMENTAL",
				Since:  "2016.10",
			},
		},
	},
}

var ExpectedPolicyType3 = policytypes.PolicyType{
	Name:    "senlin.policy.scaling",
	Version: "1.0",
	SupportStatus: map[string][]policytypes.SupportStatusType{
		"1.0": {
			{
				Status: "SUPPORTED",
				Since:  "2016.04",
			},
		},
	},
}

var ExpectedPolicyType4 = policytypes.PolicyType{
	Name:    "senlin.policy.region_placement",
	Version: "1.0",
	SupportStatus: map[string][]policytypes.SupportStatusType{
		"1.0": {
			{
				Status: "EXPERIMENTAL",
				Since:  "2016.04",
			},
			{
				Status: "SUPPORTED",
				Since:  "2016.10",
			},
		},
	},
}

var ExpectedPolicyTypes = []policytypes.PolicyType{
	ExpectedPolicyType1,
	ExpectedPolicyType2,
	ExpectedPolicyType3,
	ExpectedPolicyType4,
}

var ExpectedPolicyTypeDetail = &policytypes.PolicyTypeDetail{
	Name: "senlin.policy.batch-1.0",
	Schema: map[string]interface{}{
		"max_batch_size": map[string]interface{}{
			"default":     float64(-1),
			"description": "Maximum number of nodes that will be updated in parallel.",
			"required":    false,
			"type":        "Integer",
			"updatable":   false,
		},
		"min_in_service": map[string]interface{}{
			"default":     float64(1),
			"description": "Minimum number of nodes in service when performing updates.",
			"required":    false,
			"type":        "Integer",
			"updatable":   false,
		},
		"pause_time": map[string]interface{}{
			"default":     float64(60),
			"description": "Interval in seconds between update batches if any.",
			"required":    false,
			"type":        "Integer",
			"updatable":   false,
		},
	},
	SupportStatus: map[string][]policytypes.SupportStatusType{
		"1.0": []policytypes.SupportStatusType{
			{
				Status: "EXPERIMENTAL",
				Since:  "2017.02",
			},
		},
	},
}

func HandlePolicyTypeList(t *testing.T) {
	th.Mux.HandleFunc("/v1/policy-types",
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "GET")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

			w.Header().Add("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)

			fmt.Fprintf(w, PolicyTypeBody)
		})
}

func HandlePolicyTypeGet(t *testing.T) {
	th.Mux.HandleFunc("/v1/policy-types/"+FakePolicyTypetoGet,
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "GET")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

			w.Header().Add("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)

			fmt.Fprintf(w, PolicyTypeDetailBody)
		})
}
