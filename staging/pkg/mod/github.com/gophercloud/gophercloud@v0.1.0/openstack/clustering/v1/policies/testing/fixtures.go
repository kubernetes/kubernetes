package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/clustering/v1/policies"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const PolicyListBody1 = `
{
  "policies": [
    {
      "created_at": "2018-04-02T21:43:30.000000",
      "data": {},
      "domain": null,
      "id": "PolicyListBodyID1",
      "name": "delpol",
      "project": "018cd0909fb44cd5bc9b7a3cd664920e",
      "spec": {
        "description": "A policy for choosing victim node(s) from a cluster for deletion.",
        "properties": {
          "criteria": "OLDEST_FIRST",
          "destroy_after_deletion": true,
          "grace_period": 60,
          "reduce_desired_capacity": false
        },
        "type": "senlin.policy.deletion",
        "version": 1
      },
      "type": "senlin.policy.deletion-1.0",
      "updated_at": "2018-04-02T00:19:12Z",
      "user": "fe43e41739154b72818565e0d2580819"
    }
  ]
}
`

const PolicyListBody2 = `
{
  "policies": [
    {
      "created_at": "2018-04-02T22:29:36.000000",
      "data": {},
      "domain": null,
      "id": "PolicyListBodyID2",
      "name": "delpol2",
      "project": "018cd0909fb44cd5bc9b7a3cd664920e",
      "spec": {
        "description": "A policy for choosing victim node(s) from a cluster for deletion.",
        "properties": {
          "criteria": "OLDEST_FIRST",
          "destroy_after_deletion": true,
          "grace_period": 60,
          "reduce_desired_capacity": false
        },
        "type": "senlin.policy.deletion",
        "version": "1.0"
      },
      "type": "senlin.policy.deletion-1.0",
      "updated_at": "2018-04-02T23:15:11.000000",
      "user": "fe43e41739154b72818565e0d2580819"
    }
  ]
}
`

const PolicyCreateBody = `
{
  "policy": {
    "created_at": "2018-04-04T00:18:36Z",
    "data": {},
    "domain": null,
    "id": "b99b3ab4-3aa6-4fba-b827-69b88b9c544a",
    "name": "delpol4",
    "project": "018cd0909fb44cd5bc9b7a3cd664920e",
    "spec": {
      "description": "A policy for choosing victim node(s) from a cluster for deletion.",
      "properties": {
        "hooks": {
          "params": {
            "queue": "zaqar_queue_name"
          },
          "timeout": 180,
          "type": "zaqar"
        }
      },
      "type": "senlin.policy.deletion",
      "version": 1.1
    },
    "type": "senlin.policy.deletion-1.1",
    "updated_at": null,
    "user": "fe43e41739154b72818565e0d2580819"
  }
}
`

const PolicyGetBody = `
{
  "policy": {
      "created_at": "2018-04-02T21:43:30.000000",
      "data": {},
      "domain": null,
      "id": "b99b3ab4-3aa6-4fba-b827-69b88b9c544a",
      "name": "delpol",
      "project": "018cd0909fb44cd5bc9b7a3cd664920e",
      "spec": {
        "description": "A policy for choosing victim node(s) from a cluster for deletion.",
        "properties": {
          "criteria": "OLDEST_FIRST",
          "destroy_after_deletion": true,
          "grace_period": 60,
          "reduce_desired_capacity": false
        },
        "type": "senlin.policy.deletion",
        "version": 1
      },
      "type": "senlin.policy.deletion-1.0",
      "updated_at": "2018-04-02T00:19:12Z",
      "user": "fe43e41739154b72818565e0d2580819"
  }
}
`

const PolicyUpdateBody = `
{
  "policy": {
    "created_at": "2018-04-02T21:43:30.000000",
    "data": {},
    "domain": null,
    "id": "b99b3ab4-3aa6-4fba-b827-69b88b9c544a",
    "name": "delpol4",
    "project": "018cd0909fb44cd5bc9b7a3cd664920e",
    "spec": {
      "description": "A policy for choosing victim node(s) from a cluster for deletion.",
      "properties": {
        "hooks": {
          "params": {
            "queue": "zaqar_queue_name"
          },
          "timeout": 180,
          "type": "zaqar"
        }
      },
      "type": "senlin.policy.deletion",
      "version": 1.1
    },
    "type": "senlin.policy.deletion-1.1",
    "updated_at": null,
    "user": "fe43e41739154b72818565e0d2580819"
  }
}
`

const PolicyBadUpdateBody = `
{
  "policy": {
    "created_at": "invalid",
    "data": {},
    "domain": null,
    "id": "b99b3ab4-3aa6-4fba-b827-69b88b9c544a",
    "name": "delpol4",
    "project": "018cd0909fb44cd5bc9b7a3cd664920e",
    "spec": {
      "description": "A policy for choosing victim node(s) from a cluster for deletion.",
      "properties": {
        "hooks": {
          "params": {
            "queue": "zaqar_queue_name"
          },
          "timeout": 180,
          "type": "zaqar"
        }
      },
      "type": "senlin.policy.deletion",
      "version": 1.1
    },
    "type": "invalid",
    "updated_at": null,
    "user": "fe43e41739154b72818565e0d2580819"
  }
}
`

const PolicyValidateBody = `
{
  "policy": {
    "created_at": "2018-04-02T21:43:30.000000",
    "data": {},
    "domain": null,
    "id": "b99b3ab4-3aa6-4fba-b827-69b88b9c544a",
    "name": "delpol4",
    "project": "018cd0909fb44cd5bc9b7a3cd664920e",
    "spec": {
      "description": "A policy for choosing victim node(s) from a cluster for deletion.",
      "properties": {
        "hooks": {
          "params": {
            "queue": "zaqar_queue_name"
          },
          "timeout": 180,
          "type": "zaqar"
        }
      },
      "type": "senlin.policy.deletion",
      "version": 1.1
    },
    "type": "senlin.policy.deletion-1.1",
    "updated_at": null,
    "user": "fe43e41739154b72818565e0d2580819"
  }
}
`

const PolicyBadValidateBody = `
{
  "policy": {
    "created_at": "invalid",
    "data": {},
    "domain": null,
    "id": "b99b3ab4-3aa6-4fba-b827-69b88b9c544a",
    "name": "delpol4",
    "project": "018cd0909fb44cd5bc9b7a3cd664920e",
    "spec": {
      "description": "A policy for choosing victim node(s) from a cluster for deletion.",
      "properties": {
        "hooks": {
          "params": {
            "queue": "zaqar_queue_name"
          },
          "timeout": 180,
          "type": "zaqar"
        }
      },
      "type": "senlin.policy.deletion",
      "version": 1.1
    },
    "type": "invalid",
    "updated_at": null,
    "user": "fe43e41739154b72818565e0d2580819"
  }
}
`

const PolicyDeleteRequestID = "req-7328d1b0-9945-456f-b2cd-5166b77d14a8"
const PolicyIDtoUpdate = "b99b3ab4-3aa6-4fba-b827-69b88b9c544a"
const PolicyIDtoGet = "b99b3ab4-3aa6-4fba-b827-69b88b9c544a"
const PolicyIDtoDelete = "1"

var ExpectedPolicy1 = policies.Policy{
	CreatedAt: time.Date(2018, 4, 2, 21, 43, 30, 0, time.UTC),
	Data:      map[string]interface{}{},
	Domain:    "",
	ID:        "PolicyListBodyID1",
	Name:      "delpol",
	Project:   "018cd0909fb44cd5bc9b7a3cd664920e",

	Spec: policies.Spec{
		Description: "A policy for choosing victim node(s) from a cluster for deletion.",
		Properties: map[string]interface{}{
			"criteria":                "OLDEST_FIRST",
			"destroy_after_deletion":  true,
			"grace_period":            float64(60),
			"reduce_desired_capacity": false,
		},
		Type:    "senlin.policy.deletion",
		Version: "1.0",
	},
	Type:      "senlin.policy.deletion-1.0",
	User:      "fe43e41739154b72818565e0d2580819",
	UpdatedAt: time.Date(2018, 4, 2, 0, 19, 12, 0, time.UTC),
}

var ExpectedPolicy2 = policies.Policy{
	CreatedAt: time.Date(2018, 4, 2, 22, 29, 36, 0, time.UTC),
	Data:      map[string]interface{}{},
	Domain:    "",
	ID:        "PolicyListBodyID2",
	Name:      "delpol2",
	Project:   "018cd0909fb44cd5bc9b7a3cd664920e",

	Spec: policies.Spec{
		Description: "A policy for choosing victim node(s) from a cluster for deletion.",
		Properties: map[string]interface{}{
			"criteria":                "OLDEST_FIRST",
			"destroy_after_deletion":  true,
			"grace_period":            float64(60),
			"reduce_desired_capacity": false,
		},
		Type:    "senlin.policy.deletion",
		Version: "1.0",
	},
	Type:      "senlin.policy.deletion-1.0",
	User:      "fe43e41739154b72818565e0d2580819",
	UpdatedAt: time.Date(2018, 4, 2, 23, 15, 11, 0, time.UTC),
}

var ExpectedPolicies = [][]policies.Policy{
	[]policies.Policy{ExpectedPolicy1},
	[]policies.Policy{ExpectedPolicy2},
}

var ExpectedCreatePolicy = policies.Policy{
	CreatedAt: time.Date(2018, 4, 4, 0, 18, 36, 0, time.UTC),
	Data:      map[string]interface{}{},
	Domain:    "",
	ID:        "b99b3ab4-3aa6-4fba-b827-69b88b9c544a",
	Name:      "delpol4",
	Project:   "018cd0909fb44cd5bc9b7a3cd664920e",

	Spec: policies.Spec{
		Description: "A policy for choosing victim node(s) from a cluster for deletion.",
		Properties: map[string]interface{}{
			"hooks": map[string]interface{}{
				"params": map[string]interface{}{
					"queue": "zaqar_queue_name",
				},
				"timeout": float64(180),
				"type":    "zaqar",
			},
		},
		Type:    "senlin.policy.deletion",
		Version: "1.1",
	},
	Type: "senlin.policy.deletion-1.1",
	User: "fe43e41739154b72818565e0d2580819",
}

var ExpectedGetPolicy = policies.Policy{
	CreatedAt: time.Date(2018, 4, 2, 21, 43, 30, 0, time.UTC),
	Data:      map[string]interface{}{},
	Domain:    "",
	ID:        "b99b3ab4-3aa6-4fba-b827-69b88b9c544a",
	Name:      "delpol",
	Project:   "018cd0909fb44cd5bc9b7a3cd664920e",

	Spec: policies.Spec{
		Description: "A policy for choosing victim node(s) from a cluster for deletion.",
		Properties: map[string]interface{}{
			"criteria":                "OLDEST_FIRST",
			"destroy_after_deletion":  true,
			"grace_period":            float64(60),
			"reduce_desired_capacity": false,
		},
		Type:    "senlin.policy.deletion",
		Version: "1.0",
	},
	Type:      "senlin.policy.deletion-1.0",
	User:      "fe43e41739154b72818565e0d2580819",
	UpdatedAt: time.Date(2018, 4, 2, 0, 19, 12, 0, time.UTC),
}

var ExpectedUpdatePolicy = policies.Policy{
	CreatedAt: time.Date(2018, 4, 2, 21, 43, 30, 0, time.UTC),
	Data:      map[string]interface{}{},
	Domain:    "",
	ID:        "b99b3ab4-3aa6-4fba-b827-69b88b9c544a",
	Name:      "delpol4",
	Project:   "018cd0909fb44cd5bc9b7a3cd664920e",

	Spec: policies.Spec{
		Description: "A policy for choosing victim node(s) from a cluster for deletion.",
		Properties: map[string]interface{}{
			"hooks": map[string]interface{}{
				"params": map[string]interface{}{
					"queue": "zaqar_queue_name",
				},
				"timeout": float64(180),
				"type":    "zaqar",
			},
		},
		Type:    "senlin.policy.deletion",
		Version: "1.1",
	},
	Type: "senlin.policy.deletion-1.1",
	User: "fe43e41739154b72818565e0d2580819",
}

var ExpectedValidatePolicy = policies.Policy{
	CreatedAt: time.Date(2018, 4, 2, 21, 43, 30, 0, time.UTC),
	Data:      map[string]interface{}{},
	Domain:    "",
	ID:        "b99b3ab4-3aa6-4fba-b827-69b88b9c544a",
	Name:      "delpol4",
	Project:   "018cd0909fb44cd5bc9b7a3cd664920e",

	Spec: policies.Spec{
		Description: "A policy for choosing victim node(s) from a cluster for deletion.",
		Properties: map[string]interface{}{
			"hooks": map[string]interface{}{
				"params": map[string]interface{}{
					"queue": "zaqar_queue_name",
				},
				"timeout": float64(180),
				"type":    "zaqar",
			},
		},
		Type:    "senlin.policy.deletion",
		Version: "1.1",
	},
	Type: "senlin.policy.deletion-1.1",
	User: "fe43e41739154b72818565e0d2580819",
}

func HandlePolicyList(t *testing.T) {
	th.Mux.HandleFunc("/v1/policies", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, PolicyListBody1)
		case "PolicyListBodyID1":
			fmt.Fprintf(w, PolicyListBody2)
		case "PolicyListBodyID2":
			fmt.Fprintf(w, `{"policies":[]}`)
		default:
			t.Fatalf("Unexpected marker: [%s]", marker)
		}
	})
}

func HandlePolicyCreate(t *testing.T) {
	th.Mux.HandleFunc("/v1/policies", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, PolicyCreateBody)
	})
}

func HandlePolicyDelete(t *testing.T) {
	th.Mux.HandleFunc("/v1/policies/"+PolicyIDtoDelete, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("X-OpenStack-Request-Id", PolicyDeleteRequestID)
		w.WriteHeader(http.StatusNoContent)
	})
}

func HandlePolicyGet(t *testing.T) {
	th.Mux.HandleFunc("/v1/policies/"+PolicyIDtoGet,
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "GET")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

			w.Header().Add("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)

			fmt.Fprintf(w, PolicyGetBody)
		})
}

func HandlePolicyUpdate(t *testing.T) {
	th.Mux.HandleFunc("/v1/policies/"+PolicyIDtoUpdate, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, PolicyUpdateBody)
	})
}

func HandleBadPolicyUpdate(t *testing.T) {
	th.Mux.HandleFunc("/v1/policies/"+PolicyIDtoUpdate, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, PolicyBadUpdateBody)
	})
}

func HandlePolicyValidate(t *testing.T) {
	th.Mux.HandleFunc("/v1/policies/validate", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, PolicyValidateBody)
	})
}

func HandleBadPolicyValidate(t *testing.T) {
	th.Mux.HandleFunc("/v1/policies/validate", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, PolicyBadValidateBody)
	})
}
