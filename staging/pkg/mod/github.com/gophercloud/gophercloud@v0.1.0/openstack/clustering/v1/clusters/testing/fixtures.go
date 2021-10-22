package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/clustering/v1/clusters"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const ClusterResponse = `
{
  "cluster": {
    "config": {},
    "created_at": "2015-02-10T14:26:14Z",
    "data": {},
    "dependents": {},
    "desired_capacity": 3,
    "domain": null,
    "id": "7d85f602-a948-4a30-afd4-e84f47471c15",
    "init_at": "2015-02-10T15:26:14Z",
    "max_size": 20,
    "metadata": {},
    "min_size": 1,
    "name": "cluster1",
    "nodes": [
      "b07c57c8-7ab2-47bf-bdf8-e894c0c601b9",
      "ecc23d3e-bb68-48f8-8260-c9cf6bcb6e61",
      "da1e9c87-e584-4626-a120-022da5062dac"
    ],
    "policies": [],
    "profile_id": "edc63d0a-2ca4-48fa-9854-27926da76a4a",
    "profile_name": "mystack",
    "project": "6e18cc2bdbeb48a5b3cad2dc499f6804",
    "status": "ACTIVE",
    "status_reason": "Cluster scale-in succeeded",
    "timeout": 3600,
    "updated_at": "2015-02-10T16:26:14Z",
    "user": "5e5bf8027826429c96af157f68dc9072"
  }
}`

var ExpectedCluster = clusters.Cluster{
	Config:          map[string]interface{}{},
	CreatedAt:       time.Date(2015, 2, 10, 14, 26, 14, 0, time.UTC),
	Data:            map[string]interface{}{},
	Dependents:      map[string]interface{}{},
	DesiredCapacity: 3,
	Domain:          "",
	ID:              "7d85f602-a948-4a30-afd4-e84f47471c15",
	InitAt:          time.Date(2015, 2, 10, 15, 26, 14, 0, time.UTC),
	MaxSize:         20,
	Metadata:        map[string]interface{}{},
	MinSize:         1,
	Name:            "cluster1",
	Nodes: []string{
		"b07c57c8-7ab2-47bf-bdf8-e894c0c601b9",
		"ecc23d3e-bb68-48f8-8260-c9cf6bcb6e61",
		"da1e9c87-e584-4626-a120-022da5062dac",
	},
	Policies:     []string{},
	ProfileID:    "edc63d0a-2ca4-48fa-9854-27926da76a4a",
	ProfileName:  "mystack",
	Project:      "6e18cc2bdbeb48a5b3cad2dc499f6804",
	Status:       "ACTIVE",
	StatusReason: "Cluster scale-in succeeded",
	Timeout:      3600,
	UpdatedAt:    time.Date(2015, 2, 10, 16, 26, 14, 0, time.UTC),
	User:         "5e5bf8027826429c96af157f68dc9072",
}

const ClusterResponse_EmptyTime = `
{
  "cluster": {
    "config": {},
    "created_at": null,
    "data": {},
    "dependents": {},
    "desired_capacity": 3,
    "domain": null,
    "id": "7d85f602-a948-4a30-afd4-e84f47471c15",
    "init_at": null,
    "max_size": 20,
    "metadata": {},
    "min_size": 1,
    "name": "cluster1",
    "nodes": [
      "b07c57c8-7ab2-47bf-bdf8-e894c0c601b9",
      "ecc23d3e-bb68-48f8-8260-c9cf6bcb6e61",
      "da1e9c87-e584-4626-a120-022da5062dac"
    ],
    "policies": [],
    "profile_id": "edc63d0a-2ca4-48fa-9854-27926da76a4a",
    "profile_name": "mystack",
    "project": "6e18cc2bdbeb48a5b3cad2dc499f6804",
    "status": "ACTIVE",
    "status_reason": "Cluster scale-in succeeded",
    "timeout": 3600,
    "updated_at": null,
    "user": "5e5bf8027826429c96af157f68dc9072"
  }
}`

var ExpectedCluster_EmptyTime = clusters.Cluster{
	Config:          map[string]interface{}{},
	Data:            map[string]interface{}{},
	Dependents:      map[string]interface{}{},
	DesiredCapacity: 3,
	Domain:          "",
	ID:              "7d85f602-a948-4a30-afd4-e84f47471c15",
	MaxSize:         20,
	Metadata:        map[string]interface{}{},
	MinSize:         1,
	Name:            "cluster1",
	Nodes: []string{
		"b07c57c8-7ab2-47bf-bdf8-e894c0c601b9",
		"ecc23d3e-bb68-48f8-8260-c9cf6bcb6e61",
		"da1e9c87-e584-4626-a120-022da5062dac",
	},
	Policies:     []string{},
	ProfileID:    "edc63d0a-2ca4-48fa-9854-27926da76a4a",
	ProfileName:  "mystack",
	Project:      "6e18cc2bdbeb48a5b3cad2dc499f6804",
	Status:       "ACTIVE",
	StatusReason: "Cluster scale-in succeeded",
	Timeout:      3600,
	User:         "5e5bf8027826429c96af157f68dc9072",
}

const ClusterResponse_Metadata = `
{
  "cluster": {
    "config": {},
    "created_at": "2015-02-10T14:26:14Z",
    "data": {},
    "dependents": {},
    "desired_capacity": 3,
    "domain": null,
    "id": "7d85f602-a948-4a30-afd4-e84f47471c15",
    "init_at": "2015-02-10T15:26:14Z",
    "max_size": 20,
    "metadata": {
      "test": {
        "nil_interface": null,
        "bool_value": false,
        "string_value": "test_string",
        "float_value": 123.3
      },
      "foo": "bar"
    },
    "min_size": 1,
    "name": "cluster1",
    "nodes": [
      "b07c57c8-7ab2-47bf-bdf8-e894c0c601b9",
      "ecc23d3e-bb68-48f8-8260-c9cf6bcb6e61",
      "da1e9c87-e584-4626-a120-022da5062dac"
    ],
    "policies": [],
    "profile_id": "edc63d0a-2ca4-48fa-9854-27926da76a4a",
    "profile_name": "mystack",
    "project": "6e18cc2bdbeb48a5b3cad2dc499f6804",
    "status": "ACTIVE",
    "status_reason": "Cluster scale-in succeeded",
    "timeout": 3600,
    "updated_at": "2015-02-10T16:26:14Z",
    "user": "5e5bf8027826429c96af157f68dc9072"
  }
}`

var ExpectedCluster_Metadata = clusters.Cluster{
	Config:          map[string]interface{}{},
	CreatedAt:       time.Date(2015, 2, 10, 14, 26, 14, 0, time.UTC),
	Data:            map[string]interface{}{},
	Dependents:      map[string]interface{}{},
	DesiredCapacity: 3,
	Domain:          "",
	ID:              "7d85f602-a948-4a30-afd4-e84f47471c15",
	InitAt:          time.Date(2015, 2, 10, 15, 26, 14, 0, time.UTC),
	MaxSize:         20,
	MinSize:         1,
	Metadata: map[string]interface{}{
		"foo": "bar",
		"test": map[string]interface{}{
			"nil_interface": interface{}(nil),
			"float_value":   float64(123.3),
			"string_value":  "test_string",
			"bool_value":    false,
		},
	},
	Name: "cluster1",
	Nodes: []string{
		"b07c57c8-7ab2-47bf-bdf8-e894c0c601b9",
		"ecc23d3e-bb68-48f8-8260-c9cf6bcb6e61",
		"da1e9c87-e584-4626-a120-022da5062dac",
	},
	Policies:     []string{},
	ProfileID:    "edc63d0a-2ca4-48fa-9854-27926da76a4a",
	ProfileName:  "mystack",
	Project:      "6e18cc2bdbeb48a5b3cad2dc499f6804",
	Status:       "ACTIVE",
	StatusReason: "Cluster scale-in succeeded",
	Timeout:      3600,
	UpdatedAt:    time.Date(2015, 2, 10, 16, 26, 14, 0, time.UTC),
	User:         "5e5bf8027826429c96af157f68dc9072",
}

const ListResponse = `
{
  "clusters": [
    {
      "config": {},
      "created_at": "2015-02-10T14:26:14Z",
      "data": {},
      "dependents": {},
      "desired_capacity": 3,
      "domain": null,
      "id": "7d85f602-a948-4a30-afd4-e84f47471c15",
      "init_at": "2015-02-10T15:26:14Z",
      "max_size": 20,
      "min_size": 1,
      "metadata": {},
      "name": "cluster1",
      "nodes": [
        "b07c57c8-7ab2-47bf-bdf8-e894c0c601b9",
        "ecc23d3e-bb68-48f8-8260-c9cf6bcb6e61",
        "da1e9c87-e584-4626-a120-022da5062dac"
      ],
      "policies": [],
      "profile_id": "edc63d0a-2ca4-48fa-9854-27926da76a4a",
      "profile_name": "mystack",
      "project": "6e18cc2bdbeb48a5b3cad2dc499f6804",
      "status": "ACTIVE",
      "status_reason": "Cluster scale-in succeeded",
      "timeout": 3600,
      "updated_at": "2015-02-10T16:26:14Z",
      "user": "5e5bf8027826429c96af157f68dc9072"
    },
		{
			"config": {},
			"created_at": null,
			"data": {},
			"dependents": {},
			"desired_capacity": 3,
			"domain": null,
			"id": "7d85f602-a948-4a30-afd4-e84f47471c15",
			"init_at": null,
			"max_size": 20,
			"metadata": {},
			"min_size": 1,
			"name": "cluster1",
			"nodes": [
				"b07c57c8-7ab2-47bf-bdf8-e894c0c601b9",
				"ecc23d3e-bb68-48f8-8260-c9cf6bcb6e61",
				"da1e9c87-e584-4626-a120-022da5062dac"
			],
			"policies": [],
			"profile_id": "edc63d0a-2ca4-48fa-9854-27926da76a4a",
			"profile_name": "mystack",
			"project": "6e18cc2bdbeb48a5b3cad2dc499f6804",
			"status": "ACTIVE",
			"status_reason": "Cluster scale-in succeeded",
			"timeout": 3600,
			"updated_at": null,
			"user": "5e5bf8027826429c96af157f68dc9072"
  	}
	]
}`

var ExpectedClusters = []clusters.Cluster{ExpectedCluster, ExpectedCluster_EmptyTime}

const UpdateResponse = `
{
  "cluster": {
    "config": {},
    "created_at": "2015-02-10T14:26:14Z",
    "data": {},
    "dependents": {},
    "desired_capacity": 4,
    "domain": null,
    "id": "7d85f602-a948-4a30-afd4-e84f47471c15",
    "init_at": "2015-02-10T15:26:14Z",
    "max_size": -1,
    "metadata": {},
    "min_size": 0,
    "name": "cluster1",
    "nodes": [
      "b07c57c8-7ab2-47bf-bdf8-e894c0c601b9",
      "ecc23d3e-bb68-48f8-8260-c9cf6bcb6e61",
      "da1e9c87-e584-4626-a120-022da5062dac"
    ],
    "policies": [],
    "profile_id": "edc63d0a-2ca4-48fa-9854-27926da76a4a",
    "profile_name": "profile1",
    "project": "6e18cc2bdbeb48a5b3cad2dc499f6804",
    "status": "ACTIVE",
    "status_reason": "Cluster scale-in succeeded",
    "timeout": 3600,
    "updated_at": "2015-02-10T16:26:14Z",
    "user": "5e5bf8027826429c96af157f68dc9072"
  }
}`

const UpdateResponse_EmptyTime = `
{
  "cluster": {
    "config": {},
    "created_at": null,
    "data": {},
    "dependents": {},
    "desired_capacity": 3,
    "domain": null,
    "id": "7d85f602-a948-4a30-afd4-e84f47471c15",
    "init_at": null,
    "max_size": 20,
    "metadata": {},
    "min_size": 1,
    "name": "cluster1",
    "nodes": [
      "b07c57c8-7ab2-47bf-bdf8-e894c0c601b9",
      "ecc23d3e-bb68-48f8-8260-c9cf6bcb6e61",
      "da1e9c87-e584-4626-a120-022da5062dac"
    ],
    "policies": [],
    "profile_id": "edc63d0a-2ca4-48fa-9854-27926da76a4a",
    "profile_name": "mystack",
    "project": "6e18cc2bdbeb48a5b3cad2dc499f6804",
    "status": "ACTIVE",
    "status_reason": "Cluster scale-in succeeded",
    "timeout": 3600,
    "updated_at": null,
    "user": "5e5bf8027826429c96af157f68dc9072"
  }
}`

const ActionResponse = `
{
  "action": "2a0ff107-e789-4660-a122-3816c43af703"
}`

const ExpectedActionID = "2a0ff107-e789-4660-a122-3816c43af703"

const OperationActionResponse = `
{
  "action": "2a0ff107-e789-4660-a122-3816c43af703"
}`

const OperationExpectedActionID = "2a0ff107-e789-4660-a122-3816c43af703"

const ListPoliciesResult = `{
  "cluster_policies": [
    {
      "cluster_id":   "7d85f602-a948-4a30-afd4-e84f47471c15",
      "cluster_name": "cluster4",
      "enabled":      true,
      "id":           "06be3a1f-b238-4a96-a737-ceec5714087e",
      "policy_id":    "714fe676-a08f-4196-b7af-61d52eeded15",
      "policy_name":  "dp01",
      "policy_type":  "senlin.policy.deletion-1.0"
    }
  ]
}`

var ExpectedClusterPolicy = clusters.ClusterPolicy{
	ClusterID:   "7d85f602-a948-4a30-afd4-e84f47471c15",
	ClusterName: "cluster4",
	Enabled:     true,
	ID:          "06be3a1f-b238-4a96-a737-ceec5714087e",
	PolicyID:    "714fe676-a08f-4196-b7af-61d52eeded15",
	PolicyName:  "dp01",
	PolicyType:  "senlin.policy.deletion-1.0",
}

var ExpectedListPolicies = []clusters.ClusterPolicy{ExpectedClusterPolicy}

const GetPolicyResponse = `
{
  "cluster_policy": {
    "cluster_id":   "7d85f602-a948-4a30-afd4-e84f47471c15",
    "cluster_name": "cluster4",
    "enabled":      true,
    "id":           "06be3a1f-b238-4a96-a737-ceec5714087e",
    "policy_id":    "714fe676-a08f-4196-b7af-61d52eeded15",
    "policy_name":  "dp01",
    "policy_type":  "senlin.policy.deletion-1.0"
  }
}`

const CollectResponse = `
{
  "cluster_attributes": [{
    "id": "foo",
    "value":   "bar"
  }
  ]	
}`

var ExpectedCollectAttributes = []clusters.ClusterAttributes{
	{
		ID:    "foo",
		Value: string("bar"),
	},
}

func HandleCreateClusterSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("X-OpenStack-Request-ID", "req-781e9bdc-4163-46eb-91c9-786c53188bbb")
		w.Header().Add("Location", "http://senlin.cloud.blizzard.net:8778/v1/actions/625628cd-f877-44be-bde0-fec79f84e13d")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ClusterResponse)
	})
}

func HandleCreateClusterEmptyTimeSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ClusterResponse_EmptyTime)
	})
}

func HandleCreateClusterMetadataSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ClusterResponse_Metadata)
	})
}

func HandleGetClusterSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/7d85f602-a948-4a30-afd4-e84f47471c15", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ClusterResponse)
	})
}

func HandleGetClusterEmptyTimeSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/7d85f602-a948-4a30-afd4-e84f47471c15", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ClusterResponse_EmptyTime)
	})
}

func HandleListClusterSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ListResponse)
	})
}

func HandleUpdateClusterSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/"+ExpectedCluster.ID, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ClusterResponse)
	})
}

func HandleUpdateClusterEmptyTimeSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/"+ExpectedCluster_EmptyTime.ID, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, UpdateResponse_EmptyTime)
	})
}

func HandleDeleteClusterSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/6dc6d336e3fc4c0a951b5698cd1236ee", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusNoContent)
	})
}

func HandleResizeSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/7d85f602-a948-4a30-afd4-e84f47471c15/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ActionResponse)
	})
}

func HandleScaleInSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/edce3528-864f-41fb-8759-f4707925cc09/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ActionResponse)
	})
}

func HandleScaleOutSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/edce3528-864f-41fb-8759-f4707925cc09/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ActionResponse)
	})
}

func HandleListPoliciesSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/7d85f602-a948-4a30-afd4-e84f47471c15/policies", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ListPoliciesResult)
	})
}

func HandleGetPolicySuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/7d85f602-a948-4a30-afd4-e84f47471c15/policies/714fe676-a08f-4196-b7af-61d52eeded15", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, GetPolicyResponse)
	})
}

func HandleRecoverSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/edce3528-864f-41fb-8759-f4707925cc09/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ActionResponse)
	})
}

func HandleAttachPolicySuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/7d85f602-a948-4a30-afd4-e84f47471c15/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ActionResponse)
	})
}

func HandleDetachPolicySuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/7d85f602-a948-4a30-afd4-e84f47471c15/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ActionResponse)
	})
}

func HandleUpdatePolicySuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/7d85f602-a948-4a30-afd4-e84f47471c15/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ActionResponse)
	})
}

func HandleCheckSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/edce3528-864f-41fb-8759-f4707925cc09/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ActionResponse)
	})
}

func HandleLifecycleSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/edce3528-864f-41fb-8759-f4707925cc09/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("X-OpenStack-Request-ID", "req-781e9bdc-4163-46eb-91c9-786c53188bbb")
		w.Header().Add("Location", "http://senlin.cloud.blizzard.net:8778/v1/actions/2a0ff107-e789-4660-a122-3816c43af703")
		w.WriteHeader(http.StatusAccepted)

		fmt.Fprint(w, ActionResponse)
	})
}

func HandleAddNodesSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/7d85f602-a948-4a30-afd4-e84f47471c15/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("X-OpenStack-Request-ID", "req-781e9bdc-4163-46eb-91c9-786c53188bbb")
		w.WriteHeader(http.StatusAccepted)

		fmt.Fprint(w, ActionResponse)
	})
}

func HandleRemoveNodesSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/7d85f602-a948-4a30-afd4-e84f47471c15/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("X-OpenStack-Request-ID", "req-781e9bdc-4163-46eb-91c9-786c53188bbb")
		w.WriteHeader(http.StatusAccepted)
		fmt.Fprint(w, ActionResponse)
	})
}

func HandleReplaceNodeSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/7d85f602-a948-4a30-afd4-e84f47471c15/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("X-OpenStack-Request-ID", "req-781e9bdc-4163-46eb-91c9-786c53188bbb")
		w.WriteHeader(http.StatusAccepted)
		fmt.Fprint(w, ActionResponse)
	})
}

func HandleClusterCollectSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/7d85f602-a948-4a30-afd4-e84f47471c15/attrs/foo.bar", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("X-OpenStack-Request-ID", "req-781e9bdc-4163-46eb-91c9-786c53188bbb")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, CollectResponse)
	})
}

func HandleOpsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clusters/7d85f602-a948-4a30-afd4-e84f47471c15/ops", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)

		fmt.Fprint(w, OperationActionResponse)
	})
}
