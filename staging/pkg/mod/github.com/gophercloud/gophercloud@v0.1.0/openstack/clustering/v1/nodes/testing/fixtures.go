package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/clustering/v1/nodes"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const CreateResponse = `{
  "node": {
    "cluster_id": "e395be1e-8d8e-43bb-bd6c-943eccf76a6d",
    "created_at": "2016-05-13T07:02:20Z",
    "data": {
      "internal_ports": [
        {
          "network_id": "847e4f65-1ff1-42b1-9e74-74e6a109ad11",
          "security_group_ids": ["8db277ab-1d98-4148-ba72-724721789427"],
          "fixed_ips": [
            {
              "subnet_id": "863b20c0-c011-4650-85c2-ad531f4570a4",
              "ip_address": "10.63.177.162"
            }
          ],
          "id": "43aa53d7-a70b-4f40-812f-4feecb687018",
          "remove": true
        }
      ],
      "placement": {
        "zone": "nova"
      }
    },
    "dependents": {},
    "domain": "1235be1e-8d8e-43bb-bd6c-943eccf76a6d",
    "id": "82fe28e0-9fcb-42ca-a2fa-6eb7dddd75a1",
    "index": 2,
    "init_at": "2016-05-13T08:02:04Z",
    "metadata": {
      "test": {
        "nil_interface": null,
        "bool_value": false,
        "string_value": "test_string",
        "float_value": 123.3
      },
      "foo": "bar"
    },
    "name": "node-e395be1e-002",
    "physical_id": "66a81d68-bf48-4af5-897b-a3bfef7279a8",
    "profile_id": "d8a48377-f6a3-4af4-bbbb-6e8bcaa0cbc0",
    "profile_name": "pcirros",
    "project": "eee0b7c083e84501bdd50fb269d2a10e",
    "role": "",
    "status": "ACTIVE",
    "status_reason": "Creation succeeded",
    "updated_at": null,
    "user": "ab79b9647d074e46ac223a8fa297b846"
  }
}`

var ExpectedCreate = nodes.Node{
	ClusterID: "e395be1e-8d8e-43bb-bd6c-943eccf76a6d",
	CreatedAt: time.Date(2016, 5, 13, 7, 2, 20, 0, time.UTC),
	Data: map[string]interface{}{
		"internal_ports": []map[string]interface{}{
			{
				"network_id": "847e4f65-1ff1-42b1-9e74-74e6a109ad11",
				"security_group_ids": []interface{}{
					"8db277ab-1d98-4148-ba72-724721789427",
				},
				"fixed_ips": []interface{}{
					map[string]interface{}{
						"subnet_id":  "863b20c0-c011-4650-85c2-ad531f4570a4",
						"ip_address": "10.63.177.162",
					},
				},
				"id":     "43aa53d7-a70b-4f40-812f-4feecb687018",
				"remove": true,
			},
		},
		"placement": map[string]interface{}{
			"zone": "nova",
		},
	},
	Dependents: map[string]interface{}{},
	Domain:     "1235be1e-8d8e-43bb-bd6c-943eccf76a6d",
	ID:         "82fe28e0-9fcb-42ca-a2fa-6eb7dddd75a1",
	Index:      2,
	InitAt:     time.Date(2016, 5, 13, 8, 2, 4, 0, time.UTC),
	Metadata: map[string]interface{}{
		"foo": "bar",
		"test": map[string]interface{}{
			"nil_interface": interface{}(nil),
			"float_value":   float64(123.3),
			"string_value":  "test_string",
			"bool_value":    false,
		},
	},
	Name:         "node-e395be1e-002",
	PhysicalID:   "66a81d68-bf48-4af5-897b-a3bfef7279a8",
	ProfileID:    "d8a48377-f6a3-4af4-bbbb-6e8bcaa0cbc0",
	ProfileName:  "pcirros",
	Project:      "eee0b7c083e84501bdd50fb269d2a10e",
	Role:         "",
	Status:       "ACTIVE",
	StatusReason: "Creation succeeded",
	User:         "ab79b9647d074e46ac223a8fa297b846",
}

const ListResponse = `
{
  "nodes": [
    {
      "cluster_id": "e395be1e-8d8e-43bb-bd6c-943eccf76a6d",
      "created_at": "2016-05-13T07:02:20Z",
      "data": {},
      "dependents": {},
      "domain": null,
      "id": "82fe28e0-9fcb-42ca-a2fa-6eb7dddd75a1",
      "index": 2,
      "init_at": "2016-05-13T08:02:04Z",
      "metadata": {},
      "name": "node-e395be1e-002",
      "physical_id": "66a81d68-bf48-4af5-897b-a3bfef7279a8",
      "profile_id": "d8a48377-f6a3-4af4-bbbb-6e8bcaa0cbc0",
      "profile_name": "pcirros",
      "project": "eee0b7c083e84501bdd50fb269d2a10e",
      "role": "",
      "status": "ACTIVE",
      "status_reason": "Creation succeeded",
      "updated_at": "2016-05-13T09:02:04Z",
      "user": "ab79b9647d074e46ac223a8fa297b846"        }
  ]
}`

var ExpectedList1 = nodes.Node{
	ClusterID:    "e395be1e-8d8e-43bb-bd6c-943eccf76a6d",
	CreatedAt:    time.Date(2016, 5, 13, 7, 2, 20, 0, time.UTC),
	Data:         map[string]interface{}{},
	Dependents:   map[string]interface{}{},
	Domain:       "",
	ID:           "82fe28e0-9fcb-42ca-a2fa-6eb7dddd75a1",
	Index:        2,
	InitAt:       time.Date(2016, 5, 13, 8, 2, 4, 0, time.UTC),
	Metadata:     map[string]interface{}{},
	Name:         "node-e395be1e-002",
	PhysicalID:   "66a81d68-bf48-4af5-897b-a3bfef7279a8",
	ProfileID:    "d8a48377-f6a3-4af4-bbbb-6e8bcaa0cbc0",
	ProfileName:  "pcirros",
	Project:      "eee0b7c083e84501bdd50fb269d2a10e",
	Role:         "",
	Status:       "ACTIVE",
	StatusReason: "Creation succeeded",
	UpdatedAt:    time.Date(2016, 5, 13, 9, 2, 4, 0, time.UTC),
	User:         "ab79b9647d074e46ac223a8fa297b846",
}

var ExpectedList = []nodes.Node{ExpectedList1}

const GetResponse = `
{
  "node": {
    "cluster_id": "e395be1e-8d8e-43bb-bd6c-943eccf76a6d",
    "created_at": "2016-05-13T07:02:20Z",
    "data": {},
    "dependents": {},
    "domain": null,
    "id": "82fe28e0-9fcb-42ca-a2fa-6eb7dddd75a1",
    "index": 2,
    "init_at": "2016-05-13T07:02:04Z",
    "metadata": {"foo": "bar"},
    "name": "node-e395be1e-002",
    "physical_id": "66a81d68-bf48-4af5-897b-a3bfef7279a8",
    "profile_id": "d8a48377-f6a3-4af4-bbbb-6e8bcaa0cbc0",
    "profile_name": "pcirros",
    "project": "eee0b7c083e84501bdd50fb269d2a10e",
    "role": "",
    "status": "ACTIVE",
    "status_reason": "Creation succeeded",
    "updated_at": "2016-05-13T07:02:20Z",
    "user": "ab79b9647d074e46ac223a8fa297b846"
  }
}`

var ExpectedGet = nodes.Node{
	ClusterID:    "e395be1e-8d8e-43bb-bd6c-943eccf76a6d",
	CreatedAt:    time.Date(2016, 5, 13, 7, 2, 20, 0, time.UTC),
	Data:         map[string]interface{}{},
	Dependents:   map[string]interface{}{},
	Domain:       "",
	ID:           "82fe28e0-9fcb-42ca-a2fa-6eb7dddd75a1",
	Index:        2,
	InitAt:       time.Date(2016, 5, 13, 7, 2, 4, 0, time.UTC),
	Metadata:     map[string]interface{}{"foo": "bar"},
	Name:         "node-e395be1e-002",
	PhysicalID:   "66a81d68-bf48-4af5-897b-a3bfef7279a8",
	ProfileID:    "d8a48377-f6a3-4af4-bbbb-6e8bcaa0cbc0",
	ProfileName:  "pcirros",
	Project:      "eee0b7c083e84501bdd50fb269d2a10e",
	Role:         "",
	Status:       "ACTIVE",
	StatusReason: "Creation succeeded",
	UpdatedAt:    time.Date(2016, 5, 13, 7, 2, 20, 0, time.UTC),
	User:         "ab79b9647d074e46ac223a8fa297b846",
}

const UpdateResponse = `
{
  "node": {
    "cluster_id": "e395be1e-8d8e-43bb-bd6c-943eccf76a6d",
    "created_at": "2016-05-13T07:02:20Z",
    "data": {},
    "dependents": {},
    "domain": null,
    "id": "82fe28e0-9fcb-42ca-a2fa-6eb7dddd75a1",
    "index": 2,
    "init_at": "2016-05-13T08:02:04Z",
    "metadata": {"foo":"bar"},
    "name": "node-e395be1e-002",
    "physical_id": "66a81d68-bf48-4af5-897b-a3bfef7279a8",
    "profile_id": "d8a48377-f6a3-4af4-bbbb-6e8bcaa0cbc0",
    "profile_name": "pcirros",
    "project": "eee0b7c083e84501bdd50fb269d2a10e",
    "role": "",
    "status": "ACTIVE",
    "status_reason": "Creation succeeded",
    "updated_at": "2016-05-13T09:02:04Z",
    "user": "ab79b9647d074e46ac223a8fa297b846"
  }
}`

var ExpectedUpdate = nodes.Node{
	ClusterID:    "e395be1e-8d8e-43bb-bd6c-943eccf76a6d",
	CreatedAt:    time.Date(2016, 5, 13, 7, 2, 20, 0, time.UTC),
	Data:         map[string]interface{}{},
	Dependents:   map[string]interface{}{},
	Domain:       "",
	ID:           "82fe28e0-9fcb-42ca-a2fa-6eb7dddd75a1",
	Index:        2,
	InitAt:       time.Date(2016, 5, 13, 8, 2, 4, 0, time.UTC),
	Metadata:     map[string]interface{}{"foo": "bar"},
	Name:         "node-e395be1e-002",
	PhysicalID:   "66a81d68-bf48-4af5-897b-a3bfef7279a8",
	ProfileID:    "d8a48377-f6a3-4af4-bbbb-6e8bcaa0cbc0",
	ProfileName:  "pcirros",
	Project:      "eee0b7c083e84501bdd50fb269d2a10e",
	Role:         "",
	Status:       "ACTIVE",
	StatusReason: "Creation succeeded",
	UpdatedAt:    time.Date(2016, 5, 13, 9, 2, 4, 0, time.UTC),
	User:         "ab79b9647d074e46ac223a8fa297b846",
}

const OperationActionResponse = `
{
  "action": "2a0ff107-e789-4660-a122-3816c43af703"
}`

const OperationExpectedActionID = "2a0ff107-e789-4660-a122-3816c43af703"

const ActionResponse = `
{
  "action": "2a0ff107-e789-4660-a122-3816c43af703"
}`

const ExpectedActionID = "2a0ff107-e789-4660-a122-3816c43af703"

func HandleCreateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/nodes", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("X-OpenStack-Request-ID", "req-3791a089-9d46-4671-a3f9-55e95e55d2b4")
		w.Header().Add("Location", "http://senlin.cloud.blizzard.net:8778/v1/actions/ffd94dd8-6266-4887-9a8c-5b78b72136da")

		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, CreateResponse)
	})
}

func HandleListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/nodes", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ListResponse)
	})
}

func HandleDeleteSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/nodes/6dc6d336e3fc4c0a951b5698cd1236ee", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusNoContent)
	})
}

func HandleGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/nodes/573aa1ba-bf45-49fd-907d-6b5d6e6adfd3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, GetResponse)
	})
}

func HandleUpdateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/nodes/82fe28e0-9fcb-42ca-a2fa-6eb7dddd75a1", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, UpdateResponse)
	})
}

func HandleOpsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/nodes/7d85f602-a948-4a30-afd4-e84f47471c15/ops", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)

		fmt.Fprint(w, OperationActionResponse)
	})
}

func HandleRecoverSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/nodes/edce3528-864f-41fb-8759-f4707925cc09/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("X-OpenStack-Request-ID", "req-edce3528-864f-41fb-8759-f4707925cc09")
		w.WriteHeader(http.StatusAccepted)
		fmt.Fprint(w, ActionResponse)
	})
}

func HandleCheckSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/nodes/edce3528-864f-41fb-8759-f4707925cc09/actions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("X-OpenStack-Request-ID", "req-edce3528-864f-41fb-8759-f4707925cc09")
		w.WriteHeader(http.StatusAccepted)
		fmt.Fprint(w, ActionResponse)
	})
}
