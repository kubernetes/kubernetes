package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/clustering/v1/receivers"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const CreateResponse = `
{
	"receiver": {
		"action": "CLUSTER_SCALE_OUT",
		"actor": {
			"trust_id": [
				"6dc6d336e3fc4c0a951b5698cd1236d9"
			]
		},
		"channel": {
			"alarm_url": "http://node1:8778/v1/webhooks/e03dd2e5-8f2e-4ec1-8c6a-74ba891e5422/trigger?V=1&count=1"
		},
		"cluster_id": "ae63a10b-4a90-452c-aef1-113a0b255ee3",
		"created_at": "2015-11-04T05:21:41Z",
		"domain": "Default",
		"id": "573aa1ba-bf45-49fd-907d-6b5d6e6adfd3",
		"name": "cluster_inflate",
		"params": {
			"count": "1"
		},
		"project": "6e18cc2bdbeb48a5b3cad2dc499f6804",
		"type": "webhook",
		"updated_at": "2016-11-04T05:21:41Z",
		"user": "b4ad2d6e18cc2b9c48049f6dbe8a5b3c"
	}
}`

var ExpectedReceiver = receivers.Receiver{
	Action: "CLUSTER_SCALE_OUT",
	Actor: map[string]interface{}{
		"trust_id": []string{
			"6dc6d336e3fc4c0a951b5698cd1236d9",
		},
	},
	Channel: map[string]interface{}{
		"alarm_url": "http://node1:8778/v1/webhooks/e03dd2e5-8f2e-4ec1-8c6a-74ba891e5422/trigger?V=1&count=1",
	},
	ClusterID: "ae63a10b-4a90-452c-aef1-113a0b255ee3",
	CreatedAt: time.Date(2015, 11, 4, 5, 21, 41, 0, time.UTC),
	Domain:    "Default",
	ID:        "573aa1ba-bf45-49fd-907d-6b5d6e6adfd3",
	Name:      "cluster_inflate",
	Params: map[string]interface{}{
		"count": "1",
	},
	Project:   "6e18cc2bdbeb48a5b3cad2dc499f6804",
	Type:      "webhook",
	UpdatedAt: time.Date(2016, 11, 4, 5, 21, 41, 0, time.UTC),
	User:      "b4ad2d6e18cc2b9c48049f6dbe8a5b3c",
}

const GetResponse = `
{
	"receiver": {
		"action": "CLUSTER_SCALE_OUT",
		"actor": {
			"trust_id": [
				"6dc6d336e3fc4c0a951b5698cd1236d9"
			]
		},
		"channel": {
			"alarm_url": "http://node1:8778/v1/webhooks/e03dd2e5-8f2e-4ec1-8c6a-74ba891e5422/trigger?V=1&count=1"
		},
		"cluster_id": "ae63a10b-4a90-452c-aef1-113a0b255ee3",
		"created_at": "2015-11-04T05:21:41Z",
		"domain": "Default",
		"id": "573aa1ba-bf45-49fd-907d-6b5d6e6adfd3",
		"name": "cluster_inflate",
		"params": {
			"count": "1"
		},
		"project": "6e18cc2bdbeb48a5b3cad2dc499f6804",
		"type": "webhook",
		"updated_at": "2016-11-04T05:21:41Z",
		"user": "b4ad2d6e18cc2b9c48049f6dbe8a5b3c"
	}
}`

const UpdateResponse = `
{
	"receiver": {
		"action": "CLUSTER_SCALE_OUT",
		"actor": {
			"trust_id": [
				"6dc6d336e3fc4c0a951b5698cd1236d9"
			]
		},
		"channel": {
			"alarm_url": "http://node1:8778/v1/webhooks/e03dd2e5-8f2e-4ec1-8c6a-74ba891e5422/trigger?V=1&count=1"
		},
		"cluster_id": "ae63a10b-4a90-452c-aef1-113a0b255ee3",
		"created_at": "2015-06-27T05:09:43Z",
		"domain": "Default",
		"id": "573aa1ba-bf45-49fd-907d-6b5d6e6adfd3",
		"name": "cluster_inflate",
		"params": {
			"count": "1"
		},
		"project": "6e18cc2bdbeb48a5b3cad2dc499f6804",
		"type": "webhook",
		"updated_at": null,
		"user": "b4ad2d6e18cc2b9c48049f6dbe8a5b3c"
	}
}`

var ExpectedUpdateReceiver = receivers.Receiver{
	Action: "CLUSTER_SCALE_OUT",
	Actor: map[string]interface{}{
		"trust_id": []string{
			"6dc6d336e3fc4c0a951b5698cd1236d9",
		},
	},
	Channel: map[string]interface{}{
		"alarm_url": "http://node1:8778/v1/webhooks/e03dd2e5-8f2e-4ec1-8c6a-74ba891e5422/trigger?V=1&count=1",
	},
	ClusterID: "ae63a10b-4a90-452c-aef1-113a0b255ee3",
	CreatedAt: time.Date(2015, 6, 27, 5, 9, 43, 0, time.UTC),
	Domain:    "Default",
	ID:        "573aa1ba-bf45-49fd-907d-6b5d6e6adfd3",
	Name:      "cluster_inflate",
	Params: map[string]interface{}{
		"count": "1",
	},
	Project: "6e18cc2bdbeb48a5b3cad2dc499f6804",
	Type:    "webhook",
	User:    "b4ad2d6e18cc2b9c48049f6dbe8a5b3c",
}

const ListResponse = `
{
	"receivers": [
		{
			"action": "CLUSTER_SCALE_OUT",
			"actor": {
				"trust_id": [
					"6dc6d336e3fc4c0a951b5698cd1236d9"
				]
			},
			"channel": {
				"alarm_url": "http://node1:8778/v1/webhooks/e03dd2e5-8f2e-4ec1-8c6a-74ba891e5422/trigger?V=1&count=1"
			},
			"cluster_id": "ae63a10b-4a90-452c-aef1-113a0b255ee3",
			"created_at": "2015-06-27T05:09:43Z",
			"domain": "Default",
			"id": "573aa1ba-bf45-49fd-907d-6b5d6e6adfd3",
			"name": "cluster_inflate",
			"params": {
				"count": "1"
			},
			"project": "6e18cc2bdbeb48a5b3cad2dc499f6804",
			"type": "webhook",
			"updated_at": null,
			"user": "b4ad2d6e18cc2b9c48049f6dbe8a5b3c"
		}
	]
}`

var ExpectedReceiversList = []receivers.Receiver{ExpectedUpdateReceiver}
var ExpectedNotifyRequestID = "66a81d68-bf48-4af5-897b-a3bfef7279a8"

func HandleCreateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/receivers", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, CreateResponse)
	})
}

func HandleGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/receivers/573aa1ba-bf45-49fd-907d-6b5d6e6adfd3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, GetResponse)
	})
}

func HandleUpdateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/receivers/6dc6d336e3fc4c0a951b5698cd1236ee", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, UpdateResponse)
	})
}

func HandleListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/receivers", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestFormValues(t, r, map[string]string{"limit": "2", "sort": "name:asc,status:desc"})
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ListResponse)
	})
}

func HandleDeleteSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/receivers/6dc6d336e3fc4c0a951b5698cd1236ee", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusNoContent)
	})
}

func HandleNotifySuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/receivers/6dc6d336e3fc4c0a951b5698cd1236ee/notify", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("X-OpenStack-Request-Id", ExpectedNotifyRequestID)
		w.WriteHeader(http.StatusNoContent)
	})
}
