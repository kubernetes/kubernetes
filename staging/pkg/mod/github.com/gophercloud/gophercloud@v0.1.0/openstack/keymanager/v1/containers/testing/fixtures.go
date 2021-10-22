package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/keymanager/v1/containers"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ListResponse provides a single page of container results.
const ListResponse = `
{
  "containers": [
    {
      "consumers": [],
      "container_ref": "http://barbican:9311/v1/containers/dfdb88f3-4ddb-4525-9da6-066453caa9b0",
      "created": "2018-06-21T21:28:37",
      "creator_id": "5c70d99f4a8641c38f8084b32b5e5c0e",
      "name": "mycontainer",
      "secret_refs": [
        {
          "name": "mysecret",
          "secret_ref": "http://barbican:9311/v1/secrets/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c"
        }
      ],
      "status": "ACTIVE",
      "type": "generic",
      "updated": "2018-06-21T21:28:37"
    },
    {
      "consumers": [],
      "container_ref": "http://barbican:9311/v1/containers/47b20e73-335b-4867-82dc-3796524d5e20",
      "created": "2018-06-21T21:30:09",
      "creator_id": "5c70d99f4a8641c38f8084b32b5e5c0e",
      "name": "anothercontainer",
      "secret_refs": [
        {
          "name": "another",
          "secret_ref": "http://barbican:9311/v1/secrets/1b12b69a-8822-442e-a303-da24ade648ac"
        }
      ],
        "status": "ACTIVE",
        "type": "generic",
        "updated": "2018-06-21T21:30:09"
    }
  ],
  "total": 2
}`

// GetResponse provides a Get result.
const GetResponse = `
{
  "consumers": [],
  "container_ref": "http://barbican:9311/v1/containers/dfdb88f3-4ddb-4525-9da6-066453caa9b0",
  "created": "2018-06-21T21:28:37",
  "creator_id": "5c70d99f4a8641c38f8084b32b5e5c0e",
  "name": "mycontainer",
  "secret_refs": [
    {
      "name": "mysecret",
      "secret_ref": "http://barbican:9311/v1/secrets/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c"
    }
  ],
  "status": "ACTIVE",
  "type": "generic",
  "updated": "2018-06-21T21:28:37"
}`

// CreateRequest provides the input to a Create request.
const CreateRequest = `
{
  "name": "mycontainer",
  "secret_refs": [
    {
      "name": "mysecret",
      "secret_ref": "http://barbican:9311/v1/secrets/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c"
    }
  ],
  "type": "generic"
}`

// CreateResponse is the response of a Create request.
const CreateResponse = `
{
  "consumers": [],
  "container_ref": "http://barbican:9311/v1/containers/dfdb88f3-4ddb-4525-9da6-066453caa9b0",
  "created": "2018-06-21T21:28:37",
  "creator_id": "5c70d99f4a8641c38f8084b32b5e5c0e",
  "name": "mycontainer",
  "secret_refs": [
    {
      "name": "mysecret",
      "secret_ref": "http://barbican:9311/v1/secrets/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c"
    }
  ],
  "status": "ACTIVE",
  "type": "generic",
  "updated": "2018-06-21T21:28:37"
}`

// FirstContainer is the first resource in the List request.
var FirstContainer = containers.Container{
	Consumers:    []containers.ConsumerRef{},
	ContainerRef: "http://barbican:9311/v1/containers/dfdb88f3-4ddb-4525-9da6-066453caa9b0",
	Created:      time.Date(2018, 6, 21, 21, 28, 37, 0, time.UTC),
	CreatorID:    "5c70d99f4a8641c38f8084b32b5e5c0e",
	Name:         "mycontainer",
	SecretRefs: []containers.SecretRef{
		{
			Name:      "mysecret",
			SecretRef: "http://barbican:9311/v1/secrets/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c",
		},
	},
	Status:  "ACTIVE",
	Type:    "generic",
	Updated: time.Date(2018, 6, 21, 21, 28, 37, 0, time.UTC),
}

// SecondContainer is the second resource in the List request.
var SecondContainer = containers.Container{
	Consumers:    []containers.ConsumerRef{},
	ContainerRef: "http://barbican:9311/v1/containers/47b20e73-335b-4867-82dc-3796524d5e20",
	Created:      time.Date(2018, 6, 21, 21, 30, 9, 0, time.UTC),
	CreatorID:    "5c70d99f4a8641c38f8084b32b5e5c0e",
	Name:         "anothercontainer",
	SecretRefs: []containers.SecretRef{
		{
			Name:      "another",
			SecretRef: "http://barbican:9311/v1/secrets/1b12b69a-8822-442e-a303-da24ade648ac",
		},
	},
	Status:  "ACTIVE",
	Type:    "generic",
	Updated: time.Date(2018, 6, 21, 21, 30, 9, 0, time.UTC),
}

// ExpectedContainersSlice is the slice of containers expected to be returned from ListResponse.
var ExpectedContainersSlice = []containers.Container{FirstContainer, SecondContainer}

const ListConsumersResponse = `
{
  "consumers": [
    {
      "URL": "http://example.com",
      "created": "2018-06-22T16:26:25",
      "name": "CONSUMER-LZILN1zq",
      "status": "ACTIVE",
      "updated": "2018-06-22T16:26:25"
    }
  ],
  "total": 1
}`

// ExpectedConsumer is the expected result of a consumer retrieval.
var ExpectedConsumer = containers.Consumer{
	URL:     "http://example.com",
	Created: time.Date(2018, 6, 22, 16, 26, 25, 0, time.UTC),
	Name:    "CONSUMER-LZILN1zq",
	Status:  "ACTIVE",
	Updated: time.Date(2018, 6, 22, 16, 26, 25, 0, time.UTC),
}

// ExpectedConsumersSlice is an expected slice of consumers.
var ExpectedConsumersSlice = []containers.Consumer{ExpectedConsumer}

const CreateConsumerRequest = `
{
  "URL": "http://example.com",
  "name": "CONSUMER-LZILN1zq"
}`

const CreateConsumerResponse = `
{
  "consumers": [
    {
      "URL": "http://example.com",
      "name": "CONSUMER-LZILN1zq"
     }
  ],
  "container_ref": "http://barbican:9311/v1/containers/dfdb88f3-4ddb-4525-9da6-066453caa9b0",
  "created": "2018-06-21T21:28:37",
  "creator_id": "5c70d99f4a8641c38f8084b32b5e5c0e",
  "name": "mycontainer",
  "secret_refs": [
    {
      "name": "mysecret",
      "secret_ref": "http://barbican:9311/v1/secrets/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c"
    }
  ],
  "status": "ACTIVE",
  "type": "generic",
  "updated": "2018-06-21T21:28:37"
}`

// ExpectedCreatedConsumer is the expected result of adding a consumer.
var ExpectedCreatedConsumer = containers.Container{
	Consumers: []containers.ConsumerRef{
		{
			Name: "CONSUMER-LZILN1zq",
			URL:  "http://example.com",
		},
	},
	ContainerRef: "http://barbican:9311/v1/containers/dfdb88f3-4ddb-4525-9da6-066453caa9b0",
	Created:      time.Date(2018, 6, 21, 21, 28, 37, 0, time.UTC),
	CreatorID:    "5c70d99f4a8641c38f8084b32b5e5c0e",
	Name:         "mycontainer",
	SecretRefs: []containers.SecretRef{
		{
			Name:      "mysecret",
			SecretRef: "http://barbican:9311/v1/secrets/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c",
		},
	},
	Status:  "ACTIVE",
	Type:    "generic",
	Updated: time.Date(2018, 6, 21, 21, 28, 37, 0, time.UTC),
}

const DeleteConsumerRequest = `
{
  "URL": "http://example.com",
  "name": "CONSUMER-LZILN1zq"
}`

// HandleListContainersSuccessfully creates an HTTP handler at `/containers` on the
// test handler mux that responds with a list of two containers.
func HandleListContainersSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/containers", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListResponse)
	})
}

// HandleGetContainerSuccessfully creates an HTTP handler at `/containers` on the
// test handler mux that responds with a single resource.
func HandleGetContainerSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/containers/dfdb88f3-4ddb-4525-9da6-066453caa9b0", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, GetResponse)
	})
}

// HandleCreateContainerSuccessfully creates an HTTP handler at `/containers` on the
// test handler mux that tests resource creation.
func HandleCreateContainerSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/containers", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, CreateRequest)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, GetResponse)
	})
}

// HandleDeleteContainerSuccessfully creates an HTTP handler at `/containers` on the
// test handler mux that tests resource deletion.
func HandleDeleteContainerSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/containers/dfdb88f3-4ddb-4525-9da6-066453caa9b0", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleListConsumersSuccessfully creates an HTTP handler at
// `/containers/uuid/consumers` on the test handler mux that responds with
// a list of consumers.
func HandleListConsumersSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/containers/dfdb88f3-4ddb-4525-9da6-066453caa9b0/consumers", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListConsumersResponse)
	})
}

// HandleCreateConsumerSuccessfully creates an HTTP handler at
// `/containers/uuid/consumers` on the test handler mux that tests resource
// creation.
func HandleCreateConsumerSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/containers/dfdb88f3-4ddb-4525-9da6-066453caa9b0/consumers", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, CreateConsumerRequest)

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, CreateConsumerResponse)
	})
}

// HandleDeleteConsumerSuccessfully creates an HTTP handler at
// `/containers/uuid/consumers` on the test handler mux that tests resource
// deletion.
func HandleDeleteConsumerSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/containers/dfdb88f3-4ddb-4525-9da6-066453caa9b0/consumers", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, CreateConsumerRequest)

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, GetResponse)
	})
}
