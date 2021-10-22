package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/service/vN/resources"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ListResult provides a single page of RESOURCE results.
const ListResult = `
{
}
`

// GetResult provides a Get result.
const GetResult = `
{
}
`

// CreateRequest provides the input to a Create request.
const CreateRequest = `
{
}
`

// UpdateRequest provides the input to as Update request.
const UpdateRequest = `
{
}
`

// UpdateResult provides an update result.
const UpdateResult = `
{
}
`

// FirstResource is the first resource in the List request.
var FirstResource = resources.Resource{}

// SecondResource is the second resource in the List request.
var SecondResource = resources.Resource{}

// SecondResourceUpdated is how SecondResource should look after an Update.
var SecondResourceUpdated = resources.Resource{}

// ExpectedResourcesSlice is the slice of resources expected to be returned from ListResult.
var ExpectedResourcesSlice = []resources.Resource{FirstResource, SecondResource}

// HandleListResourceSuccessfully creates an HTTP handler at `/resources` on the
// test handler mux that responds with a list of two resources.
func HandleListResourceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/resources", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListResult)
	})
}

// HandleGetResourceSuccessfully creates an HTTP handler at `/resources` on the
// test handler mux that responds with a single resource.
func HandleGetResourceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/resources/9fe1d3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, GetResult)
	})
}

// HandleCreateResourceSuccessfully creates an HTTP handler at `/resources` on the
// test handler mux that tests resource creation.
func HandleCreateResourceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/resources", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, CreateRequest)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, GetResult)
	})
}

// HandleDeleteResourceSuccessfully creates an HTTP handler at `/resources` on the
// test handler mux that tests resource deletion.
func HandleDeleteResourceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/resources/9fe1d3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleUpdateResourceSuccessfully creates an HTTP handler at `/resources` on the
// test handler mux that tests resource update.
func HandleUpdateResourceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/resources/9fe1d3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, UpdateRequest)

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, UpdateResult)
	})
}
