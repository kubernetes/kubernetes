package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/keymanager/v1/acls"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

const GetResponse = `
{
  "read": {
    "created": "2018-06-22T17:54:24",
    "project-access": false,
    "updated": "2018-06-22T17:54:24",
    "users": [
      "GG27dVwR9gBMnsOaRoJ1DFJmZfdVjIdW"
    ]
  }
}`

const SetRequest = `
{
  "read": {
    "project-access": false,
    "users": [
      "GG27dVwR9gBMnsOaRoJ1DFJmZfdVjIdW"
    ]
  }
}`

const SecretSetResponse = `
{
  "acl_ref": "http://barbican:9311/v1/secrets/4befede0-fbde-4480-982c-b160c1014a47/acl"
}`

const ContainerSetResponse = `
{
  "acl_ref": "http://barbican:9311/v1/containers/4befede0-fbde-4480-982c-b160c1014a47/acl"
}`

var ExpectedACL = acls.ACL{
	"read": acls.ACLDetails{
		Created:       time.Date(2018, 6, 22, 17, 54, 24, 0, time.UTC),
		ProjectAccess: false,
		Updated:       time.Date(2018, 6, 22, 17, 54, 24, 0, time.UTC),
		Users: []string{
			"GG27dVwR9gBMnsOaRoJ1DFJmZfdVjIdW",
		},
	},
}

var ExpectedSecretACLRef = acls.ACLRef("http://barbican:9311/v1/secrets/4befede0-fbde-4480-982c-b160c1014a47/acl")

var ExpectedContainerACLRef = acls.ACLRef("http://barbican:9311/v1/containers/4befede0-fbde-4480-982c-b160c1014a47/acl")

const UpdateRequest = `
{
  "read": {
    "users": []
  }
}`

// HandleGetSecretACLSuccessfully creates an HTTP handler at `/secrets/uuid/acl`
// on the test handler mux that responds with an acl.
func HandleGetSecretACLSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/secrets/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c/acl", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, GetResponse)
	})
}

// HandleGetContainerACLSuccessfully creates an HTTP handler at `/secrets/uuid/acl`
// on the test handler mux that responds with an acl.
func HandleGetContainerACLSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/containers/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c/acl", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, GetResponse)
	})
}

// HandleSetSecretACLSuccessfully creates an HTTP handler at `/secrets` on the
// test handler mux that tests secret creation.
func HandleSetSecretACLSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/secrets/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c/acl", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, SetRequest)

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, SecretSetResponse)
	})
}

// HandleSetContainerACLSuccessfully creates an HTTP handler at `/secrets` on the
// test handler mux that tests secret creation.
func HandleSetContainerACLSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/containers/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c/acl", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, SetRequest)

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ContainerSetResponse)
	})
}

// HandleUpdateSecretACLSuccessfully creates an HTTP handler at `/secrets` on the
// test handler mux that tests secret creation.
func HandleUpdateSecretACLSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/secrets/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c/acl", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, UpdateRequest)

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, SecretSetResponse)
	})
}

// HandleUpdateContainerACLSuccessfully creates an HTTP handler at `/secrets` on the
// test handler mux that tests secret creation.
func HandleUpdateContainerACLSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/containers/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c/acl", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, UpdateRequest)

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ContainerSetResponse)
	})
}

// HandleDeleteSecretACLSuccessfully creates an HTTP handler at `/secrets` on the
// test handler mux that tests secret deletion.
func HandleDeleteSecretACLSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/secrets/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c/acl", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusOK)
	})
}

// HandleDeleteContainerACLSuccessfully creates an HTTP handler at `/secrets` on the
// test handler mux that tests secret deletion.
func HandleDeleteContainerACLSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/containers/1b8068c4-3bb6-4be6-8f1e-da0d1ea0b67c/acl", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusOK)
	})
}
