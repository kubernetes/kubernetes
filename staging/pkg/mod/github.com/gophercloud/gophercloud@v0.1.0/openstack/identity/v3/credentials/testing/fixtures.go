package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/credentials"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

const userID = "bb5476fd12884539b41d5a88f838d773"
const credentialID = "3d3367228f9c7665266604462ec60029bcd83ad89614021a80b2eb879c572510"
const projectID = "731fc6f265cd486d900f16e84c5cb594"

// ListOutput provides a single page of Credential results.
const ListOutput = `
{
    "credentials": [
        {
            "user_id": "bb5476fd12884539b41d5a88f838d773",
            "links": {
                "self": "http://identity/v3/credentials/3d3367228f9c7665266604462ec60029bcd83ad89614021a80b2eb879c572510"
            },
            "blob": "{\"access\":\"181920\",\"secret\":\"secretKey\"}",
            "project_id": "731fc6f265cd486d900f16e84c5cb594",
            "type": "ec2",
            "id": "3d3367228f9c7665266604462ec60029bcd83ad89614021a80b2eb879c572510"
        },
        {
            "user_id": "6f556708d04b4ea6bc72d7df2296b71a",
            "links": {
                "self": "http://identity/v3/credentials/2441494e52ab6d594a34d74586075cb299489bdd1e9389e3ab06467a4f460609"
            },
            "blob": "{\"access\":\"7da79ff0aa364e1396f067e352b9b79a\",\"secret\":\"7a18d68ba8834b799d396f3ff6f1e98c\"}",
            "project_id": "1a1d14690f3c4ec5bf5f321c5fde3c16",
            "type": "ec2",
            "id": "2441494e52ab6d594a34d74586075cb299489bdd1e9389e3ab06467a4f460609"
        }
	],
    "links": {
        "self": "http://identity/v3/credentials",
        "previous": null,
        "next": null
    }
}
`

// GetOutput provides a Get result.
const GetOutput = `
{
    "credential": {
        "user_id": "bb5476fd12884539b41d5a88f838d773",
        "links": {
            "self": "http://identity/v3/credentials/3d3367228f9c7665266604462ec60029bcd83ad89614021a80b2eb879c572510"
        },
        "blob": "{\"access\":\"181920\",\"secret\":\"secretKey\"}",
        "project_id": "731fc6f265cd486d900f16e84c5cb594",
        "type": "ec2",
        "id": "3d3367228f9c7665266604462ec60029bcd83ad89614021a80b2eb879c572510"
    }
}
`

// CreateRequest provides the input to a Create request.
const CreateRequest = `
{
    "credential": {
        "blob": "{\"access\":\"181920\",\"secret\":\"secretKey\"}",
        "project_id": "731fc6f265cd486d900f16e84c5cb594",
        "type": "ec2",
        "user_id": "bb5476fd12884539b41d5a88f838d773"
    }
}
`

// UpdateRequest provides the input to as Update request.
const UpdateRequest = `
{
    "credential": {
        "blob": "{\"access\":\"181920\",\"secret\":\"secretKey\"}",
        "project_id": "731fc6f265cd486d900f16e84c5cb594",
        "type": "ec2",
        "user_id": "bb5476fd12884539b41d5a88f838d773"
    }
}
`

// UpdateOutput provides an update result.
const UpdateOutput = `
{
    "credential": {
        "user_id": "bb5476fd12884539b41d5a88f838d773",
        "links": {
            "self": "http://identity/v3/credentials/2441494e52ab6d594a34d74586075cb299489bdd1e9389e3ab06467a4f460609"
        },
        "blob": "{\"access\":\"181920\",\"secret\":\"secretKey\"}",
        "project_id": "731fc6f265cd486d900f16e84c5cb594",
        "type": "ec2",
        "id": "2441494e52ab6d594a34d74586075cb299489bdd1e9389e3ab06467a4f460609"
    }
}
`

var Credential = credentials.Credential{
	ID:        credentialID,
	ProjectID: projectID,
	Type:      "ec2",
	UserID:    userID,
	Blob:      "{\"access\":\"181920\",\"secret\":\"secretKey\"}",
	Links: map[string]interface{}{
		"self": "http://identity/v3/credentials/3d3367228f9c7665266604462ec60029bcd83ad89614021a80b2eb879c572510",
	},
}

var FirstCredential = credentials.Credential{
	ID:        credentialID,
	ProjectID: projectID,
	Type:      "ec2",
	UserID:    userID,
	Blob:      "{\"access\":\"181920\",\"secret\":\"secretKey\"}",
	Links: map[string]interface{}{
		"self": "http://identity/v3/credentials/3d3367228f9c7665266604462ec60029bcd83ad89614021a80b2eb879c572510",
	},
}

var SecondCredential = credentials.Credential{
	ID:        "2441494e52ab6d594a34d74586075cb299489bdd1e9389e3ab06467a4f460609",
	ProjectID: "1a1d14690f3c4ec5bf5f321c5fde3c16",
	Type:      "ec2",
	UserID:    "6f556708d04b4ea6bc72d7df2296b71a",
	Blob:      "{\"access\":\"7da79ff0aa364e1396f067e352b9b79a\",\"secret\":\"7a18d68ba8834b799d396f3ff6f1e98c\"}",
	Links: map[string]interface{}{
		"self": "http://identity/v3/credentials/2441494e52ab6d594a34d74586075cb299489bdd1e9389e3ab06467a4f460609",
	},
}

// SecondCredentialUpdated is how SecondCredential should look after an Update.
var SecondCredentialUpdated = credentials.Credential{
	ID:        "2441494e52ab6d594a34d74586075cb299489bdd1e9389e3ab06467a4f460609",
	ProjectID: projectID,
	Type:      "ec2",
	UserID:    userID,
	Blob:      "{\"access\":\"181920\",\"secret\":\"secretKey\"}",
	Links: map[string]interface{}{
		"self": "http://identity/v3/credentials/2441494e52ab6d594a34d74586075cb299489bdd1e9389e3ab06467a4f460609",
	},
}

// ExpectedCredentialsSlice is the slice of credentials expected to be returned from ListOutput.
var ExpectedCredentialsSlice = []credentials.Credential{FirstCredential, SecondCredential}

// HandleListCredentialsSuccessfully creates an HTTP handler at `/credentials` on the
// test handler mux that responds with a list of two credentials.
func HandleListCredentialsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/credentials", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListOutput)
	})
}

// HandleGetCredentialSuccessfully creates an HTTP handler at `/credentials` on the
// test handler mux that responds with a single credential.
func HandleGetCredentialSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/credentials/3d3367228f9c7665266604462ec60029bcd83ad89614021a80b2eb879c572510", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleCreateCredentialSuccessfully creates an HTTP handler at `/credentials` on the
// test handler mux that tests credential creation.
func HandleCreateCredentialSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/credentials", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, CreateRequest)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleDeleteCredentialSuccessfully creates an HTTP handler at `/credentials` on the
// test handler mux that tests credential deletion.
func HandleDeleteCredentialSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/credentials/3d3367228f9c7665266604462ec60029bcd83ad89614021a80b2eb879c572510", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleUpdateCredentialsSuccessfully creates an HTTP handler at `/credentials` on the
// test handler mux that tests credentials update.
func HandleUpdateCredentialSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/credentials/2441494e52ab6d594a34d74586075cb299489bdd1e9389e3ab06467a4f460609", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, UpdateRequest)

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, UpdateOutput)
	})
}
