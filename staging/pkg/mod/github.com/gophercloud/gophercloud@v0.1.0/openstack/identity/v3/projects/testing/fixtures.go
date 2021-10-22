package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/projects"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ListOutput provides a single page of Project results.
const ListOutput = `
{
  "projects": [
    {
      "is_domain": false,
      "description": "The team that is red",
      "domain_id": "default",
      "enabled": true,
      "id": "1234",
      "name": "Red Team",
      "parent_id": null
    },
    {
      "is_domain": false,
      "description": "The team that is blue",
      "domain_id": "default",
      "enabled": true,
      "id": "9876",
      "name": "Blue Team",
      "parent_id": null
    }
  ],
  "links": {
    "next": null,
    "previous": null
  }
}
`

// GetOutput provides a Get result.
const GetOutput = `
{
  "project": {
		"is_domain": false,
		"description": "The team that is red",
		"domain_id": "default",
		"enabled": true,
		"id": "1234",
		"name": "Red Team",
		"parent_id": null
  }
}
`

// CreateRequest provides the input to a Create request.
const CreateRequest = `
{
  "project": {
		"description": "The team that is red",
		"name": "Red Team"
  }
}
`

// UpdateRequest provides the input to an Update request.
const UpdateRequest = `
{
  "project": {
		"description": "The team that is bright red",
		"name": "Bright Red Team"
  }
}
`

// UpdateOutput provides an Update response.
const UpdateOutput = `
{
  "project": {
		"is_domain": false,
		"description": "The team that is bright red",
		"domain_id": "default",
		"enabled": true,
		"id": "1234",
		"name": "Bright Red Team",
		"parent_id": null
  }
}
`

// RedTeam is a Project fixture.
var RedTeam = projects.Project{
	IsDomain:    false,
	Description: "The team that is red",
	DomainID:    "default",
	Enabled:     true,
	ID:          "1234",
	Name:        "Red Team",
	ParentID:    "",
}

// BlueTeam is a Project fixture.
var BlueTeam = projects.Project{
	IsDomain:    false,
	Description: "The team that is blue",
	DomainID:    "default",
	Enabled:     true,
	ID:          "9876",
	Name:        "Blue Team",
	ParentID:    "",
}

// UpdatedRedTeam is a Project Fixture.
var UpdatedRedTeam = projects.Project{
	IsDomain:    false,
	Description: "The team that is bright red",
	DomainID:    "default",
	Enabled:     true,
	ID:          "1234",
	Name:        "Bright Red Team",
	ParentID:    "",
}

// ExpectedProjectSlice is the slice of projects expected to be returned from ListOutput.
var ExpectedProjectSlice = []projects.Project{RedTeam, BlueTeam}

// HandleListProjectsSuccessfully creates an HTTP handler at `/projects` on the
// test handler mux that responds with a list of two tenants.
func HandleListProjectsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/projects", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListOutput)
	})
}

// HandleGetProjectSuccessfully creates an HTTP handler at `/projects` on the
// test handler mux that responds with a single project.
func HandleGetProjectSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/projects/1234", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleCreateProjectSuccessfully creates an HTTP handler at `/projects` on the
// test handler mux that tests project creation.
func HandleCreateProjectSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/projects", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, CreateRequest)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleDeleteProjectSuccessfully creates an HTTP handler at `/projects` on the
// test handler mux that tests project deletion.
func HandleDeleteProjectSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/projects/1234", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleUpdateProjectSuccessfully creates an HTTP handler at `/projects` on the
// test handler mux that tests project updates.
func HandleUpdateProjectSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/projects/1234", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, UpdateRequest)

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, UpdateOutput)
	})
}
