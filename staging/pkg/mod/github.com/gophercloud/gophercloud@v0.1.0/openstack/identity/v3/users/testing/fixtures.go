package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/groups"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/projects"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/users"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ListOutput provides a single page of User results.
const ListOutput = `
{
    "links": {
        "next": null,
        "previous": null,
        "self": "http://example.com/identity/v3/users"
    },
    "users": [
        {
            "domain_id": "default",
            "enabled": true,
            "id": "2844b2a08be147a08ef58317d6471f1f",
            "links": {
                "self": "http://example.com/identity/v3/users/2844b2a08be147a08ef58317d6471f1f"
            },
            "name": "glance",
            "password_expires_at": null,
            "description": "some description",
            "extra": {
              "email": "glance@localhost"
            }
        },
        {
            "default_project_id": "263fd9",
            "domain_id": "1789d1",
            "enabled": true,
            "id": "9fe1d3",
            "links": {
                "self": "https://example.com/identity/v3/users/9fe1d3"
            },
            "name": "jsmith",
            "password_expires_at": "2016-11-06T15:32:17.000000",
            "email": "jsmith@example.com",
            "options": {
                "ignore_password_expiry": true,
                "multi_factor_auth_rules": [["password", "totp"], ["password", "custom-auth-method"]]
            }
        }
    ]
}
`

// GetOutput provides a Get result.
const GetOutput = `
{
    "user": {
        "default_project_id": "263fd9",
        "domain_id": "1789d1",
        "enabled": true,
        "id": "9fe1d3",
        "links": {
            "self": "https://example.com/identity/v3/users/9fe1d3"
        },
        "name": "jsmith",
        "password_expires_at": "2016-11-06T15:32:17.000000",
        "email": "jsmith@example.com",
        "options": {
            "ignore_password_expiry": true,
            "multi_factor_auth_rules": [["password", "totp"], ["password", "custom-auth-method"]]
        }
    }
}
`

// GetOutputNoOptions provides a Get result of a user with no options.
const GetOutputNoOptions = `
{
    "user": {
        "default_project_id": "263fd9",
        "domain_id": "1789d1",
        "enabled": true,
        "id": "9fe1d3",
        "links": {
            "self": "https://example.com/identity/v3/users/9fe1d3"
        },
        "name": "jsmith",
        "password_expires_at": "2016-11-06T15:32:17.000000",
        "email": "jsmith@example.com"
    }
}
`

// CreateRequest provides the input to a Create request.
const CreateRequest = `
{
    "user": {
        "default_project_id": "263fd9",
        "domain_id": "1789d1",
        "enabled": true,
        "name": "jsmith",
        "password": "secretsecret",
        "email": "jsmith@example.com",
        "options": {
            "ignore_password_expiry": true,
            "multi_factor_auth_rules": [["password", "totp"], ["password", "custom-auth-method"]]
        }
    }
}
`

// CreateNoOptionsRequest provides the input to a Create request with no Options.
const CreateNoOptionsRequest = `
{
    "user": {
        "default_project_id": "263fd9",
        "domain_id": "1789d1",
        "enabled": true,
        "name": "jsmith",
        "password": "secretsecret",
        "email": "jsmith@example.com"
    }
}
`

// UpdateRequest provides the input to an Update request.
const UpdateRequest = `
{
    "user": {
        "enabled": false,
        "disabled_reason": "DDOS",
        "options": {
            "multi_factor_auth_rules": null
        }
    }
}
`

// UpdateOutput provides an update result.
const UpdateOutput = `
{
    "user": {
        "default_project_id": "263fd9",
        "domain_id": "1789d1",
        "enabled": false,
        "id": "9fe1d3",
        "links": {
            "self": "https://example.com/identity/v3/users/9fe1d3"
        },
        "name": "jsmith",
        "password_expires_at": "2016-11-06T15:32:17.000000",
        "email": "jsmith@example.com",
        "disabled_reason": "DDOS",
        "options": {
            "ignore_password_expiry": true
        }
    }
}
`

// ChangePasswordRequest provides the input to a ChangePassword request.
const ChangePasswordRequest = `
{
    "user": {
        "password": "new_secretsecret",
        "original_password": "secretsecret"
    }
}
`

// ListGroupsOutput provides a ListGroups result.
const ListGroupsOutput = `
{
    "groups": [
        {
            "description": "Developers cleared for work on all general projects",
            "domain_id": "1789d1",
            "id": "ea167b",
            "links": {
                "self": "https://example.com/identity/v3/groups/ea167b"
            },
            "building": "Hilltop A",
            "name": "Developers"
        },
        {
            "description": "Developers cleared for work on secret projects",
            "domain_id": "1789d1",
            "id": "a62db1",
            "links": {
                "self": "https://example.com/identity/v3/groups/a62db1"
            },
            "name": "Secure Developers"
        }
    ],
    "links": {
        "self": "http://example.com/identity/v3/users/9fe1d3/groups",
        "previous": null,
        "next": null
    }
}
`

// ListProjectsOutput provides a ListProjects result.
const ListProjectsOutput = `
{
    "links": {
        "next": null,
        "previous": null,
        "self": "http://localhost:5000/identity/v3/users/foobar/projects"
    },
    "projects": [
        {
            "description": "my first project",
            "domain_id": "11111",
            "enabled": true,
            "id": "abcde",
            "links": {
                "self": "http://localhost:5000/identity/v3/projects/abcde"
            },
            "name": "project 1",
            "parent_id": "11111"
        },
        {
            "description": "my second project",
            "domain_id": "22222",
            "enabled": true,
            "id": "bcdef",
            "links": {
                "self": "http://localhost:5000/identity/v3/projects/bcdef"
            },
            "name": "project 2",
            "parent_id": "22222"
        }
    ]
}
`

// FirstUser is the first user in the List request.
var nilTime time.Time
var FirstUser = users.User{
	DomainID: "default",
	Enabled:  true,
	ID:       "2844b2a08be147a08ef58317d6471f1f",
	Links: map[string]interface{}{
		"self": "http://example.com/identity/v3/users/2844b2a08be147a08ef58317d6471f1f",
	},
	Name:              "glance",
	PasswordExpiresAt: nilTime,
	Description:       "some description",
	Extra: map[string]interface{}{
		"email": "glance@localhost",
	},
}

// SecondUser is the second user in the List request.
var SecondUserPasswordExpiresAt, _ = time.Parse(gophercloud.RFC3339MilliNoZ, "2016-11-06T15:32:17.000000")
var SecondUser = users.User{
	DefaultProjectID: "263fd9",
	DomainID:         "1789d1",
	Enabled:          true,
	ID:               "9fe1d3",
	Links: map[string]interface{}{
		"self": "https://example.com/identity/v3/users/9fe1d3",
	},
	Name:              "jsmith",
	PasswordExpiresAt: SecondUserPasswordExpiresAt,
	Extra: map[string]interface{}{
		"email": "jsmith@example.com",
	},
	Options: map[string]interface{}{
		"ignore_password_expiry": true,
		"multi_factor_auth_rules": []interface{}{
			[]string{"password", "totp"},
			[]string{"password", "custom-auth-method"},
		},
	},
}

var SecondUserNoOptions = users.User{
	DefaultProjectID: "263fd9",
	DomainID:         "1789d1",
	Enabled:          true,
	ID:               "9fe1d3",
	Links: map[string]interface{}{
		"self": "https://example.com/identity/v3/users/9fe1d3",
	},
	Name:              "jsmith",
	PasswordExpiresAt: SecondUserPasswordExpiresAt,
	Extra: map[string]interface{}{
		"email": "jsmith@example.com",
	},
}

// SecondUserUpdated is how SecondUser should look after an Update.
var SecondUserUpdated = users.User{
	DefaultProjectID: "263fd9",
	DomainID:         "1789d1",
	Enabled:          false,
	ID:               "9fe1d3",
	Links: map[string]interface{}{
		"self": "https://example.com/identity/v3/users/9fe1d3",
	},
	Name:              "jsmith",
	PasswordExpiresAt: SecondUserPasswordExpiresAt,
	Extra: map[string]interface{}{
		"email":           "jsmith@example.com",
		"disabled_reason": "DDOS",
	},
	Options: map[string]interface{}{
		"ignore_password_expiry": true,
	},
}

// ExpectedUsersSlice is the slice of users expected to be returned from ListOutput.
var ExpectedUsersSlice = []users.User{FirstUser, SecondUser}

var FirstGroup = groups.Group{
	Description: "Developers cleared for work on all general projects",
	DomainID:    "1789d1",
	ID:          "ea167b",
	Links: map[string]interface{}{
		"self": "https://example.com/identity/v3/groups/ea167b",
	},
	Extra: map[string]interface{}{
		"building": "Hilltop A",
	},
	Name: "Developers",
}

var SecondGroup = groups.Group{
	Description: "Developers cleared for work on secret projects",
	DomainID:    "1789d1",
	ID:          "a62db1",
	Links: map[string]interface{}{
		"self": "https://example.com/identity/v3/groups/a62db1",
	},
	Extra: map[string]interface{}{},
	Name:  "Secure Developers",
}

var ExpectedGroupsSlice = []groups.Group{FirstGroup, SecondGroup}

var FirstProject = projects.Project{
	Description: "my first project",
	DomainID:    "11111",
	Enabled:     true,
	ID:          "abcde",
	Name:        "project 1",
	ParentID:    "11111",
}

var SecondProject = projects.Project{
	Description: "my second project",
	DomainID:    "22222",
	Enabled:     true,
	ID:          "bcdef",
	Name:        "project 2",
	ParentID:    "22222",
}

var ExpectedProjectsSlice = []projects.Project{FirstProject, SecondProject}

// HandleListUsersSuccessfully creates an HTTP handler at `/users` on the
// test handler mux that responds with a list of two users.
func HandleListUsersSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListOutput)
	})
}

// HandleGetUserSuccessfully creates an HTTP handler at `/users` on the
// test handler mux that responds with a single user.
func HandleGetUserSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/users/9fe1d3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleCreateUserSuccessfully creates an HTTP handler at `/users` on the
// test handler mux that tests user creation.
func HandleCreateUserSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, CreateRequest)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleCreateNoOptionsUserSuccessfully creates an HTTP handler at `/users` on the
// test handler mux that tests user creation.
func HandleCreateNoOptionsUserSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, CreateNoOptionsRequest)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, GetOutputNoOptions)
	})
}

// HandleUpdateUserSuccessfully creates an HTTP handler at `/users` on the
// test handler mux that tests user update.
func HandleUpdateUserSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/users/9fe1d3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, UpdateRequest)

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, UpdateOutput)
	})
}

// HandleChangeUserPasswordSuccessfully creates an HTTP handler at `/users` on the
// test handler mux that tests change user password.
func HandleChangeUserPasswordSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/users/9fe1d3/password", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, ChangePasswordRequest)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleDeleteUserSuccessfully creates an HTTP handler at `/users` on the
// test handler mux that tests user deletion.
func HandleDeleteUserSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/users/9fe1d3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleListUserGroupsSuccessfully creates an HTTP handler at /users/{userID}/groups
// on the test handler mux that respons with a list of two groups
func HandleListUserGroupsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/users/9fe1d3/groups", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListGroupsOutput)
	})
}

// HandleAddToGroupSuccessfully creates an HTTP handler at /groups/{groupID}/users/{userID}
// on the test handler mux that tests adding user to group.
func HandleAddToGroupSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/groups/ea167b/users/9fe1d3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleIsMemberOfGroupSuccessfully creates an HTTP handler at /groups/{groupID}/users/{userID}
// on the test handler mux that tests checking whether user belongs to group.
func HandleIsMemberOfGroupSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/groups/ea167b/users/9fe1d3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "HEAD")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleRemoveFromGroupSuccessfully creates an HTTP handler at /groups/{groupID}/users/{userID}
// on the test handler mux that tests removing user from group.
func HandleRemoveFromGroupSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/groups/ea167b/users/9fe1d3", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleListUserProjectsSuccessfully creates an HTTP handler at /users/{userID}/projects
// on the test handler mux that respons wit a list of two projects
func HandleListUserProjectsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/users/9fe1d3/projects", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListProjectsOutput)
	})
}

// HandleListInGroupSuccessfully creates an HTTP handler at /groups/{groupID}/users
// on the test handler mux that response with a list of two users
func HandleListInGroupSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/groups/ea167b/users", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListOutput)
	})
}
