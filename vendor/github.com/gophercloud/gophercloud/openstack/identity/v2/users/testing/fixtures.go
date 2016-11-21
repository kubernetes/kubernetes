package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func MockListUserResponse(t *testing.T) {
	th.Mux.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "users":[
        {
            "id": "u1000",
						"name": "John Smith",
            "username": "jqsmith",
            "email": "john.smith@example.org",
            "enabled": true,
						"tenant_id": "12345"
        },
        {
            "id": "u1001",
						"name": "Jane Smith",
            "username": "jqsmith",
            "email": "jane.smith@example.org",
            "enabled": true,
						"tenant_id": "12345"
        }
    ]
}
  `)
	})
}

func mockCreateUserResponse(t *testing.T) {
	th.Mux.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
    "user": {
		    "name": "new_user",
		    "tenant_id": "12345",
				"enabled": false,
				"email": "new_user@foo.com"
    }
}
	`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "user": {
        "name": "new_user",
        "tenant_id": "12345",
        "enabled": false,
        "email": "new_user@foo.com",
        "id": "c39e3de9be2d4c779f1dfd6abacc176d"
    }
}
`)
	})
}

func mockGetUserResponse(t *testing.T) {
	th.Mux.HandleFunc("/users/new_user", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
		"user": {
				"name": "new_user",
				"tenant_id": "12345",
				"enabled": false,
				"email": "new_user@foo.com",
				"id": "c39e3de9be2d4c779f1dfd6abacc176d"
		}
}
`)
	})
}

func mockUpdateUserResponse(t *testing.T) {
	th.Mux.HandleFunc("/users/c39e3de9be2d4c779f1dfd6abacc176d", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
    "user": {
		    "name": "new_name",
		    "enabled": true,
		    "email": "new_email@foo.com"
    }
}
`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
		"user": {
				"name": "new_name",
				"tenant_id": "12345",
				"enabled": true,
				"email": "new_email@foo.com",
				"id": "c39e3de9be2d4c779f1dfd6abacc176d"
		}
}
`)
	})
}

func mockDeleteUserResponse(t *testing.T) {
	th.Mux.HandleFunc("/users/c39e3de9be2d4c779f1dfd6abacc176d", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})
}

func mockListRolesResponse(t *testing.T) {
	th.Mux.HandleFunc("/tenants/1d8b6120dcc640fda4fc9194ffc80273/users/c39e3de9be2d4c779f1dfd6abacc176d/roles", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "roles": [
        {
            "id": "9fe2ff9ee4384b1894a90878d3e92bab",
            "name": "foo_role"
        },
        {
            "id": "1ea3d56793574b668e85960fbf651e13",
            "name": "admin"
        }
    ]
}
	`)
	})
}
