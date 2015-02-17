package users

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func mockListResponse(t *testing.T) {
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
						"username": "jqsmith",
						"email": "john.smith@example.org",
						"enabled": true
				},
				{
						"id": "u1001",
						"username": "jqsmith",
						"email": "jane.smith@example.org",
						"enabled": true
				}
		]
}
	`)
	})
}

func mockCreateUser(t *testing.T) {
	th.Mux.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
    "user": {
        "username": "new_user",
        "enabled": false,
        "email": "new_user@foo.com",
				"OS-KSADM:password": "foo"
    }
}
  `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
  "user": {
    "RAX-AUTH:defaultRegion": "DFW",
    "RAX-AUTH:domainId": "5830280",
    "id": "123456",
    "username": "new_user",
    "email": "new_user@foo.com",
    "enabled": false
  }
}
`)
	})
}

func mockGetUser(t *testing.T) {
	th.Mux.HandleFunc("/users/new_user", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
	"user": {
		"RAX-AUTH:defaultRegion": "DFW",
		"RAX-AUTH:domainId": "5830280",
		"RAX-AUTH:multiFactorEnabled": "true",
		"id": "c39e3de9be2d4c779f1dfd6abacc176d",
		"username": "jqsmith",
		"email": "john.smith@example.org",
		"enabled": true
	}
}
`)
	})
}

func mockUpdateUser(t *testing.T) {
	th.Mux.HandleFunc("/users/c39e3de9be2d4c779f1dfd6abacc176d", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
	"user": {
		"email": "new_email@foo.com",
		"enabled": true
	}
}
`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
	"user": {
		"RAX-AUTH:defaultRegion": "DFW",
		"RAX-AUTH:domainId": "5830280",
		"RAX-AUTH:multiFactorEnabled": "true",
		"id": "123456",
		"username": "jqsmith",
		"email": "new_email@foo.com",
		"enabled": true
	}
}
`)
	})
}

func mockDeleteUser(t *testing.T) {
	th.Mux.HandleFunc("/users/c39e3de9be2d4c779f1dfd6abacc176d", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})
}

func mockResetAPIKey(t *testing.T) {
	th.Mux.HandleFunc("/users/99/OS-KSADM/credentials/RAX-KSKEY:apiKeyCredentials/RAX-AUTH/reset", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
{
    "RAX-KSKEY:apiKeyCredentials": {
        "username": "joesmith",
        "apiKey": "mooH1eiLahd5ahYood7r"
    }
}`)
	})
}
