package roles

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func MockListRoleResponse(t *testing.T) {
	th.Mux.HandleFunc("/OS-KSADM/roles", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "roles": [
        {
            "id": "123",
            "name": "compute:admin",
            "description": "Nova Administrator",
            "serviceId": "cke5372ebabeeabb70a0e702a4626977x4406e5"
        }
    ]
}
  `)
	})
}

func MockAddUserRoleResponse(t *testing.T) {
	th.Mux.HandleFunc("/users/{user_id}/roles/OS-KSADM/{role_id}", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusCreated)
	})
}

func MockDeleteUserRoleResponse(t *testing.T) {
	th.Mux.HandleFunc("/users/{user_id}/roles/OS-KSADM/{role_id}", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})
}
