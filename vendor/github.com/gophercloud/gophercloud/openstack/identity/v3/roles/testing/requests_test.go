package testing

import (
	"fmt"
	"net/http"
	"reflect"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/roles"
	"github.com/gophercloud/gophercloud/pagination"
	"github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListSinglePage(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()

	testhelper.Mux.HandleFunc("/role_assignments", func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "GET")
		testhelper.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, `
			{
                "role_assignments": [
                    {
                        "links": {
                            "assignment": "http://identity:35357/v3/domains/161718/users/313233/roles/123456"
                        },
                        "role": {
                            "id": "123456"
                        },
                        "scope": {
                            "domain": {
                                "id": "161718"
                            }
                        },
                        "user": {
                            "id": "313233"
                        }
                    },
                    {
                        "links": {
                            "assignment": "http://identity:35357/v3/projects/456789/groups/101112/roles/123456",
                            "membership": "http://identity:35357/v3/groups/101112/users/313233"
                        },
                        "role": {
                            "id": "123456"
                        },
                        "scope": {
                            "project": {
                                "id": "456789"
                            }
                        },
                        "user": {
                            "id": "313233"
                        }
                    }
                ],
                "links": {
                    "self": "http://identity:35357/v3/role_assignments?effective",
                    "previous": null,
                    "next": null
                }
            }
		`)
	})

	count := 0
	err := roles.ListAssignments(client.ServiceClient(), roles.ListAssignmentsOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := roles.ExtractRoleAssignments(page)
		if err != nil {
			return false, err
		}

		expected := []roles.RoleAssignment{
			{
				Role:  roles.Role{ID: "123456"},
				Scope: roles.Scope{Domain: roles.Domain{ID: "161718"}},
				User:  roles.User{ID: "313233"},
				Group: roles.Group{},
			},
			{
				Role:  roles.Role{ID: "123456"},
				Scope: roles.Scope{Project: roles.Project{ID: "456789"}},
				User:  roles.User{ID: "313233"},
				Group: roles.Group{},
			},
		}

		if !reflect.DeepEqual(expected, actual) {
			t.Errorf("Expected %#v, got %#v", expected, actual)
		}

		return true, nil
	})
	if err != nil {
		t.Errorf("Unexpected error while paging: %v", err)
	}
	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}
