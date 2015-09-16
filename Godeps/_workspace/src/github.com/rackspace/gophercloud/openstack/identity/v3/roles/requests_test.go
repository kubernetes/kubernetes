package roles

import (
	"fmt"
	"net/http"
	"reflect"
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
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
	err := ListAssignments(client.ServiceClient(), ListAssignmentsOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractRoleAssignments(page)
		if err != nil {
			return false, err
		}

		expected := []RoleAssignment{
			RoleAssignment{
				Role:  Role{ID: "123456"},
				Scope: Scope{Domain: Domain{ID: "161718"}},
				User:  User{ID: "313233"},
				Group: Group{},
			},
			RoleAssignment{
				Role:  Role{ID: "123456"},
				Scope: Scope{Project: Project{ID: "456789"}},
				User:  User{ID: "313233"},
				Group: Group{},
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
