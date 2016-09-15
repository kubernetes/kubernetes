package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/orchestration/v1/apiversions"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListVersions(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "versions": [
        {
            "status": "CURRENT",
            "id": "v1.0",
            "links": [
                {
                    "href": "http://23.253.228.211:8000/v1",
                    "rel": "self"
                }
            ]
        }
    ]
}`)
	})

	count := 0

	apiversions.ListVersions(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := apiversions.ExtractAPIVersions(page)
		if err != nil {
			t.Errorf("Failed to extract API versions: %v", err)
			return false, err
		}

		expected := []apiversions.APIVersion{
			{
				Status: "CURRENT",
				ID:     "v1.0",
				Links: []gophercloud.Link{
					gophercloud.Link{
						Href: "http://23.253.228.211:8000/v1",
						Rel:  "self",
					},
				},
			},
		}

		th.AssertDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestNonJSONCannotBeExtractedIntoAPIVersions(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	apiversions.ListVersions(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		if _, err := apiversions.ExtractAPIVersions(page); err == nil {
			t.Fatalf("Expected error, got nil")
		}
		return true, nil
	})
}
