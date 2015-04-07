package apiversions

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestListVersions(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `{
			"versions": [
				{
					"status": "CURRENT",
					"updated": "2012-01-04T11:33:21Z",
					"id": "v1.0",
					"links": [
						{
							"href": "http://23.253.228.211:8776/v1/",
							"rel": "self"
						}
					]
			    },
				{
					"status": "CURRENT",
					"updated": "2012-11-21T11:33:21Z",
					"id": "v2.0",
					"links": [
						{
							"href": "http://23.253.228.211:8776/v2/",
							"rel": "self"
						}
					]
				}
			]
		}`)
	})

	count := 0

	List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractAPIVersions(page)
		if err != nil {
			t.Errorf("Failed to extract API versions: %v", err)
			return false, err
		}

		expected := []APIVersion{
			APIVersion{
				ID:      "v1.0",
				Status:  "CURRENT",
				Updated: "2012-01-04T11:33:21Z",
			},
			APIVersion{
				ID:      "v2.0",
				Status:  "CURRENT",
				Updated: "2012-11-21T11:33:21Z",
			},
		}

		th.AssertDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestAPIInfo(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v1/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `{
			"version": {
				"status": "CURRENT",
				"updated": "2012-01-04T11:33:21Z",
				"media-types": [
					{
						"base": "application/xml",
						"type": "application/vnd.openstack.volume+xml;version=1"
					},
					{
						"base": "application/json",
						"type": "application/vnd.openstack.volume+json;version=1"
					}
				],
				"id": "v1.0",
				"links": [
					{
						"href": "http://23.253.228.211:8776/v1/",
						"rel": "self"
					},
					{
						"href": "http://jorgew.github.com/block-storage-api/content/os-block-storage-1.0.pdf",
						"type": "application/pdf",
						"rel": "describedby"
					},
					{
						"href": "http://docs.rackspacecloud.com/servers/api/v1.1/application.wadl",
						"type": "application/vnd.sun.wadl+xml",
						"rel": "describedby"
					}
				]
			}
		}`)
	})

	actual, err := Get(client.ServiceClient(), "v1").Extract()
	if err != nil {
		t.Errorf("Failed to extract version: %v", err)
	}

	expected := APIVersion{
		ID:      "v1.0",
		Status:  "CURRENT",
		Updated: "2012-01-04T11:33:21Z",
	}

	th.AssertEquals(t, actual.ID, expected.ID)
	th.AssertEquals(t, actual.Status, expected.Status)
	th.AssertEquals(t, actual.Updated, expected.Updated)
}
