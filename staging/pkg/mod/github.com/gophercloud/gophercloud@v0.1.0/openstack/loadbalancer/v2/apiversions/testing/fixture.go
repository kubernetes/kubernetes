package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/apiversions"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

const OctaviaAllAPIVersionsResponse = `
{
    "versions": [
        {
            "id": "v1",
            "links": [
                {
                    "href": "http://10.0.0.105:9876/v1",
                    "rel": "self"
                }
            ],
            "status": "DEPRECATED",
            "updated": "2014-12-11T00:00:00Z"
        },
        {
            "id": "v2.0",
            "links": [
                {
                    "href": "http://10.0.0.105:9876/v2",
                    "rel": "self"
                }
            ],
            "status": "SUPPORTED",
            "updated": "2016-12-11T00:00:00Z"
        },
        {
            "id": "v2.1",
            "links": [
                {
                    "href": "http://10.0.0.105:9876/v2",
                    "rel": "self"
                }
            ],
            "status": "SUPPORTED",
            "updated": "2018-04-20T00:00:00Z"
        },
        {
            "id": "v2.2",
            "links": [
                {
                    "href": "http://10.0.0.105:9876/v2",
                    "rel": "self"
                }
            ],
            "status": "CURRENT",
            "updated": "2018-07-31T00:00:00Z"
        }
    ]
}
`

var OctaviaAllAPIVersionResults = []apiversions.APIVersion{
	apiversions.APIVersion{
		ID:     "v1",
		Status: "DEPRECATED",
	},
	apiversions.APIVersion{
		ID:     "v2.0",
		Status: "SUPPORTED",
	},
	apiversions.APIVersion{
		ID:     "v2.1",
		Status: "SUPPORTED",
	},
	apiversions.APIVersion{
		ID:     "v2.2",
		Status: "CURRENT",
	},
}

func MockListResponse(t *testing.T) {
	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, OctaviaAllAPIVersionsResponse)
	})
}
