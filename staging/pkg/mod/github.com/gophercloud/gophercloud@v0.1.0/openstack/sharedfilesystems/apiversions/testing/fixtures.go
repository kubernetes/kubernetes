package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/apiversions"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

const ManilaAPIVersionResponse = `
{
    "versions": [
        {
            "id": "v2.0",
            "links": [
                {
                    "href": "http://docs.openstack.org/",
                    "rel": "describedby",
                    "type": "text/html"
                },
                {
                    "href": "http://localhost:8786/v2/",
                    "rel": "self"
                }
            ],
            "media-types": [
                {
                    "base": "application/json",
                    "type": "application/vnd.openstack.share+json;version=1"
                }
            ],
            "min_version": "2.0",
            "status": "CURRENT",
            "updated": "2015-08-27T11:33:21Z",
            "version": "2.32"
        }
    ]
}
`

const ManilaAPIInvalidVersionResponse_1 = `
{
    "versions": [
    ]
}
`

const ManilaAPIInvalidVersionResponse_2 = `
{
    "versions": [
        {
            "id": "v2.0",
            "links": [
                {
                    "href": "http://docs.openstack.org/",
                    "rel": "describedby",
                    "type": "text/html"
                },
                {
                    "href": "http://localhost:8786/v2/",
                    "rel": "self"
                }
            ],
            "media-types": [
                {
                    "base": "application/json",
                    "type": "application/vnd.openstack.share+json;version=1"
                }
            ],
            "min_version": "2.0",
            "status": "CURRENT",
            "updated": "2015-08-27T11:33:21Z",
            "version": "2.32"
        },
        {
            "id": "v2.9",
            "links": [
                {
                    "href": "http://docs.openstack.org/",
                    "rel": "describedby",
                    "type": "text/html"
                },
                {
                    "href": "http://localhost:8786/v2/",
                    "rel": "self"
                }
            ],
            "media-types": [
                {
                    "base": "application/json",
                    "type": "application/vnd.openstack.share+json;version=1"
                }
            ],
            "min_version": "2.9",
            "status": "CURRENT",
            "updated": "2015-08-27T11:33:21Z",
            "version": "2.99"
        }
    ]
}
`

const ManilaAllAPIVersionsResponse = `
{
    "versions": [
        {
            "id": "v1.0",
            "links": [
                {
                    "href": "http://docs.openstack.org/",
                    "rel": "describedby",
                    "type": "text/html"
                },
                {
                    "href": "http://localhost:8786/v1/",
                    "rel": "self"
                }
            ],
            "media-types": [
                {
                    "base": "application/json",
                    "type": "application/vnd.openstack.share+json;version=1"
                }
            ],
            "min_version": "",
            "status": "DEPRECATED",
            "updated": "2015-08-27T11:33:21Z",
            "version": ""
        },
        {
            "id": "v2.0",
            "links": [
                {
                    "href": "http://docs.openstack.org/",
                    "rel": "describedby",
                    "type": "text/html"
                },
                {
                    "href": "http://localhost:8786/v2/",
                    "rel": "self"
                }
            ],
            "media-types": [
                {
                    "base": "application/json",
                    "type": "application/vnd.openstack.share+json;version=1"
                }
            ],
            "min_version": "2.0",
            "status": "CURRENT",
            "updated": "2015-08-27T11:33:21Z",
            "version": "2.32"
        }
    ]
}
`

var ManilaAPIVersion1Result = apiversions.APIVersion{
	ID:      "v1.0",
	Status:  "DEPRECATED",
	Updated: time.Date(2015, 8, 27, 11, 33, 21, 0, time.UTC),
}

var ManilaAPIVersion2Result = apiversions.APIVersion{
	ID:         "v2.0",
	Status:     "CURRENT",
	Updated:    time.Date(2015, 8, 27, 11, 33, 21, 0, time.UTC),
	MinVersion: "2.0",
	Version:    "2.32",
}

var ManilaAllAPIVersionResults = []apiversions.APIVersion{
	ManilaAPIVersion1Result,
	ManilaAPIVersion2Result,
}

func MockListResponse(t *testing.T) {
	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, ManilaAllAPIVersionsResponse)
	})
}

func MockGetResponse(t *testing.T) {
	th.Mux.HandleFunc("/v2/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, ManilaAPIVersionResponse)
	})
}

func MockGetNoResponse(t *testing.T) {
	th.Mux.HandleFunc("/v2/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, ManilaAPIInvalidVersionResponse_1)
	})
}

func MockGetMultipleResponses(t *testing.T) {
	th.Mux.HandleFunc("/v2/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, ManilaAPIInvalidVersionResponse_2)
	})
}
