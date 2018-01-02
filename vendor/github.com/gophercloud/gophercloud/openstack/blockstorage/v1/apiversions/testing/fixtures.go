package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func MockListResponse(t *testing.T) {
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
}

func MockGetResponse(t *testing.T) {
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
}
