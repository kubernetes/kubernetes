package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const (
	shareEndpoint = "/shares"
	shareID       = "011d21e2-fbc3-4e4a-9993-9ea223f73264"
)

var createRequest = `{
		"share": {
			"name": "my_test_share",
			"size": 1,
			"share_proto": "NFS"
		}
	}`

var createResponse = `{
		"share": {
			"name": "my_test_share",
			"share_proto": "NFS",
			"size": 1,
			"status": null,
			"share_server_id": null,
			"project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
			"share_type": "25747776-08e5-494f-ab40-a64b9d20d8f7",
			"share_type_name": "default",
			"availability_zone": null,
			"created_at": "2015-09-18T10:25:24.533287",
			"export_location": null,
			"links": [
				{
					"href": "http://172.18.198.54:8786/v1/16e1ab15c35a457e9c2b2aa189f544e1/shares/011d21e2-fbc3-4e4a-9993-9ea223f73264",
					"rel": "self"
				},
				{
					"href": "http://172.18.198.54:8786/16e1ab15c35a457e9c2b2aa189f544e1/shares/011d21e2-fbc3-4e4a-9993-9ea223f73264",
					"rel": "bookmark"
				}
			],
			"share_network_id": null,
			"export_locations": [],
			"host": null,
			"access_rules_status": "active",
			"has_replicas": false,
			"replication_type": null,
			"task_state": null,
			"snapshot_support": true,
			"consistency_group_id": "9397c191-8427-4661-a2e8-b23820dc01d4",
			"source_cgsnapshot_member_id": null,
			"volume_type": "default",
			"snapshot_id": null,
			"is_public": true,
			"metadata": {
				"project": "my_app",
				"aim": "doc"
			},
			"id": "011d21e2-fbc3-4e4a-9993-9ea223f73264",
			"description": "My custom share London"
		}
	}`

// MockCreateResponse creates a mock response
func MockCreateResponse(t *testing.T) {
	th.Mux.HandleFunc(shareEndpoint, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, createRequest)
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, createResponse)
	})
}

// MockDeleteResponse creates a mock delete response
func MockDeleteResponse(t *testing.T) {
	th.Mux.HandleFunc(shareEndpoint+"/"+shareID, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}

var getResponse = `{
    "share": {
        "links": [
            {
                "href": "http://172.18.198.54:8786/v2/16e1ab15c35a457e9c2b2aa189f544e1/shares/011d21e2-fbc3-4e4a-9993-9ea223f73264",
                "rel": "self"
            },
            {
                "href": "http://172.18.198.54:8786/16e1ab15c35a457e9c2b2aa189f544e1/shares/011d21e2-fbc3-4e4a-9993-9ea223f73264",
                "rel": "bookmark"
            }
        ],
        "availability_zone": "nova",
        "share_network_id": "713df749-aac0-4a54-af52-10f6c991e80c",
        "share_server_id": "e268f4aa-d571-43dd-9ab3-f49ad06ffaef",
        "snapshot_id": null,
        "id": "011d21e2-fbc3-4e4a-9993-9ea223f73264",
        "size": 1,
        "share_type": "25747776-08e5-494f-ab40-a64b9d20d8f7",
        "share_type_name": "default",
        "consistency_group_id": "9397c191-8427-4661-a2e8-b23820dc01d4",
        "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
        "metadata": {
            "project": "my_app",
            "aim": "doc"
        },
        "status": "available",
        "description": "My custom share London",
        "host": "manila2@generic1#GENERIC1",
        "has_replicas": false,
        "replication_type": null,
        "task_state": null,
        "is_public": true,
        "snapshot_support": true,
        "name": "my_test_share",
        "created_at": "2015-09-18T10:25:24.000000",
        "share_proto": "NFS",
        "volume_type": "default",
        "source_cgsnapshot_member_id": null
    }
}`

// MockGetResponse creates a mock get response
func MockGetResponse(t *testing.T) {
	th.Mux.HandleFunc(shareEndpoint+"/"+shareID, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, getResponse)
	})
}

var getExportLocationsResponse = `{
    "export_locations": [
        {
		"path": "127.0.0.1:/var/lib/manila/mnt/share-9a922036-ad26-4d27-b955-7a1e285fa74d",
        	"share_instance_id": "011d21e2-fbc3-4e4a-9993-9ea223f73264",
		"is_admin_only": false,
        	"id": "80ed63fc-83bc-4afc-b881-da4a345ac83d",
		"preferred": false
	}
    ]
}`

// MockGetExportLocationsResponse creates a mock get export locations response
func MockGetExportLocationsResponse(t *testing.T) {
	th.Mux.HandleFunc(shareEndpoint+"/"+shareID+"/export_locations", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, getExportLocationsResponse)
	})
}

var grantAccessRequest = `{
		"allow_access": {
			"access_type": "ip",
			"access_to": "0.0.0.0/0",
			"access_level": "rw"
		}
	}`

var grantAccessResponse = `{
    "access": {
	"share_id": "011d21e2-fbc3-4e4a-9993-9ea223f73264",
	"access_type": "ip",
	"access_to": "0.0.0.0/0",
	"access_key": "",
	"access_level": "rw",
	"state": "new",
	"id": "a2f226a5-cee8-430b-8a03-78a59bd84ee8"
    }
}`

// MockGrantAccessResponse creates a mock grant access response
func MockGrantAccessResponse(t *testing.T) {
	th.Mux.HandleFunc(shareEndpoint+"/"+shareID+"/action", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, grantAccessRequest)
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, grantAccessResponse)
	})
}
