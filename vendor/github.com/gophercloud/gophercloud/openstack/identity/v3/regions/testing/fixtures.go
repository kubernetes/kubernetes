package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/regions"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ListOutput provides a single page of Region results.
const ListOutput = `
{
    "links": {
        "next": null,
        "previous": null,
        "self": "http://example.com/identity/v3/regions"
    },
    "regions": [
        {
            "id": "RegionOne-East",
            "description": "East sub-region of RegionOne",
            "links": {
                "self": "http://example.com/identity/v3/regions/RegionOne-East"
            },
            "parent_region_id": "RegionOne"
        },
        {
            "id": "RegionOne-West",
            "description": "West sub-region of RegionOne",
            "links": {
                "self": "https://example.com/identity/v3/regions/RegionOne-West"
            },
            "extra": {
                "email": "westsupport@example.com"
            },
            "parent_region_id": "RegionOne"
        }
    ]
}
`

// GetOutput provides a Get result.
const GetOutput = `
{
    "region": {
        "id": "RegionOne-West",
        "description": "West sub-region of RegionOne",
        "links": {
            "self": "https://example.com/identity/v3/regions/RegionOne-West"
        },
        "name": "support",
        "extra": {
            "email": "westsupport@example.com"
        },
        "parent_region_id": "RegionOne"
    }
}
`

// CreateRequest provides the input to a Create request.
const CreateRequest = `
{
    "region": {
        "id": "RegionOne-West",
        "description": "West sub-region of RegionOne",
        "email": "westsupport@example.com",
        "parent_region_id": "RegionOne"
    }
}
`

/*
	// Due to a bug in Keystone, the Extra column of the Region table
	// is not updatable, see: https://bugs.launchpad.net/keystone/+bug/1729933
	// The following line should be added to region in UpdateRequest once the
	// fix is merged.

	"email": "1stwestsupport@example.com"
*/
// UpdateRequest provides the input to as Update request.
const UpdateRequest = `
{
    "region": {
        "description": "First West sub-region of RegionOne"
    }
}
`

/*
	// Due to a bug in Keystone, the Extra column of the Region table
	// is not updatable, see: https://bugs.launchpad.net/keystone/+bug/1729933
	// This following line should replace the email in UpdateOutput.extra once
	// the fix is merged.

	"email": "1stwestsupport@example.com"
*/
// UpdateOutput provides an update result.
const UpdateOutput = `
{
    "region": {
        "id": "RegionOne-West",
        "links": {
            "self": "https://example.com/identity/v3/regions/RegionOne-West"
        },
        "description": "First West sub-region of RegionOne",
        "extra": {
            "email": "westsupport@example.com"
        },
        "parent_region_id": "RegionOne"
    }
}
`

// FirstRegion is the first region in the List request.
var FirstRegion = regions.Region{
	ID: "RegionOne-East",
	Links: map[string]interface{}{
		"self": "http://example.com/identity/v3/regions/RegionOne-East",
	},
	Description:    "East sub-region of RegionOne",
	Extra:          map[string]interface{}{},
	ParentRegionID: "RegionOne",
}

// SecondRegion is the second region in the List request.
var SecondRegion = regions.Region{
	ID: "RegionOne-West",
	Links: map[string]interface{}{
		"self": "https://example.com/identity/v3/regions/RegionOne-West",
	},
	Description: "West sub-region of RegionOne",
	Extra: map[string]interface{}{
		"email": "westsupport@example.com",
	},
	ParentRegionID: "RegionOne",
}

/*
	// Due to a bug in Keystone, the Extra column of the Region table
	// is not updatable, see: https://bugs.launchpad.net/keystone/+bug/1729933
	// This should replace the email in SecondRegionUpdated.Extra once the fix
	// is merged.

	"email": "1stwestsupport@example.com"
*/
// SecondRegionUpdated is the second region in the List request.
var SecondRegionUpdated = regions.Region{
	ID: "RegionOne-West",
	Links: map[string]interface{}{
		"self": "https://example.com/identity/v3/regions/RegionOne-West",
	},
	Description: "First West sub-region of RegionOne",
	Extra: map[string]interface{}{
		"email": "westsupport@example.com",
	},
	ParentRegionID: "RegionOne",
}

// ExpectedRegionsSlice is the slice of regions expected to be returned from ListOutput.
var ExpectedRegionsSlice = []regions.Region{FirstRegion, SecondRegion}

// HandleListRegionsSuccessfully creates an HTTP handler at `/regions` on the
// test handler mux that responds with a list of two regions.
func HandleListRegionsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/regions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListOutput)
	})
}

// HandleGetRegionSuccessfully creates an HTTP handler at `/regions` on the
// test handler mux that responds with a single region.
func HandleGetRegionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/regions/RegionOne-West", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleCreateRegionSuccessfully creates an HTTP handler at `/regions` on the
// test handler mux that tests region creation.
func HandleCreateRegionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/regions", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, CreateRequest)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, GetOutput)
	})
}

// HandleUpdateRegionSuccessfully creates an HTTP handler at `/regions` on the
// test handler mux that tests region update.
func HandleUpdateRegionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/regions/RegionOne-West", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, UpdateRequest)

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, UpdateOutput)
	})
}

// HandleDeleteRegionSuccessfully creates an HTTP handler at `/regions` on the
// test handler mux that tests region deletion.
func HandleDeleteRegionSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/regions/RegionOne-West", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})
}
