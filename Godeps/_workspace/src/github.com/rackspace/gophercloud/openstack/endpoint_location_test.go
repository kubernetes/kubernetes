package openstack

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/rackspace/gophercloud"
	tokens2 "github.com/rackspace/gophercloud/openstack/identity/v2/tokens"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

// Service catalog fixtures take too much vertical space!
var catalog2 = tokens2.ServiceCatalog{
	Entries: []tokens2.CatalogEntry{
		tokens2.CatalogEntry{
			Type: "same",
			Name: "same",
			Endpoints: []tokens2.Endpoint{
				tokens2.Endpoint{
					Region:      "same",
					PublicURL:   "https://public.correct.com/",
					InternalURL: "https://internal.correct.com/",
					AdminURL:    "https://admin.correct.com/",
				},
				tokens2.Endpoint{
					Region:    "different",
					PublicURL: "https://badregion.com/",
				},
			},
		},
		tokens2.CatalogEntry{
			Type: "same",
			Name: "different",
			Endpoints: []tokens2.Endpoint{
				tokens2.Endpoint{
					Region:    "same",
					PublicURL: "https://badname.com/",
				},
				tokens2.Endpoint{
					Region:    "different",
					PublicURL: "https://badname.com/+badregion",
				},
			},
		},
		tokens2.CatalogEntry{
			Type: "different",
			Name: "different",
			Endpoints: []tokens2.Endpoint{
				tokens2.Endpoint{
					Region:    "same",
					PublicURL: "https://badtype.com/+badname",
				},
				tokens2.Endpoint{
					Region:    "different",
					PublicURL: "https://badtype.com/+badregion+badname",
				},
			},
		},
	},
}

func TestV2EndpointExact(t *testing.T) {
	expectedURLs := map[gophercloud.Availability]string{
		gophercloud.AvailabilityPublic:   "https://public.correct.com/",
		gophercloud.AvailabilityAdmin:    "https://admin.correct.com/",
		gophercloud.AvailabilityInternal: "https://internal.correct.com/",
	}

	for availability, expected := range expectedURLs {
		actual, err := V2EndpointURL(&catalog2, gophercloud.EndpointOpts{
			Type:         "same",
			Name:         "same",
			Region:       "same",
			Availability: availability,
		})
		th.AssertNoErr(t, err)
		th.CheckEquals(t, expected, actual)
	}
}

func TestV2EndpointNone(t *testing.T) {
	_, err := V2EndpointURL(&catalog2, gophercloud.EndpointOpts{
		Type:         "nope",
		Availability: gophercloud.AvailabilityPublic,
	})
	th.CheckEquals(t, gophercloud.ErrEndpointNotFound, err)
}

func TestV2EndpointMultiple(t *testing.T) {
	_, err := V2EndpointURL(&catalog2, gophercloud.EndpointOpts{
		Type:         "same",
		Region:       "same",
		Availability: gophercloud.AvailabilityPublic,
	})
	if !strings.HasPrefix(err.Error(), "Discovered 2 matching endpoints:") {
		t.Errorf("Received unexpected error: %v", err)
	}
}

func TestV2EndpointBadAvailability(t *testing.T) {
	_, err := V2EndpointURL(&catalog2, gophercloud.EndpointOpts{
		Type:         "same",
		Name:         "same",
		Region:       "same",
		Availability: "wat",
	})
	th.CheckEquals(t, err.Error(), "Unexpected availability in endpoint query: wat")
}

func setupV3Responses(t *testing.T) {
	// Mock the service query.
	th.Mux.HandleFunc("/services", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, `
			{
				"links": {
					"next": null,
					"previous": null
				},
				"services": [
					{
						"description": "Correct",
						"id": "1234",
						"name": "same",
						"type": "same"
					},
					{
						"description": "Bad Name",
						"id": "9876",
						"name": "different",
						"type": "same"
					}
				]
			}
		`)
	})

	// Mock the endpoint query.
	th.Mux.HandleFunc("/endpoints", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestFormValues(t, r, map[string]string{
			"service_id": "1234",
			"interface":  "public",
		})

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, `
			{
				"endpoints": [
					{
						"id": "12",
						"interface": "public",
						"name": "the-right-one",
						"region": "same",
						"service_id": "1234",
						"url": "https://correct:9000/"
					},
					{
						"id": "14",
						"interface": "public",
						"name": "bad-region",
						"region": "different",
						"service_id": "1234",
						"url": "https://bad-region:9001/"
					}
				],
				"links": {
					"next": null,
					"previous": null
				}
			}
    `)
	})
}

func TestV3EndpointExact(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	setupV3Responses(t)

	actual, err := V3EndpointURL(fake.ServiceClient(), gophercloud.EndpointOpts{
		Type:         "same",
		Name:         "same",
		Region:       "same",
		Availability: gophercloud.AvailabilityPublic,
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, actual, "https://correct:9000/")
}

func TestV3EndpointNoService(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/services", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, `
      {
        "links": {
          "next": null,
          "previous": null
        },
        "services": []
      }
    `)
	})

	_, err := V3EndpointURL(fake.ServiceClient(), gophercloud.EndpointOpts{
		Type:         "nope",
		Name:         "same",
		Region:       "same",
		Availability: gophercloud.AvailabilityPublic,
	})
	th.CheckEquals(t, gophercloud.ErrServiceNotFound, err)
}
