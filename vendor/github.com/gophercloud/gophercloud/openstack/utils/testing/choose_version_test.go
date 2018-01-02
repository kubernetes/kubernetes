package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/utils"
	"github.com/gophercloud/gophercloud/testhelper"
)

func setupVersionHandler() {
	testhelper.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `
			{
				"versions": {
					"values": [
						{
							"status": "stable",
							"id": "v3.0",
							"links": [
								{ "href": "%s/v3.0", "rel": "self" }
							]
						},
						{
							"status": "stable",
							"id": "v2.0",
							"links": [
								{ "href": "%s/v2.0", "rel": "self" }
							]
						}
					]
				}
			}
		`, testhelper.Server.URL, testhelper.Server.URL)
	})
}

func TestChooseVersion(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	setupVersionHandler()

	v2 := &utils.Version{ID: "v2.0", Priority: 2, Suffix: "blarg"}
	v3 := &utils.Version{ID: "v3.0", Priority: 3, Suffix: "hargl"}

	c := &gophercloud.ProviderClient{
		IdentityBase:     testhelper.Endpoint(),
		IdentityEndpoint: "",
	}
	v, endpoint, err := utils.ChooseVersion(c, []*utils.Version{v2, v3})

	if err != nil {
		t.Fatalf("Unexpected error from ChooseVersion: %v", err)
	}

	if v != v3 {
		t.Errorf("Expected %#v to win, but %#v did instead", v3, v)
	}

	expected := testhelper.Endpoint() + "v3.0/"
	if endpoint != expected {
		t.Errorf("Expected endpoint [%s], but was [%s] instead", expected, endpoint)
	}
}

func TestChooseVersionOpinionatedLink(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()
	setupVersionHandler()

	v2 := &utils.Version{ID: "v2.0", Priority: 2, Suffix: "nope"}
	v3 := &utils.Version{ID: "v3.0", Priority: 3, Suffix: "northis"}

	c := &gophercloud.ProviderClient{
		IdentityBase:     testhelper.Endpoint(),
		IdentityEndpoint: testhelper.Endpoint() + "v2.0/",
	}
	v, endpoint, err := utils.ChooseVersion(c, []*utils.Version{v2, v3})
	if err != nil {
		t.Fatalf("Unexpected error from ChooseVersion: %v", err)
	}

	if v != v2 {
		t.Errorf("Expected %#v to win, but %#v did instead", v2, v)
	}

	expected := testhelper.Endpoint() + "v2.0/"
	if endpoint != expected {
		t.Errorf("Expected endpoint [%s], but was [%s] instead", expected, endpoint)
	}
}

func TestChooseVersionFromSuffix(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()

	v2 := &utils.Version{ID: "v2.0", Priority: 2, Suffix: "/v2.0/"}
	v3 := &utils.Version{ID: "v3.0", Priority: 3, Suffix: "/v3.0/"}

	c := &gophercloud.ProviderClient{
		IdentityBase:     testhelper.Endpoint(),
		IdentityEndpoint: testhelper.Endpoint() + "v2.0/",
	}
	v, endpoint, err := utils.ChooseVersion(c, []*utils.Version{v2, v3})
	if err != nil {
		t.Fatalf("Unexpected error from ChooseVersion: %v", err)
	}

	if v != v2 {
		t.Errorf("Expected %#v to win, but %#v did instead", v2, v)
	}

	expected := testhelper.Endpoint() + "v2.0/"
	if endpoint != expected {
		t.Errorf("Expected endpoint [%s], but was [%s] instead", expected, endpoint)
	}
}
