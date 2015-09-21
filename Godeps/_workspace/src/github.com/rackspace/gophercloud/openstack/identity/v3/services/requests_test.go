package services

import (
	"fmt"
	"net/http"
	"reflect"
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestCreateSuccessful(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()

	testhelper.Mux.HandleFunc("/services", func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "POST")
		testhelper.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		testhelper.TestJSONRequest(t, r, `{ "type": "compute" }`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, `{
        "service": {
          "description": "Here's your service",
          "id": "1234",
          "name": "InscrutableOpenStackProjectName",
          "type": "compute"
        }
    }`)
	})

	result, err := Create(client.ServiceClient(), "compute").Extract()
	if err != nil {
		t.Fatalf("Unexpected error from Create: %v", err)
	}

	if result.Description == nil || *result.Description != "Here's your service" {
		t.Errorf("Service description was unexpected [%s]", *result.Description)
	}
	if result.ID != "1234" {
		t.Errorf("Service ID was unexpected [%s]", result.ID)
	}
	if result.Name != "InscrutableOpenStackProjectName" {
		t.Errorf("Service name was unexpected [%s]", result.Name)
	}
	if result.Type != "compute" {
		t.Errorf("Service type was unexpected [%s]", result.Type)
	}
}

func TestListSinglePage(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()

	testhelper.Mux.HandleFunc("/services", func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "GET")
		testhelper.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, `
			{
				"links": {
					"next": null,
					"previous": null
				},
				"services": [
					{
						"description": "Service One",
						"id": "1234",
						"name": "service-one",
						"type": "identity"
					},
					{
						"description": "Service Two",
						"id": "9876",
						"name": "service-two",
						"type": "compute"
					}
				]
			}
		`)
	})

	count := 0
	err := List(client.ServiceClient(), ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractServices(page)
		if err != nil {
			return false, err
		}

		desc0 := "Service One"
		desc1 := "Service Two"
		expected := []Service{
			Service{
				Description: &desc0,
				ID:          "1234",
				Name:        "service-one",
				Type:        "identity",
			},
			Service{
				Description: &desc1,
				ID:          "9876",
				Name:        "service-two",
				Type:        "compute",
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

func TestGetSuccessful(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()

	testhelper.Mux.HandleFunc("/services/12345", func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "GET")
		testhelper.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, `
			{
				"service": {
						"description": "Service One",
						"id": "12345",
						"name": "service-one",
						"type": "identity"
				}
			}
		`)
	})

	result, err := Get(client.ServiceClient(), "12345").Extract()
	if err != nil {
		t.Fatalf("Error fetching service information: %v", err)
	}

	if result.ID != "12345" {
		t.Errorf("Unexpected service ID: %s", result.ID)
	}
	if *result.Description != "Service One" {
		t.Errorf("Unexpected service description: [%s]", *result.Description)
	}
	if result.Name != "service-one" {
		t.Errorf("Unexpected service name: [%s]", result.Name)
	}
	if result.Type != "identity" {
		t.Errorf("Unexpected service type: [%s]", result.Type)
	}
}

func TestUpdateSuccessful(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()

	testhelper.Mux.HandleFunc("/services/12345", func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "PATCH")
		testhelper.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		testhelper.TestJSONRequest(t, r, `{ "type": "lasermagic" }`)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, `
			{
				"service": {
						"id": "12345",
						"type": "lasermagic"
				}
			}
		`)
	})

	result, err := Update(client.ServiceClient(), "12345", "lasermagic").Extract()
	if err != nil {
		t.Fatalf("Unable to update service: %v", err)
	}

	if result.ID != "12345" {
		t.Fatalf("Expected ID 12345, was %s", result.ID)
	}
}

func TestDeleteSuccessful(t *testing.T) {
	testhelper.SetupHTTP()
	defer testhelper.TeardownHTTP()

	testhelper.Mux.HandleFunc("/services/12345", func(w http.ResponseWriter, r *http.Request) {
		testhelper.TestMethod(t, r, "DELETE")
		testhelper.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.WriteHeader(http.StatusNoContent)
	})

	res := Delete(client.ServiceClient(), "12345")
	testhelper.AssertNoErr(t, res.Err)
}
