package endpoints

import (
	"testing"

	"github.com/rackspace/gophercloud"
)

func TestGetListURL(t *testing.T) {
	client := gophercloud.ServiceClient{Endpoint: "http://localhost:5000/v3/"}
	url := listURL(&client)
	if url != "http://localhost:5000/v3/endpoints" {
		t.Errorf("Unexpected list URL generated: [%s]", url)
	}
}

func TestGetEndpointURL(t *testing.T) {
	client := gophercloud.ServiceClient{Endpoint: "http://localhost:5000/v3/"}
	url := endpointURL(&client, "1234")
	if url != "http://localhost:5000/v3/endpoints/1234" {
		t.Errorf("Unexpected service URL generated: [%s]", url)
	}
}
