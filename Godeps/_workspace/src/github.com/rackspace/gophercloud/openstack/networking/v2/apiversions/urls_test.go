package apiversions

import (
	"testing"

	"github.com/rackspace/gophercloud"
	th "github.com/rackspace/gophercloud/testhelper"
)

const endpoint = "http://localhost:57909/"

func endpointClient() *gophercloud.ServiceClient {
	return &gophercloud.ServiceClient{Endpoint: endpoint}
}

func TestAPIVersionsURL(t *testing.T) {
	actual := apiVersionsURL(endpointClient())
	expected := endpoint
	th.AssertEquals(t, expected, actual)
}

func TestAPIInfoURL(t *testing.T) {
	actual := apiInfoURL(endpointClient(), "v2.0")
	expected := endpoint + "v2.0/"
	th.AssertEquals(t, expected, actual)
}
