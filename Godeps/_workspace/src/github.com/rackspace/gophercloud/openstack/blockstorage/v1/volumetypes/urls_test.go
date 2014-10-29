package volumetypes

import (
	"testing"

	"github.com/rackspace/gophercloud"
	th "github.com/rackspace/gophercloud/testhelper"
)

const endpoint = "http://localhost:57909"

func endpointClient() *gophercloud.ServiceClient {
	return &gophercloud.ServiceClient{Endpoint: endpoint}
}

func TestListURL(t *testing.T) {
	actual := listURL(endpointClient())
	expected := endpoint + "types"
	th.AssertEquals(t, expected, actual)
}

func TestCreateURL(t *testing.T) {
	actual := createURL(endpointClient())
	expected := endpoint + "types"
	th.AssertEquals(t, expected, actual)
}

func TestGetURL(t *testing.T) {
	actual := getURL(endpointClient(), "foo")
	expected := endpoint + "types/foo"
	th.AssertEquals(t, expected, actual)
}

func TestDeleteURL(t *testing.T) {
	actual := deleteURL(endpointClient(), "foo")
	expected := endpoint + "types/foo"
	th.AssertEquals(t, expected, actual)
}
