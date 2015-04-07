package subnets

import (
	"testing"

	"github.com/rackspace/gophercloud"
	th "github.com/rackspace/gophercloud/testhelper"
)

const endpoint = "http://localhost:57909/"

func endpointClient() *gophercloud.ServiceClient {
	return &gophercloud.ServiceClient{Endpoint: endpoint, ResourceBase: endpoint + "v2.0/"}
}

func TestListURL(t *testing.T) {
	actual := listURL(endpointClient())
	expected := endpoint + "v2.0/subnets"
	th.AssertEquals(t, expected, actual)
}

func TestGetURL(t *testing.T) {
	actual := getURL(endpointClient(), "foo")
	expected := endpoint + "v2.0/subnets/foo"
	th.AssertEquals(t, expected, actual)
}

func TestCreateURL(t *testing.T) {
	actual := createURL(endpointClient())
	expected := endpoint + "v2.0/subnets"
	th.AssertEquals(t, expected, actual)
}

func TestUpdateURL(t *testing.T) {
	actual := updateURL(endpointClient(), "foo")
	expected := endpoint + "v2.0/subnets/foo"
	th.AssertEquals(t, expected, actual)
}

func TestDeleteURL(t *testing.T) {
	actual := deleteURL(endpointClient(), "foo")
	expected := endpoint + "v2.0/subnets/foo"
	th.AssertEquals(t, expected, actual)
}
