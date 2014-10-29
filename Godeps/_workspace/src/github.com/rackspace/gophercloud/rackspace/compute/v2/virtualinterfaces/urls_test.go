package virtualinterfaces

import (
	"testing"

	"github.com/rackspace/gophercloud"
	th "github.com/rackspace/gophercloud/testhelper"
)

const endpoint = "http://localhost:57909/"

func endpointClient() *gophercloud.ServiceClient {
	return &gophercloud.ServiceClient{Endpoint: endpoint}
}

func TestCreateURL(t *testing.T) {
	actual := createURL(endpointClient(), "12345")
	expected := endpoint + "servers/12345/os-virtual-interfacesv2"
	th.AssertEquals(t, expected, actual)
}

func TestListURL(t *testing.T) {
	actual := createURL(endpointClient(), "12345")
	expected := endpoint + "servers/12345/os-virtual-interfacesv2"
	th.AssertEquals(t, expected, actual)
}

func TestDeleteURL(t *testing.T) {
	actual := deleteURL(endpointClient(), "12345", "6789")
	expected := endpoint + "servers/12345/os-virtual-interfacesv2/6789"
	th.AssertEquals(t, expected, actual)
}
