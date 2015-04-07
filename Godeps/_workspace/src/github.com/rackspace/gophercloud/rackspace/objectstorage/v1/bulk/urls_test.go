package bulk

import (
	"testing"

	"github.com/rackspace/gophercloud"
	th "github.com/rackspace/gophercloud/testhelper"
)

const endpoint = "http://localhost:57909/"

func endpointClient() *gophercloud.ServiceClient {
	return &gophercloud.ServiceClient{Endpoint: endpoint}
}

func TestDeleteURL(t *testing.T) {
	actual := deleteURL(endpointClient())
	expected := endpoint + "?bulk-delete"
	th.CheckEquals(t, expected, actual)
}

func TestExtractURL(t *testing.T) {
	actual := extractURL(endpointClient(), "tar")
	expected := endpoint + "?extract-archive=tar"
	th.CheckEquals(t, expected, actual)
}
