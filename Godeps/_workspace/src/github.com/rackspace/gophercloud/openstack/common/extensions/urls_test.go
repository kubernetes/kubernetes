package extensions

import (
	"testing"

	"github.com/rackspace/gophercloud"
	th "github.com/rackspace/gophercloud/testhelper"
)

const endpoint = "http://localhost:57909/"

func endpointClient() *gophercloud.ServiceClient {
	return &gophercloud.ServiceClient{Endpoint: endpoint}
}

func TestExtensionURL(t *testing.T) {
	actual := ExtensionURL(endpointClient(), "agent")
	expected := endpoint + "extensions/agent"
	th.AssertEquals(t, expected, actual)
}

func TestListExtensionURL(t *testing.T) {
	actual := ListExtensionURL(endpointClient())
	expected := endpoint + "extensions"
	th.AssertEquals(t, expected, actual)
}
