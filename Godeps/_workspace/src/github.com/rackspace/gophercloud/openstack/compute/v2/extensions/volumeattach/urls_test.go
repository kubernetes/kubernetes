package volumeattach

import (
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestListURL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	c := client.ServiceClient()
	serverId := "4d8c3732-a248-40ed-bebc-539a6ffd25c0"

	th.CheckEquals(t, c.Endpoint+"servers/"+serverId+"/os-volume_attachments", listURL(c, serverId))
}

func TestCreateURL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	c := client.ServiceClient()
	serverId := "4d8c3732-a248-40ed-bebc-539a6ffd25c0"

	th.CheckEquals(t, c.Endpoint+"servers/"+serverId+"/os-volume_attachments", createURL(c, serverId))
}

func TestGetURL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	c := client.ServiceClient()
	serverId := "4d8c3732-a248-40ed-bebc-539a6ffd25c0"
	aId := "a26887c6-c47b-4654-abb5-dfadf7d3f804"

	th.CheckEquals(t, c.Endpoint+"servers/"+serverId+"/os-volume_attachments/"+aId, getURL(c, serverId, aId))
}

func TestDeleteURL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	c := client.ServiceClient()
	serverId := "4d8c3732-a248-40ed-bebc-539a6ffd25c0"
	aId := "a26887c6-c47b-4654-abb5-dfadf7d3f804"

	th.CheckEquals(t, c.Endpoint+"servers/"+serverId+"/os-volume_attachments/"+aId, deleteURL(c, serverId, aId))
}
