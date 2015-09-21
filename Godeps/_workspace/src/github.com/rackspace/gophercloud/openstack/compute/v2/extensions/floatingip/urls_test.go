package floatingip

import (
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestListURL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	c := client.ServiceClient()

	th.CheckEquals(t, c.Endpoint+"os-floating-ips", listURL(c))
}

func TestCreateURL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	c := client.ServiceClient()

	th.CheckEquals(t, c.Endpoint+"os-floating-ips", createURL(c))
}

func TestGetURL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	c := client.ServiceClient()
	id := "1"

	th.CheckEquals(t, c.Endpoint+"os-floating-ips/"+id, getURL(c, id))
}

func TestDeleteURL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	c := client.ServiceClient()
	id := "1"

	th.CheckEquals(t, c.Endpoint+"os-floating-ips/"+id, deleteURL(c, id))
}

func TestAssociateURL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	c := client.ServiceClient()
	serverId := "4d8c3732-a248-40ed-bebc-539a6ffd25c0"

	th.CheckEquals(t, c.Endpoint+"servers/"+serverId+"/action", associateURL(c, serverId))
}

func TestDisassociateURL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	c := client.ServiceClient()
	serverId := "4d8c3732-a248-40ed-bebc-539a6ffd25c0"

	th.CheckEquals(t, c.Endpoint+"servers/"+serverId+"/action", disassociateURL(c, serverId))
}
