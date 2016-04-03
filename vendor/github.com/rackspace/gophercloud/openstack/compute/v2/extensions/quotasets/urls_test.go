package quotasets

import (
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestGetURL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	c := client.ServiceClient()

	th.CheckEquals(t, c.Endpoint+"os-quota-sets/wat", getURL(c, "wat"))
}
