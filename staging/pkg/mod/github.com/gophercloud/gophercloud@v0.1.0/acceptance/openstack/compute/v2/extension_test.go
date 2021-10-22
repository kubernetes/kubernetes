// +build acceptance compute extensions

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/common/extensions"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestExtensionsList(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	allPages, err := extensions.List(client).AllPages()
	th.AssertNoErr(t, err)

	allExtensions, err := extensions.ExtractExtensions(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, extension := range allExtensions {
		tools.PrintResource(t, extension)

		if extension.Name == "SchedulerHints" {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}

func TestExtensionsGet(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	extension, err := extensions.Get(client, "os-admin-actions").Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, extension)

	th.AssertEquals(t, extension.Name, "AdminActions")
}
