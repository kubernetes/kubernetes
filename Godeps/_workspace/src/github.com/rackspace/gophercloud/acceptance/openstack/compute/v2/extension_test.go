// +build acceptance compute extensionss

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestListExtensions(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	err = extensions.List(client).EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page ---")

		exts, err := extensions.ExtractExtensions(page)
		th.AssertNoErr(t, err)

		for i, ext := range exts {
			t.Logf("[%02d]    name=[%s]\n", i, ext.Name)
			t.Logf("       alias=[%s]\n", ext.Alias)
			t.Logf(" description=[%s]\n", ext.Description)
		}

		return true, nil
	})
	th.AssertNoErr(t, err)
}

func TestGetExtension(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	ext, err := extensions.Get(client, "os-admin-actions").Extract()
	th.AssertNoErr(t, err)

	t.Logf("Extension details:")
	t.Logf("        name=[%s]\n", ext.Name)
	t.Logf("   namespace=[%s]\n", ext.Namespace)
	t.Logf("       alias=[%s]\n", ext.Alias)
	t.Logf(" description=[%s]\n", ext.Description)
	t.Logf("     updated=[%s]\n", ext.Updated)
}
