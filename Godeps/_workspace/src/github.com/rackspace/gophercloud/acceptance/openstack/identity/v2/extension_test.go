// +build acceptance identity

package v2

import (
	"testing"

	extensions2 "github.com/rackspace/gophercloud/openstack/identity/v2/extensions"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestEnumerateExtensions(t *testing.T) {
	service := authenticatedClient(t)

	t.Logf("Extensions available on this identity endpoint:")
	count := 0
	err := extensions2.List(service).EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page %02d ---", count)

		extensions, err := extensions2.ExtractExtensions(page)
		th.AssertNoErr(t, err)

		for i, ext := range extensions {
			t.Logf("[%02d] name=[%s] namespace=[%s]", i, ext.Name, ext.Namespace)
			t.Logf("     alias=[%s] updated=[%s]", ext.Alias, ext.Updated)
			t.Logf("     description=[%s]", ext.Description)
		}

		count++
		return true, nil
	})
	th.AssertNoErr(t, err)
}

func TestGetExtension(t *testing.T) {
	service := authenticatedClient(t)

	ext, err := extensions2.Get(service, "OS-KSCRUD").Extract()
	th.AssertNoErr(t, err)

	th.CheckEquals(t, "OpenStack Keystone User CRUD", ext.Name)
	th.CheckEquals(t, "http://docs.openstack.org/identity/api/ext/OS-KSCRUD/v1.0", ext.Namespace)
	th.CheckEquals(t, "OS-KSCRUD", ext.Alias)
	th.CheckEquals(t, "OpenStack extensions to Keystone v2.0 API enabling User Operations.", ext.Description)
}
