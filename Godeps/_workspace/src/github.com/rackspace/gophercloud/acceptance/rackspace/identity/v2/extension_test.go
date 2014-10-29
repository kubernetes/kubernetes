// +build acceptance

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	extensions2 "github.com/rackspace/gophercloud/rackspace/identity/v2/extensions"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestExtensions(t *testing.T) {
	service := authenticatedClient(t)

	t.Logf("Extensions available on this identity endpoint:")
	count := 0
	var chosen string
	err := extensions2.List(service).EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page %02d ---", count)

		extensions, err := extensions2.ExtractExtensions(page)
		th.AssertNoErr(t, err)

		for i, ext := range extensions {
			if chosen == "" {
				chosen = ext.Alias
			}

			t.Logf("[%02d] name=[%s] namespace=[%s]", i, ext.Name, ext.Namespace)
			t.Logf("     alias=[%s] updated=[%s]", ext.Alias, ext.Updated)
			t.Logf("     description=[%s]", ext.Description)
		}

		count++
		return true, nil
	})
	th.AssertNoErr(t, err)

	if chosen == "" {
		t.Logf("No extensions found.")
		return
	}

	ext, err := extensions2.Get(service, chosen).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Detail for extension [%s]:", chosen)
	t.Logf("        name=[%s]", ext.Name)
	t.Logf("   namespace=[%s]", ext.Namespace)
	t.Logf("       alias=[%s]", ext.Alias)
	t.Logf("     updated=[%s]", ext.Updated)
	t.Logf(" description=[%s]", ext.Description)
}
