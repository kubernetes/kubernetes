// +build acceptance

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/flavors"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestListFlavors(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	count := 0
	err = flavors.ListDetail(client, nil).EachPage(func(page pagination.Page) (bool, error) {
		count++
		t.Logf("-- Page %0d --", count)

		fs, err := flavors.ExtractFlavors(page)
		th.AssertNoErr(t, err)

		for i, flavor := range fs {
			t.Logf("[%02d]      id=[%s]", i, flavor.ID)
			t.Logf("        name=[%s]", flavor.Name)
			t.Logf("        disk=[%d]", flavor.Disk)
			t.Logf("         RAM=[%d]", flavor.RAM)
			t.Logf(" rxtx_factor=[%f]", flavor.RxTxFactor)
			t.Logf("        swap=[%d]", flavor.Swap)
			t.Logf("       VCPUs=[%d]", flavor.VCPUs)
		}

		return true, nil
	})
	th.AssertNoErr(t, err)
	if count == 0 {
		t.Errorf("No flavors listed!")
	}
}

func TestGetFlavor(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	options, err := optionsFromEnv()
	th.AssertNoErr(t, err)

	flavor, err := flavors.Get(client, options.flavorID).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Requested flavor:")
	t.Logf("          id=[%s]", flavor.ID)
	t.Logf("        name=[%s]", flavor.Name)
	t.Logf("        disk=[%d]", flavor.Disk)
	t.Logf("         RAM=[%d]", flavor.RAM)
	t.Logf(" rxtx_factor=[%f]", flavor.RxTxFactor)
	t.Logf("        swap=[%d]", flavor.Swap)
	t.Logf("       VCPUs=[%d]", flavor.VCPUs)
}
