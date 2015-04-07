// +build acceptance

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/images"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestListImages(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	count := 0
	err = images.ListDetail(client, nil).EachPage(func(page pagination.Page) (bool, error) {
		count++
		t.Logf("-- Page %02d --", count)

		is, err := images.ExtractImages(page)
		th.AssertNoErr(t, err)

		for i, image := range is {
			t.Logf("[%02d]   id=[%s]", i, image.ID)
			t.Logf("     name=[%s]", image.Name)
			t.Logf("  created=[%s]", image.Created)
			t.Logf("  updated=[%s]", image.Updated)
			t.Logf(" min disk=[%d]", image.MinDisk)
			t.Logf("  min RAM=[%d]", image.MinRAM)
			t.Logf(" progress=[%d]", image.Progress)
			t.Logf("   status=[%s]", image.Status)
		}

		return true, nil
	})
	th.AssertNoErr(t, err)
	if count < 1 {
		t.Errorf("Expected at least one page of images.")
	}
}

func TestGetImage(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	options, err := optionsFromEnv()
	th.AssertNoErr(t, err)

	image, err := images.Get(client, options.imageID).Extract()
	th.AssertNoErr(t, err)

	t.Logf("Requested image:")
	t.Logf("       id=[%s]", image.ID)
	t.Logf("     name=[%s]", image.Name)
	t.Logf("  created=[%s]", image.Created)
	t.Logf("  updated=[%s]", image.Updated)
	t.Logf(" min disk=[%d]", image.MinDisk)
	t.Logf("  min RAM=[%d]", image.MinRAM)
	t.Logf(" progress=[%d]", image.Progress)
	t.Logf("   status=[%s]", image.Status)
}
