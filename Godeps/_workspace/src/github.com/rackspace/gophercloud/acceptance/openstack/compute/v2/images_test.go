// +build acceptance compute images

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud/openstack/compute/v2/images"
	"github.com/rackspace/gophercloud/pagination"
)

func TestListImages(t *testing.T) {
	client, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute: client: %v", err)
	}

	t.Logf("ID\tRegion\tName\tStatus\tCreated")

	pager := images.ListDetail(client, nil)
	count, pages := 0, 0
	pager.EachPage(func(page pagination.Page) (bool, error) {
		pages++
		images, err := images.ExtractImages(page)
		if err != nil {
			return false, err
		}

		for _, i := range images {
			t.Logf("%s\t%s\t%s\t%s", i.ID, i.Name, i.Status, i.Created)
		}

		return true, nil
	})

	t.Logf("--------\n%d images listed on %d pages.", count, pages)
}
