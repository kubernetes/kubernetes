// +build acceptance

package v1

import (
	"testing"
	"time"

	"github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumetypes"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestVolumeTypes(t *testing.T) {
	client, err := newClient(t)
	th.AssertNoErr(t, err)

	vt, err := volumetypes.Create(client, &volumetypes.CreateOpts{
		ExtraSpecs: map[string]interface{}{
			"capabilities": "gpu",
			"priority":     3,
		},
		Name: "gophercloud-test-volumeType",
	}).Extract()
	th.AssertNoErr(t, err)
	defer func() {
		time.Sleep(10000 * time.Millisecond)
		err = volumetypes.Delete(client, vt.ID).ExtractErr()
		if err != nil {
			t.Error(err)
			return
		}
	}()
	t.Logf("Created volume type: %+v\n", vt)

	vt, err = volumetypes.Get(client, vt.ID).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Got volume type: %+v\n", vt)

	err = volumetypes.List(client).EachPage(func(page pagination.Page) (bool, error) {
		volTypes, err := volumetypes.ExtractVolumeTypes(page)
		if len(volTypes) != 1 {
			t.Errorf("Expected 1 volume type, got %d", len(volTypes))
		}
		t.Logf("Listing volume types: %+v\n", volTypes)
		return true, err
	})
	th.AssertNoErr(t, err)
}
