package volumetypes

import (
	"testing"

	os "github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumetypes"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	os.MockListResponse(t)

	count := 0

	err := List(fake.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractVolumeTypes(page)
		if err != nil {
			t.Errorf("Failed to extract volume types: %v", err)
			return false, err
		}

		expected := []VolumeType{
			VolumeType{
				ID:   "289da7f8-6440-407c-9fb4-7db01ec49164",
				Name: "vol-type-001",
				ExtraSpecs: map[string]interface{}{
					"capabilities": "gpu",
				},
			},
			VolumeType{
				ID:         "96c3bda7-c82a-4f50-be73-ca7621794835",
				Name:       "vol-type-002",
				ExtraSpecs: map[string]interface{}{},
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	th.AssertEquals(t, 1, count)
	th.AssertNoErr(t, err)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	os.MockGetResponse(t)

	vt, err := Get(fake.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, vt.ExtraSpecs, map[string]interface{}{"serverNumber": "2"})
	th.AssertEquals(t, vt.Name, "vol-type-001")
	th.AssertEquals(t, vt.ID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
}
