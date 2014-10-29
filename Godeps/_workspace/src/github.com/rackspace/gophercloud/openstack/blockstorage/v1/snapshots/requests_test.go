package snapshots

import (
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockListResponse(t)

	count := 0

	List(client.ServiceClient(), &ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractSnapshots(page)
		if err != nil {
			t.Errorf("Failed to extract snapshots: %v", err)
			return false, err
		}

		expected := []Snapshot{
			Snapshot{
				ID:   "289da7f8-6440-407c-9fb4-7db01ec49164",
				Name: "snapshot-001",
			},
			Snapshot{
				ID:   "96c3bda7-c82a-4f50-be73-ca7621794835",
				Name: "snapshot-002",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockGetResponse(t)

	v, err := Get(client.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, v.Name, "snapshot-001")
	th.AssertEquals(t, v.ID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockCreateResponse(t)

	options := CreateOpts{VolumeID: "1234", Name: "snapshot-001"}
	n, err := Create(client.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.VolumeID, "1234")
	th.AssertEquals(t, n.Name, "snapshot-001")
	th.AssertEquals(t, n.ID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
}

func TestUpdateMetadata(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockUpdateMetadataResponse(t)

	expected := map[string]interface{}{"key": "v1"}

	options := &UpdateMetadataOpts{
		Metadata: map[string]interface{}{
			"key": "v1",
		},
	}

	actual, err := UpdateMetadata(client.ServiceClient(), "123", options).ExtractMetadata()

	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, actual, expected)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockDeleteResponse(t)

	res := Delete(client.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22")
	th.AssertNoErr(t, res.Err)
}
