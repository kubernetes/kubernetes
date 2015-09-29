package volumes

import (
	"testing"

	fixtures "github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes/testing"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	fixtures.MockListResponse(t)

	count := 0

	List(client.ServiceClient(), &ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractVolumes(page)
		if err != nil {
			t.Errorf("Failed to extract volumes: %v", err)
			return false, err
		}

		expected := []Volume{
			Volume{
				ID:   "289da7f8-6440-407c-9fb4-7db01ec49164",
				Name: "vol-001",
			},
			Volume{
				ID:   "96c3bda7-c82a-4f50-be73-ca7621794835",
				Name: "vol-002",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestListAll(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	fixtures.MockListResponse(t)

	allPages, err := List(client.ServiceClient(), &ListOpts{}).AllPages()
	th.AssertNoErr(t, err)
	actual, err := ExtractVolumes(allPages)
	th.AssertNoErr(t, err)

	expected := []Volume{
		Volume{
			ID:   "289da7f8-6440-407c-9fb4-7db01ec49164",
			Name: "vol-001",
		},
		Volume{
			ID:   "96c3bda7-c82a-4f50-be73-ca7621794835",
			Name: "vol-002",
		},
	}

	th.CheckDeepEquals(t, expected, actual)

}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	fixtures.MockGetResponse(t)

	v, err := Get(client.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, v.Name, "vol-001")
	th.AssertEquals(t, v.ID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
	th.AssertEquals(t, v.Attachments[0]["device"], "/dev/vde")
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	fixtures.MockCreateResponse(t)

	options := &CreateOpts{Size: 75}
	n, err := Create(client.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Size, 4)
	th.AssertEquals(t, n.ID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	fixtures.MockDeleteResponse(t)

	res := Delete(client.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22")
	th.AssertNoErr(t, res.Err)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	fixtures.MockUpdateResponse(t)

	options := UpdateOpts{Name: "vol-002"}
	v, err := Update(client.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22", options).Extract()
	th.AssertNoErr(t, err)
	th.CheckEquals(t, "vol-002", v.Name)
}
