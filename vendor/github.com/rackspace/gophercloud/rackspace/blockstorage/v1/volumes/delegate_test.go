package volumes

import (
	"testing"

	"github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes"
	os "github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes/testing"
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

	th.AssertEquals(t, 1, count)
	th.AssertNoErr(t, err)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	os.MockGetResponse(t)

	v, err := Get(fake.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, v.Name, "vol-001")
	th.AssertEquals(t, v.ID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	os.MockCreateResponse(t)

	n, err := Create(fake.ServiceClient(), CreateOpts{volumes.CreateOpts{Size: 75}}).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Size, 4)
	th.AssertEquals(t, n.ID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
}

func TestSizeRange(t *testing.T) {
	_, err := Create(fake.ServiceClient(), CreateOpts{volumes.CreateOpts{Size: 1}}).Extract()
	if err == nil {
		t.Fatalf("Expected error, got none")
	}

	_, err = Create(fake.ServiceClient(), CreateOpts{volumes.CreateOpts{Size: 2000}}).Extract()
	if err == nil {
		t.Fatalf("Expected error, got none")
	}
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	os.MockDeleteResponse(t)

	res := Delete(fake.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22")
	th.AssertNoErr(t, res.Err)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	os.MockUpdateResponse(t)

	options := &UpdateOpts{Name: "vol-002"}
	v, err := Update(fake.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22", options).Extract()
	th.AssertNoErr(t, err)
	th.CheckEquals(t, "vol-002", v.Name)
}
