package testing

import (
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/blockstorage/v1/volumes"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockListResponse(t)

	count := 0

	volumes.List(client.ServiceClient(), &volumes.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := volumes.ExtractVolumes(page)
		if err != nil {
			t.Errorf("Failed to extract volumes: %v", err)
			return false, err
		}

		expected := []volumes.Volume{
			{
				ID:   "289da7f8-6440-407c-9fb4-7db01ec49164",
				Name: "vol-001",
			},
			{
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

	MockListResponse(t)

	allPages, err := volumes.List(client.ServiceClient(), &volumes.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)
	actual, err := volumes.ExtractVolumes(allPages)
	th.AssertNoErr(t, err)

	expected := []volumes.Volume{
		{
			ID:   "289da7f8-6440-407c-9fb4-7db01ec49164",
			Name: "vol-001",
		},
		{
			ID:   "96c3bda7-c82a-4f50-be73-ca7621794835",
			Name: "vol-002",
		},
	}

	th.CheckDeepEquals(t, expected, actual)

}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockGetResponse(t)

	actual, err := volumes.Get(client.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").Extract()
	th.AssertNoErr(t, err)

	expected := &volumes.Volume{
		Status: "active",
		Name:   "vol-001",
		Attachments: []map[string]interface{}{
			{
				"attachment_id": "03987cd1-0ad5-40d1-9b2a-7cc48295d4fa",
				"id":            "47e9ecc5-4045-4ee3-9a4b-d859d546a0cf",
				"volume_id":     "6c80f8ac-e3e2-480c-8e6e-f1db92fe4bfe",
				"server_id":     "d1c4788b-9435-42e2-9b81-29f3be1cd01f",
				"host_name":     "mitaka",
				"device":        "/",
			},
		},
		AvailabilityZone: "us-east1",
		Bootable:         "false",
		CreatedAt:        time.Date(2012, 2, 14, 20, 53, 07, 0, time.UTC),
		Description:      "Another volume.",
		VolumeType:       "289da7f8-6440-407c-9fb4-7db01ec49164",
		SnapshotID:       "",
		SourceVolID:      "",
		Metadata: map[string]string{
			"contents": "junk",
		},
		ID:   "521752a6-acf6-4b2d-bc7a-119f9148cd8c",
		Size: 30,
	}

	th.AssertDeepEquals(t, expected, actual)
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockCreateResponse(t)

	options := &volumes.CreateOpts{
		Size:             75,
		AvailabilityZone: "us-east1",
	}
	n, err := volumes.Create(client.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Size, 4)
	th.AssertEquals(t, n.ID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockDeleteResponse(t)

	res := volumes.Delete(client.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22")
	th.AssertNoErr(t, res.Err)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockUpdateResponse(t)

	var name = "vol-002"
	options := volumes.UpdateOpts{Name: &name}
	v, err := volumes.Update(client.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22", options).Extract()
	th.AssertNoErr(t, err)
	th.CheckEquals(t, "vol-002", v.Name)
}
