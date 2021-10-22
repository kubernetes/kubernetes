package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/blockstorage/v3/volumetypes"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListAll(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockListResponse(t)
	pages := 0
	err := volumetypes.List(client.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		pages++
		actual, err := volumetypes.ExtractVolumeTypes(page)
		if err != nil {
			return false, err
		}
		expected := []volumetypes.VolumeType{
			{
				ID:           "6685584b-1eac-4da6-b5c3-555430cf68ff",
				Name:         "SSD",
				ExtraSpecs:   map[string]string{"volume_backend_name": "lvmdriver-1"},
				IsPublic:     true,
				Description:  "",
				QosSpecID:    "",
				PublicAccess: true,
			}, {
				ID:           "8eb69a46-df97-4e41-9586-9a40a7533803",
				Name:         "SATA",
				ExtraSpecs:   map[string]string{"volume_backend_name": "lvmdriver-1"},
				IsPublic:     true,
				Description:  "",
				QosSpecID:    "",
				PublicAccess: true,
			},
		}
		th.CheckDeepEquals(t, expected, actual)
		return true, nil
	})
	th.AssertNoErr(t, err)
	th.AssertEquals(t, pages, 1)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockGetResponse(t)

	v, err := volumetypes.Get(client.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, v.Name, "vol-type-001")
	th.AssertEquals(t, v.ID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
	th.AssertEquals(t, v.ExtraSpecs["capabilities"], "gpu")
	th.AssertEquals(t, v.QosSpecID, "d32019d3-bc6e-4319-9c1d-6722fc136a22")
	th.AssertEquals(t, v.PublicAccess, true)
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockCreateResponse(t)

	var isPublic = true

	options := &volumetypes.CreateOpts{
		Name:        "test_type",
		IsPublic:    &isPublic,
		Description: "test_type_desc",
		ExtraSpecs:  map[string]string{"capabilities": "gpu"},
	}

	n, err := volumetypes.Create(client.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Name, "test_type")
	th.AssertEquals(t, n.Description, "test_type_desc")
	th.AssertEquals(t, n.IsPublic, true)
	th.AssertEquals(t, n.PublicAccess, true)
	th.AssertEquals(t, n.ID, "6d0ff92a-0007-4780-9ece-acfe5876966a")
	th.AssertEquals(t, n.ExtraSpecs["capabilities"], "gpu")
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockDeleteResponse(t)

	res := volumetypes.Delete(client.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22")
	th.AssertNoErr(t, res.Err)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockUpdateResponse(t)

	var isPublic = true
	var name = "vol-type-002"
	options := volumetypes.UpdateOpts{
		Name:     &name,
		IsPublic: &isPublic,
	}

	v, err := volumetypes.Update(client.ServiceClient(), "d32019d3-bc6e-4319-9c1d-6722fc136a22", options).Extract()
	th.AssertNoErr(t, err)
	th.CheckEquals(t, "vol-type-002", v.Name)
	th.CheckEquals(t, true, v.IsPublic)
}
