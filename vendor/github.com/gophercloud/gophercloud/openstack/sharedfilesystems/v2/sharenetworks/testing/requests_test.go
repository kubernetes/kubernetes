package testing

import (
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/sharenetworks"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// Verifies that a share network can be created correctly
func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockCreateResponse(t)

	options := &sharenetworks.CreateOpts{
		Name:            "my_network",
		Description:     "This is my share network",
		NeutronNetID:    "998b42ee-2cee-4d36-8b95-67b5ca1f2109",
		NeutronSubnetID: "53482b62-2c84-4a53-b6ab-30d9d9800d06",
	}

	n, err := sharenetworks.Create(client.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Name, "my_network")
	th.AssertEquals(t, n.Description, "This is my share network")
	th.AssertEquals(t, n.NeutronNetID, "998b42ee-2cee-4d36-8b95-67b5ca1f2109")
	th.AssertEquals(t, n.NeutronSubnetID, "53482b62-2c84-4a53-b6ab-30d9d9800d06")
}

// Verifies that share network deletion works
func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockDeleteResponse(t)

	res := sharenetworks.Delete(client.ServiceClient(), "fa158a3d-6d9f-4187-9ca5-abbb82646eb2")
	th.AssertNoErr(t, res.Err)
}

// Verifies that share networks can be listed correctly
func TestListDetail(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockListResponse(t)

	allPages, err := sharenetworks.ListDetail(client.ServiceClient(), &sharenetworks.ListOpts{}).AllPages()

	th.AssertNoErr(t, err)
	actual, err := sharenetworks.ExtractShareNetworks(allPages)
	th.AssertNoErr(t, err)

	var nilTime time.Time
	expected := []sharenetworks.ShareNetwork{
		{
			ID:              "32763294-e3d4-456a-998d-60047677c2fb",
			Name:            "net_my1",
			CreatedAt:       gophercloud.JSONRFC3339MilliNoZ(time.Date(2015, 9, 4, 14, 57, 13, 0, time.UTC)),
			Description:     "descr",
			NetworkType:     "",
			CIDR:            "",
			NovaNetID:       "",
			NeutronNetID:    "998b42ee-2cee-4d36-8b95-67b5ca1f2109",
			NeutronSubnetID: "53482b62-2c84-4a53-b6ab-30d9d9800d06",
			IPVersion:       0,
			SegmentationID:  0,
			UpdatedAt:       gophercloud.JSONRFC3339MilliNoZ(nilTime),
			ProjectID:       "16e1ab15c35a457e9c2b2aa189f544e1",
		},
		{
			ID:              "713df749-aac0-4a54-af52-10f6c991e80c",
			Name:            "net_my",
			CreatedAt:       gophercloud.JSONRFC3339MilliNoZ(time.Date(2015, 9, 4, 14, 54, 25, 0, time.UTC)),
			Description:     "desecr",
			NetworkType:     "",
			CIDR:            "",
			NovaNetID:       "",
			NeutronNetID:    "998b42ee-2cee-4d36-8b95-67b5ca1f2109",
			NeutronSubnetID: "53482b62-2c84-4a53-b6ab-30d9d9800d06",
			IPVersion:       0,
			SegmentationID:  0,
			UpdatedAt:       gophercloud.JSONRFC3339MilliNoZ(nilTime),
			ProjectID:       "16e1ab15c35a457e9c2b2aa189f544e1",
		},
		{
			ID:              "fa158a3d-6d9f-4187-9ca5-abbb82646eb2",
			Name:            "",
			CreatedAt:       gophercloud.JSONRFC3339MilliNoZ(time.Date(2015, 9, 4, 14, 51, 41, 0, time.UTC)),
			Description:     "",
			NetworkType:     "",
			CIDR:            "",
			NovaNetID:       "",
			NeutronNetID:    "",
			NeutronSubnetID: "",
			IPVersion:       0,
			SegmentationID:  0,
			UpdatedAt:       gophercloud.JSONRFC3339MilliNoZ(nilTime),
			ProjectID:       "16e1ab15c35a457e9c2b2aa189f544e1",
		},
	}

	th.CheckDeepEquals(t, expected, actual)
}

// Verifies that share networks list can be called with query parameters
func TestPaginatedListDetail(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockFilteredListResponse(t)

	options := &sharenetworks.ListOpts{
		Offset: 0,
		Limit:  1,
	}

	count := 0

	err := sharenetworks.ListDetail(client.ServiceClient(), options).EachPage(func(page pagination.Page) (bool, error) {
		count++
		_, err := sharenetworks.ExtractShareNetworks(page)
		if err != nil {
			t.Errorf("Failed to extract share networks: %v", err)
			return false, err
		}

		return true, nil
	})
	th.AssertNoErr(t, err)

	th.AssertEquals(t, count, 3)
}

// Verifies that it is possible to get a share network
func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockGetResponse(t)

	var nilTime time.Time
	expected := sharenetworks.ShareNetwork{
		ID:              "7f950b52-6141-4a08-bbb5-bb7ffa3ea5fd",
		Name:            "net_my1",
		CreatedAt:       gophercloud.JSONRFC3339MilliNoZ(time.Date(2015, 9, 4, 14, 56, 45, 0, time.UTC)),
		Description:     "descr",
		NetworkType:     "",
		CIDR:            "",
		NovaNetID:       "",
		NeutronNetID:    "998b42ee-2cee-4d36-8b95-67b5ca1f2109",
		NeutronSubnetID: "53482b62-2c84-4a53-b6ab-30d9d9800d06",
		IPVersion:       0,
		SegmentationID:  0,
		UpdatedAt:       gophercloud.JSONRFC3339MilliNoZ(nilTime),
		ProjectID:       "16e1ab15c35a457e9c2b2aa189f544e1",
	}

	n, err := sharenetworks.Get(client.ServiceClient(), "7f950b52-6141-4a08-bbb5-bb7ffa3ea5fd").Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, &expected, n)
}
