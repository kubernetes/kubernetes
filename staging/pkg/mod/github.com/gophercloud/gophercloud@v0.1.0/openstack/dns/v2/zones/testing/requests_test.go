package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/dns/v2/zones"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListSuccessfully(t)

	count := 0
	err := zones.List(client.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := zones.ExtractZones(page)
		th.AssertNoErr(t, err)
		th.CheckDeepEquals(t, ExpectedZonesSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, count)
}

func TestListAllPages(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListSuccessfully(t)

	allPages, err := zones.List(client.ServiceClient(), nil).AllPages()
	th.AssertNoErr(t, err)
	allZones, err := zones.ExtractZones(allPages)
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 2, len(allZones))
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSuccessfully(t)

	actual, err := zones.Get(client.ServiceClient(), "a86dba58-0043-4cc6-a1bb-69d5e86f3ca3").Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &FirstZone, actual)
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateSuccessfully(t)

	createOpts := zones.CreateOpts{
		Name:        "example.org.",
		Email:       "joe@example.org",
		Type:        "PRIMARY",
		TTL:         7200,
		Description: "This is an example zone.",
	}

	actual, err := zones.Create(client.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &CreatedZone, actual)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateSuccessfully(t)

	var description = "Updated Description"
	updateOpts := zones.UpdateOpts{
		TTL:         600,
		Description: &description,
	}

	UpdatedZone := CreatedZone
	UpdatedZone.Status = "PENDING"
	UpdatedZone.Action = "UPDATE"
	UpdatedZone.TTL = 600
	UpdatedZone.Description = "Updated Description"

	actual, err := zones.Update(client.ServiceClient(), UpdatedZone.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &UpdatedZone, actual)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteSuccessfully(t)

	DeletedZone := CreatedZone
	DeletedZone.Status = "PENDING"
	DeletedZone.Action = "DELETE"
	DeletedZone.TTL = 600
	DeletedZone.Description = "Updated Description"

	actual, err := zones.Delete(client.ServiceClient(), DeletedZone.ID).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &DeletedZone, actual)
}
