package testing

import (
	"encoding/json"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/dns/v2/recordsets"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListByZone(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListByZoneSuccessfully(t)

	count := 0
	err := recordsets.ListByZone(client.ServiceClient(), "2150b1bf-dee2-4221-9d85-11f7886fb15f", nil).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := recordsets.ExtractRecordSets(page)
		th.AssertNoErr(t, err)
		th.CheckDeepEquals(t, ExpectedRecordSetSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, count)
}

func TestListByZoneLimited(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListByZoneSuccessfully(t)

	count := 0
	listOpts := recordsets.ListOpts{
		Limit:  1,
		Marker: "f7b10e9b-0cae-4a91-b162-562bc6096648",
	}
	err := recordsets.ListByZone(client.ServiceClient(), "2150b1bf-dee2-4221-9d85-11f7886fb15f", listOpts).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := recordsets.ExtractRecordSets(page)
		th.AssertNoErr(t, err)
		th.CheckDeepEquals(t, ExpectedRecordSetSliceLimited, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, count)
}

func TestListByZoneAllPages(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListByZoneSuccessfully(t)

	allPages, err := recordsets.ListByZone(client.ServiceClient(), "2150b1bf-dee2-4221-9d85-11f7886fb15f", nil).AllPages()
	th.AssertNoErr(t, err)
	allRecordSets, err := recordsets.ExtractRecordSets(allPages)
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 2, len(allRecordSets))
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSuccessfully(t)

	actual, err := recordsets.Get(client.ServiceClient(), "2150b1bf-dee2-4221-9d85-11f7886fb15f", "f7b10e9b-0cae-4a91-b162-562bc6096648").Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &FirstRecordSet, actual)
}

func TestNextPageURL(t *testing.T) {
	var page recordsets.RecordSetPage
	var body map[string]interface{}
	err := json.Unmarshal([]byte(NextPageRequest), &body)
	if err != nil {
		t.Fatalf("Error unmarshaling data into page body: %v", err)
	}
	page.Body = body
	expected := "http://127.0.0.1:9001/v2/zones/2150b1bf-dee2-4221-9d85-11f7886fb15f/recordsets?limit=1&marker=f7b10e9b-0cae-4a91-b162-562bc6096648"
	actual, err := page.NextPageURL()
	th.AssertNoErr(t, err)
	th.CheckEquals(t, expected, actual)
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateSuccessfully(t)

	createOpts := recordsets.CreateOpts{
		Name:        "example.org.",
		Type:        "A",
		TTL:         3600,
		Description: "This is an example record set.",
		Records:     []string{"10.1.0.2"},
	}

	actual, err := recordsets.Create(client.ServiceClient(), "2150b1bf-dee2-4221-9d85-11f7886fb15f", createOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &CreatedRecordSet, actual)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateSuccessfully(t)

	updateOpts := recordsets.UpdateOpts{
		TTL:         0,
		Description: "Updated description",
		Records:     []string{"10.1.0.2", "10.1.0.3"},
	}

	UpdatedRecordSet := CreatedRecordSet
	UpdatedRecordSet.Status = "PENDING"
	UpdatedRecordSet.Action = "UPDATE"
	UpdatedRecordSet.Description = "Updated description"
	UpdatedRecordSet.Records = []string{"10.1.0.2", "10.1.0.3"}
	UpdatedRecordSet.Version = 2

	actual, err := recordsets.Update(client.ServiceClient(), UpdatedRecordSet.ZoneID, UpdatedRecordSet.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &UpdatedRecordSet, actual)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteSuccessfully(t)

	DeletedRecordSet := CreatedRecordSet
	DeletedRecordSet.Status = "PENDING"
	DeletedRecordSet.Action = "UPDATE"
	DeletedRecordSet.Description = "Updated description"
	DeletedRecordSet.Records = []string{"10.1.0.2", "10.1.0.3"}
	DeletedRecordSet.Version = 2

	err := recordsets.Delete(client.ServiceClient(), DeletedRecordSet.ZoneID, DeletedRecordSet.ID).ExtractErr()
	th.AssertNoErr(t, err)
	//th.CheckDeepEquals(t, &DeletedZone, actual)
}
