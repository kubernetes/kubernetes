package testing

import (
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/imageservice/v2/members"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fakeclient "github.com/gophercloud/gophercloud/testhelper/client"
)

const createdAtString = "2013-09-20T19:22:19Z"
const updatedAtString = "2013-09-20T19:25:31Z"

func TestCreateMemberSuccessfully(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCreateImageMemberSuccessfully(t)
	im, err := members.Create(fakeclient.ServiceClient(), "da3b75d9-3f4a-40e7-8a2c-bfab23927dea",
		"8989447062e04a818baf9e073fd04fa7").Extract()
	th.AssertNoErr(t, err)

	createdAt, err := time.Parse(time.RFC3339, createdAtString)
	th.AssertNoErr(t, err)

	updatedAt, err := time.Parse(time.RFC3339, updatedAtString)
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, members.Member{
		CreatedAt: createdAt,
		ImageID:   "da3b75d9-3f4a-40e7-8a2c-bfab23927dea",
		MemberID:  "8989447062e04a818baf9e073fd04fa7",
		Schema:    "/v2/schemas/member",
		Status:    "pending",
		UpdatedAt: updatedAt,
	}, *im)

}

func TestMemberListSuccessfully(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleImageMemberList(t)

	pager := members.List(fakeclient.ServiceClient(), "da3b75d9-3f4a-40e7-8a2c-bfab23927dea")
	t.Logf("Pager state %v", pager)
	count, pages := 0, 0
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		pages++
		t.Logf("Page %v", page)
		members, err := members.ExtractMembers(page)
		if err != nil {
			return false, err
		}

		for _, i := range members {
			t.Logf("%s\t%s\t%s\t%s\t\n", i.ImageID, i.MemberID, i.Status, i.Schema)
			count++
		}

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, pages)
	th.AssertEquals(t, 2, count)
}

func TestMemberListEmpty(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleImageMemberEmptyList(t)

	pager := members.List(fakeclient.ServiceClient(), "da3b75d9-3f4a-40e7-8a2c-bfab23927dea")
	t.Logf("Pager state %v", pager)
	count, pages := 0, 0
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		pages++
		t.Logf("Page %v", page)
		members, err := members.ExtractMembers(page)
		if err != nil {
			return false, err
		}

		for _, i := range members {
			t.Logf("%s\t%s\t%s\t%s\t\n", i.ImageID, i.MemberID, i.Status, i.Schema)
			count++
		}

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 0, pages)
	th.AssertEquals(t, 0, count)
}

func TestShowMemberDetails(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleImageMemberDetails(t)
	md, err := members.Get(fakeclient.ServiceClient(),
		"da3b75d9-3f4a-40e7-8a2c-bfab23927dea",
		"8989447062e04a818baf9e073fd04fa7").Extract()

	th.AssertNoErr(t, err)
	if md == nil {
		t.Errorf("Expected non-nil value for md")
	}

	createdAt, err := time.Parse(time.RFC3339, "2013-11-26T07:21:21Z")
	th.AssertNoErr(t, err)

	updatedAt, err := time.Parse(time.RFC3339, "2013-11-26T07:21:21Z")
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, members.Member{
		CreatedAt: createdAt,
		ImageID:   "da3b75d9-3f4a-40e7-8a2c-bfab23927dea",
		MemberID:  "8989447062e04a818baf9e073fd04fa7",
		Schema:    "/v2/schemas/member",
		Status:    "pending",
		UpdatedAt: updatedAt,
	}, *md)
}

func TestDeleteMember(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	counter := HandleImageMemberDeleteSuccessfully(t)

	result := members.Delete(fakeclient.ServiceClient(), "da3b75d9-3f4a-40e7-8a2c-bfab23927dea",
		"8989447062e04a818baf9e073fd04fa7")
	th.AssertEquals(t, 1, counter.Counter)
	th.AssertNoErr(t, result.Err)
}

func TestMemberUpdateSuccessfully(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	counter := HandleImageMemberUpdate(t)
	im, err := members.Update(fakeclient.ServiceClient(), "da3b75d9-3f4a-40e7-8a2c-bfab23927dea",
		"8989447062e04a818baf9e073fd04fa7",
		members.UpdateOpts{
			Status: "accepted",
		}).Extract()
	th.AssertEquals(t, 1, counter.Counter)
	th.AssertNoErr(t, err)

	createdAt, err := time.Parse(time.RFC3339, "2013-11-26T07:21:21Z")
	th.AssertNoErr(t, err)

	updatedAt, err := time.Parse(time.RFC3339, "2013-11-26T07:21:21Z")
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, members.Member{
		CreatedAt: createdAt,
		ImageID:   "da3b75d9-3f4a-40e7-8a2c-bfab23927dea",
		MemberID:  "8989447062e04a818baf9e073fd04fa7",
		Schema:    "/v2/schemas/member",
		Status:    "accepted",
		UpdatedAt: updatedAt,
	}, *im)

}
