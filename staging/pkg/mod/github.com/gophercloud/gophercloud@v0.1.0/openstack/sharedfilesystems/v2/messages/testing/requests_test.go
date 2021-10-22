package testing

import (
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/messages"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// Verifies that message deletion works
func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockDeleteResponse(t)

	res := messages.Delete(client.ServiceClient(), "messageID")
	th.AssertNoErr(t, res.Err)
}

// Verifies that messages can be listed correctly
func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockListResponse(t)

	allPages, err := messages.List(client.ServiceClient(), &messages.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)
	actual, err := messages.ExtractMessages(allPages)
	th.AssertNoErr(t, err)
	expected := []messages.Message{
		{
			ResourceID:   "0d0b883f-95ef-406c-b930-55612ee48a6d",
			MessageLevel: "ERROR",
			UserMessage:  "create: Could not find an existing share server or allocate one on the share network provided. You may use a different share network, or verify the network details in the share network and retry your request. If this doesn't work, contact your administrator to troubleshoot issues with your network.",
			ExpiresAt:    time.Date(2019, 1, 6, 8, 53, 38, 0, time.UTC),
			ID:           "143a6cc2-1998-44d0-8356-22070b0ebdaa",
			CreatedAt:    time.Date(2018, 12, 7, 8, 53, 38, 0, time.UTC),
			DetailID:     "004",
			RequestID:    "req-21767eee-22ca-40a4-b6c0-ae7d35cd434f",
			ProjectID:    "a5e9d48232dc4aa59a716b5ced963584",
			ResourceType: "SHARE",
			ActionID:     "002",
		},
		{
			ResourceID:   "4336d74f-3bdc-4f27-9657-c01ec63680bf",
			MessageLevel: "ERROR",
			UserMessage:  "create: Could not find an existing share server or allocate one on the share network provided. You may use a different share network, or verify the network details in the share network and retry your request. If this doesn't work, contact your administrator to troubleshoot issues with your network.",
			ExpiresAt:    time.Date(2019, 1, 6, 8, 53, 34, 0, time.UTC),
			ID:           "2076373e-13a7-4b84-9e67-15ce8cceaff8",
			CreatedAt:    time.Date(2018, 12, 7, 8, 53, 34, 0, time.UTC),
			DetailID:     "004",
			RequestID:    "req-957792ed-f38b-42db-a86a-850f815cbbe9",
			ProjectID:    "a5e9d48232dc4aa59a716b5ced963584",
			ResourceType: "SHARE",
			ActionID:     "002",
		},
	}

	th.CheckDeepEquals(t, expected, actual)
}

// Verifies that messages list can be called with query parameters
func TestFilteredList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockFilteredListResponse(t)

	options := &messages.ListOpts{
		RequestID: "req-21767eee-22ca-40a4-b6c0-ae7d35cd434f",
	}

	allPages, err := messages.List(client.ServiceClient(), options).AllPages()
	th.AssertNoErr(t, err)
	actual, err := messages.ExtractMessages(allPages)
	th.AssertNoErr(t, err)
	expected := []messages.Message{
		{
			ResourceID:   "4336d74f-3bdc-4f27-9657-c01ec63680bf",
			MessageLevel: "ERROR",
			UserMessage:  "create: Could not find an existing share server or allocate one on the share network provided. You may use a different share network, or verify the network details in the share network and retry your request. If this doesn't work, contact your administrator to troubleshoot issues with your network.",
			ExpiresAt:    time.Date(2019, 1, 6, 8, 53, 34, 0, time.UTC),
			ID:           "2076373e-13a7-4b84-9e67-15ce8cceaff8",
			CreatedAt:    time.Date(2018, 12, 7, 8, 53, 34, 0, time.UTC),
			DetailID:     "004",
			RequestID:    "req-957792ed-f38b-42db-a86a-850f815cbbe9",
			ProjectID:    "a5e9d48232dc4aa59a716b5ced963584",
			ResourceType: "SHARE",
			ActionID:     "002",
		},
	}

	th.CheckDeepEquals(t, expected, actual)
}

// Verifies that it is possible to get a message
func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockGetResponse(t)

	expected := messages.Message{
		ResourceID:   "4336d74f-3bdc-4f27-9657-c01ec63680bf",
		MessageLevel: "ERROR",
		UserMessage:  "create: Could not find an existing share server or allocate one on the share network provided. You may use a different share network, or verify the network details in the share network and retry your request. If this doesn't work, contact your administrator to troubleshoot issues with your network.",
		ExpiresAt:    time.Date(2019, 1, 6, 8, 53, 34, 0, time.UTC),
		ID:           "2076373e-13a7-4b84-9e67-15ce8cceaff8",
		CreatedAt:    time.Date(2018, 12, 7, 8, 53, 34, 0, time.UTC),
		DetailID:     "004",
		RequestID:    "req-957792ed-f38b-42db-a86a-850f815cbbe9",
		ProjectID:    "a5e9d48232dc4aa59a716b5ced963584",
		ResourceType: "SHARE",
		ActionID:     "002",
	}

	n, err := messages.Get(client.ServiceClient(), "2076373e-13a7-4b84-9e67-15ce8cceaff8").Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, &expected, n)
}
