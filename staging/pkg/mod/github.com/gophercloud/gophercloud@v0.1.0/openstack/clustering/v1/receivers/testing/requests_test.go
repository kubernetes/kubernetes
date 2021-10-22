package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/clustering/v1/receivers"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestCreateReceiver(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCreateSuccessfully(t)

	opts := receivers.CreateOpts{
		Name:      "cluster_inflate",
		ClusterID: "ae63a10b-4a90-452c-aef1-113a0b255ee3",
		Type:      receivers.WebhookReceiver,
		Action:    "CLUSTER_SCALE_OUT",
		Actor:     map[string]interface{}{},
		Params:    map[string]interface{}{},
	}

	actual, err := receivers.Create(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedReceiver, *actual)
}

func TestGetReceivers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGetSuccessfully(t)

	actual, err := receivers.Get(fake.ServiceClient(), "573aa1ba-bf45-49fd-907d-6b5d6e6adfd3").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedReceiver, *actual)
}

func TestUpdateReceiver(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleUpdateSuccessfully(t)

	opts := receivers.UpdateOpts{
		Name:   "cluster_inflate",
		Action: "CLUSTER_SCALE_OUT",
		Params: map[string]interface{}{
			"count": "2",
		},
	}
	actual, err := receivers.Update(fake.ServiceClient(), "6dc6d336e3fc4c0a951b5698cd1236ee", opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedUpdateReceiver, *actual)
}

func TestListReceivers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleListSuccessfully(t)

	opts := receivers.ListOpts{
		Limit: 2,
		Sort:  "name:asc,status:desc",
	}

	count := 0
	receivers.List(fake.ServiceClient(), opts).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := receivers.ExtractReceivers(page)
		th.AssertNoErr(t, err)

		th.AssertDeepEquals(t, ExpectedReceiversList, actual)

		return true, nil
	})

	th.AssertEquals(t, count, 1)
}

func TestDeleteReceiver(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleDeleteSuccessfully(t)

	deleteResult := receivers.Delete(fake.ServiceClient(), "6dc6d336e3fc4c0a951b5698cd1236ee")
	th.AssertNoErr(t, deleteResult.ExtractErr())
}

func TestNotifyReceivers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleNotifySuccessfully(t)

	requestID, err := receivers.Notify(fake.ServiceClient(), "6dc6d336e3fc4c0a951b5698cd1236ee").Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, ExpectedNotifyRequestID, requestID)
}
