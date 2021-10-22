package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/messaging/v2/queues"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListSuccessfully(t)

	listOpts := queues.ListOpts{
		Limit: 1,
	}

	count := 0
	err := queues.List(fake.ServiceClient(), listOpts).EachPage(func(page pagination.Page) (bool, error) {
		actual, err := queues.ExtractQueues(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, ExpectedQueueSlice[count], actual)
		count++

		return true, nil
	})
	th.AssertNoErr(t, err)

	th.CheckEquals(t, 2, count)
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateSuccessfully(t)

	createOpts := queues.CreateOpts{
		QueueName:                  QueueName,
		MaxMessagesPostSize:        262144,
		DefaultMessageTTL:          3600,
		DefaultMessageDelay:        30,
		DeadLetterQueue:            "dead_letter",
		DeadLetterQueueMessagesTTL: 3600,
		MaxClaimCount:              10,
		Extra:                      map[string]interface{}{"description": "Queue for unit testing."},
	}

	err := queues.Create(fake.ServiceClient(), createOpts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateSuccessfully(t)

	updateOpts := queues.BatchUpdateOpts{
		queues.UpdateOpts{
			Op:    queues.ReplaceOp,
			Path:  "/metadata/description",
			Value: "Update queue description",
		},
	}
	updatedQueueResult := queues.QueueDetails{
		Extra: map[string]interface{}{"description": "Update queue description"},
	}

	actual, err := queues.Update(fake.ServiceClient(), QueueName, updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, updatedQueueResult, actual)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSuccessfully(t)

	actual, err := queues.Get(fake.ServiceClient(), QueueName).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, QueueDetails, actual)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteSuccessfully(t)

	err := queues.Delete(fake.ServiceClient(), QueueName).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestGetStat(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetStatsSuccessfully(t)

	actual, err := queues.GetStats(fake.ServiceClient(), QueueName).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedStats, actual)
}

func TestShare(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleShareSuccessfully(t)

	shareOpts := queues.ShareOpts{
		Paths:   []queues.SharePath{queues.PathMessages, queues.PathClaims, queues.PathSubscriptions},
		Methods: []queues.ShareMethod{queues.MethodGet, queues.MethodPost, queues.MethodPut, queues.MethodPatch},
		Expires: "2016-09-01T00:00:00",
	}

	actual, err := queues.Share(fake.ServiceClient(), QueueName, shareOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedShare, actual)
}

func TestPurge(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePurgeSuccessfully(t)

	purgeOpts := queues.PurgeOpts{
		ResourceTypes: []queues.PurgeResource{queues.ResourceMessages, queues.ResourceSubscriptions},
	}

	err := queues.Purge(fake.ServiceClient(), QueueName, purgeOpts).ExtractErr()
	th.AssertNoErr(t, err)
}
