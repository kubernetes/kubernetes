package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/listeners"
	fake "github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/testhelper"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestListListeners(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListenerListSuccessfully(t)

	pages := 0
	err := listeners.List(fake.ServiceClient(), listeners.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := listeners.ExtractListeners(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 2 {
			t.Fatalf("Expected 2 listeners, got %d", len(actual))
		}
		th.CheckDeepEquals(t, ListenerWeb, actual[0])
		th.CheckDeepEquals(t, ListenerDb, actual[1])

		return true, nil
	})

	th.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestListAllListeners(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListenerListSuccessfully(t)

	allPages, err := listeners.List(fake.ServiceClient(), listeners.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)
	actual, err := listeners.ExtractListeners(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ListenerWeb, actual[0])
	th.CheckDeepEquals(t, ListenerDb, actual[1])
}

func TestCreateListener(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListenerCreationSuccessfully(t, SingleListenerBody)

	actual, err := listeners.Create(fake.ServiceClient(), listeners.CreateOpts{
		Protocol:               "TCP",
		Name:                   "db",
		LoadbalancerID:         "79e05663-7f03-45d2-a092-8b94062f22ab",
		AdminStateUp:           gophercloud.Enabled,
		DefaultTlsContainerRef: "2c433435-20de-4411-84ae-9cc8917def76",
		DefaultPoolID:          "41efe233-7591-43c5-9cf7-923964759f9e",
		ProtocolPort:           3306,
		InsertHeaders:          map[string]string{"X-Forwarded-For": "true"},
	}).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, ListenerDb, *actual)
}

func TestRequiredCreateOpts(t *testing.T) {
	res := listeners.Create(fake.ServiceClient(), listeners.CreateOpts{})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
	res = listeners.Create(fake.ServiceClient(), listeners.CreateOpts{Name: "foo"})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
	res = listeners.Create(fake.ServiceClient(), listeners.CreateOpts{Name: "foo", ProjectID: "bar"})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
	res = listeners.Create(fake.ServiceClient(), listeners.CreateOpts{Name: "foo", ProjectID: "bar", Protocol: "bar"})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
	res = listeners.Create(fake.ServiceClient(), listeners.CreateOpts{Name: "foo", ProjectID: "bar", Protocol: "bar", ProtocolPort: 80})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
}

func TestGetListener(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListenerGetSuccessfully(t)

	client := fake.ServiceClient()
	actual, err := listeners.Get(client, "4ec89087-d057-4e2c-911f-60a3b47ee304").Extract()
	if err != nil {
		t.Fatalf("Unexpected Get error: %v", err)
	}

	th.CheckDeepEquals(t, ListenerDb, *actual)
}

func TestDeleteListener(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListenerDeletionSuccessfully(t)

	res := listeners.Delete(fake.ServiceClient(), "4ec89087-d057-4e2c-911f-60a3b47ee304")
	th.AssertNoErr(t, res.Err)
}

func TestUpdateListener(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListenerUpdateSuccessfully(t)

	client := fake.ServiceClient()
	i1001 := 1001
	i181000 := 181000
	name := "NewListenerName"
	defaultPoolID := ""
	actual, err := listeners.Update(client, "4ec89087-d057-4e2c-911f-60a3b47ee304", listeners.UpdateOpts{
		Name:                 &name,
		ConnLimit:            &i1001,
		DefaultPoolID:        &defaultPoolID,
		TimeoutMemberData:    &i181000,
		TimeoutClientData:    &i181000,
		TimeoutMemberConnect: &i181000,
		TimeoutTCPInspect:    &i181000,
	}).Extract()
	if err != nil {
		t.Fatalf("Unexpected Update error: %v", err)
	}

	th.CheckDeepEquals(t, ListenerUpdated, *actual)
}

func TestGetListenerStatsTree(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListenerGetStatsTree(t)

	client := fake.ServiceClient()
	actual, err := listeners.GetStats(client, "4ec89087-d057-4e2c-911f-60a3b47ee304").Extract()
	if err != nil {
		t.Fatalf("Unexpected Get error: %v", err)
	}

	th.CheckDeepEquals(t, ListenerStatsTree, *actual)
}
