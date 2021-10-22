package testing

import (
	"strings"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/clustering/v1/nodes"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestCreateNode(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCreateSuccessfully(t)

	createOpts := nodes.CreateOpts{
		ClusterID: "e395be1e-8d8e-43bb-bd6c-943eccf76a6d",
		Metadata: map[string]interface{}{
			"foo": "bar",
			"test": map[string]interface{}{
				"nil_interface": interface{}(nil),
				"float_value":   float64(123.3),
				"string_value":  "test_string",
				"bool_value":    false,
			},
		},
		Name:      "node-e395be1e-002",
		ProfileID: "d8a48377-f6a3-4af4-bbbb-6e8bcaa0cbc0",
		Role:      "",
	}

	res := nodes.Create(fake.ServiceClient(), createOpts)
	th.AssertNoErr(t, res.Err)

	requestID := res.Header.Get("X-Openstack-Request-Id")
	th.AssertEquals(t, "req-3791a089-9d46-4671-a3f9-55e95e55d2b4", requestID)

	location := res.Header.Get("Location")
	th.AssertEquals(t, "http://senlin.cloud.blizzard.net:8778/v1/actions/ffd94dd8-6266-4887-9a8c-5b78b72136da", location)

	locationFields := strings.Split(location, "actions/")
	actionID := locationFields[1]
	th.AssertEquals(t, "ffd94dd8-6266-4887-9a8c-5b78b72136da", actionID)

	actual, err := res.Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedCreate, *actual)
}

func TestListNodes(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleListSuccessfully(t)

	count := 0
	err := nodes.List(fake.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		actual, err := nodes.ExtractNodes(page)
		th.AssertNoErr(t, err)
		th.AssertDeepEquals(t, ExpectedList, actual)
		count++
		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, count, 1)
}

func TestDeleteNode(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleDeleteSuccessfully(t)

	deleteResult := nodes.Delete(fake.ServiceClient(), "6dc6d336e3fc4c0a951b5698cd1236ee")
	th.AssertNoErr(t, deleteResult.ExtractErr())
}

func TestGetNode(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGetSuccessfully(t)

	actual, err := nodes.Get(fake.ServiceClient(), "573aa1ba-bf45-49fd-907d-6b5d6e6adfd3").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedGet, *actual)
}

func TestUpdateNode(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleUpdateSuccessfully(t)

	nodeOpts := nodes.UpdateOpts{
		Name: "node-e395be1e-002",
	}
	actual, err := nodes.Update(fake.ServiceClient(), ExpectedUpdate.ID, nodeOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedUpdate, *actual)
}

func TestOpsNode(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleOpsSuccessfully(t)

	nodeOpts := nodes.OperationOpts{
		Operation: nodes.PauseOperation,
	}
	actual, err := nodes.Ops(fake.ServiceClient(), "7d85f602-a948-4a30-afd4-e84f47471c15", nodeOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, OperationExpectedActionID, actual)
}

func TestNodeRecover(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleRecoverSuccessfully(t)
	recoverOpts := nodes.RecoverOpts{
		Operation: nodes.RebuildRecovery,
		Check:     new(bool),
	}
	actionID, err := nodes.Recover(fake.ServiceClient(), "edce3528-864f-41fb-8759-f4707925cc09", recoverOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, ExpectedActionID, actionID)
}

func TestNodeCheck(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCheckSuccessfully(t)

	actionID, err := nodes.Check(fake.ServiceClient(), "edce3528-864f-41fb-8759-f4707925cc09").Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, ExpectedActionID, actionID)
}
