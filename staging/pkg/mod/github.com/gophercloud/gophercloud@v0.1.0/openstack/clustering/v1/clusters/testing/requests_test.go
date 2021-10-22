package testing

import (
	"strings"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/clustering/v1/clusters"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestCreateCluster(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCreateClusterSuccessfully(t)

	minSize := 1
	opts := clusters.CreateOpts{
		Name:            "cluster1",
		DesiredCapacity: 3,
		ProfileID:       "d8a48377-f6a3-4af4-bbbb-6e8bcaa0cbc0",
		MinSize:         &minSize,
		MaxSize:         20,
		Timeout:         3600,
		Metadata:        map[string]interface{}{},
		Config:          map[string]interface{}{},
	}

	res := clusters.Create(fake.ServiceClient(), opts)
	th.AssertNoErr(t, res.Err)

	location := res.Header.Get("Location")
	th.AssertEquals(t, "http://senlin.cloud.blizzard.net:8778/v1/actions/625628cd-f877-44be-bde0-fec79f84e13d", location)

	locationFields := strings.Split(location, "actions/")
	actionID := locationFields[1]
	th.AssertEquals(t, "625628cd-f877-44be-bde0-fec79f84e13d", actionID)

	actual, err := res.Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedCluster, *actual)
}

func TestCreateClusterEmptyTime(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCreateClusterEmptyTimeSuccessfully(t)

	minSize := 1
	opts := clusters.CreateOpts{
		Name:            "cluster1",
		DesiredCapacity: 3,
		ProfileID:       "d8a48377-f6a3-4af4-bbbb-6e8bcaa0cbc0",
		MinSize:         &minSize,
		MaxSize:         20,
		Timeout:         3600,
		Metadata:        map[string]interface{}{},
		Config:          map[string]interface{}{},
	}

	actual, err := clusters.Create(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedCluster_EmptyTime, *actual)
}

func TestCreateClusterMetadata(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCreateClusterMetadataSuccessfully(t)

	minSize := 1
	opts := clusters.CreateOpts{
		Name:            "cluster1",
		DesiredCapacity: 3,
		ProfileID:       "d8a48377-f6a3-4af4-bbbb-6e8bcaa0cbc0",
		MinSize:         &minSize,
		MaxSize:         20,
		Timeout:         3600,
		Metadata: map[string]interface{}{
			"foo": "bar",
			"test": map[string]interface{}{
				"nil_interface": interface{}(nil),
				"float_value":   float64(123.3),
				"string_value":  "test_string",
				"bool_value":    false,
			},
		},
		Config: map[string]interface{}{},
	}

	actual, err := clusters.Create(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedCluster_Metadata, *actual)
}

func TestGetCluster(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGetClusterSuccessfully(t)

	actual, err := clusters.Get(fake.ServiceClient(), "7d85f602-a948-4a30-afd4-e84f47471c15").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedCluster, *actual)
}

func TestGetClusterEmptyTime(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGetClusterEmptyTimeSuccessfully(t)

	actual, err := clusters.Get(fake.ServiceClient(), "7d85f602-a948-4a30-afd4-e84f47471c15").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedCluster_EmptyTime, *actual)
}

func TestListClusters(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleListClusterSuccessfully(t)

	count := 0

	clusters.List(fake.ServiceClient(), clusters.ListOpts{GlobalProject: new(bool)}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := clusters.ExtractClusters(page)
		th.AssertNoErr(t, err)
		th.AssertDeepEquals(t, ExpectedClusters, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestUpdateCluster(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleUpdateClusterSuccessfully(t)

	updateOpts := clusters.UpdateOpts{
		Name:      "cluster1",
		ProfileID: "edc63d0a-2ca4-48fa-9854-27926da76a4a",
	}

	actual, err := clusters.Update(fake.ServiceClient(), ExpectedCluster.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedCluster, *actual)
}

func TestUpdateClusterEmptyTime(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleUpdateClusterEmptyTimeSuccessfully(t)

	updateOpts := clusters.UpdateOpts{
		Name:      "cluster1",
		ProfileID: "edc63d0a-2ca4-48fa-9854-27926da76a4a",
	}

	actual, err := clusters.Update(fake.ServiceClient(), ExpectedCluster_EmptyTime.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedCluster_EmptyTime, *actual)
}

func TestDeleteCluster(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleDeleteClusterSuccessfully(t)

	err := clusters.Delete(fake.ServiceClient(), "6dc6d336e3fc4c0a951b5698cd1236ee").ExtractErr()
	th.AssertNoErr(t, err)
}

func TestResizeCluster(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleResizeSuccessfully(t)

	maxSize := 5
	minSize := 1
	number := -2
	strict := true
	opts := clusters.ResizeOpts{
		AdjustmentType: "CHANGE_IN_CAPACITY",
		MaxSize:        &maxSize,
		MinSize:        &minSize,
		Number:         number,
		Strict:         &strict,
	}

	actionID, err := clusters.Resize(fake.ServiceClient(), "7d85f602-a948-4a30-afd4-e84f47471c15", opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, ExpectedActionID, actionID)
}

// Test case for Number field having a float value
func TestResizeClusterNumberFloat(t *testing.T) {
	maxSize := 5
	minSize := 1
	number := 100.0
	strict := true
	opts := clusters.ResizeOpts{
		AdjustmentType: "CHANGE_IN_PERCENTAGE",
		MaxSize:        &maxSize,
		MinSize:        &minSize,
		Number:         number,
		Strict:         &strict,
	}

	_, err := opts.ToClusterResizeMap()
	th.AssertNoErr(t, err)
}

// Test case for missing Number field.
func TestResizeClusterMissingNumber(t *testing.T) {
	maxSize := 5
	minSize := 1
	strict := true
	opts := clusters.ResizeOpts{
		MaxSize: &maxSize,
		MinSize: &minSize,
		Strict:  &strict,
	}

	_, err := opts.ToClusterResizeMap()
	th.AssertNoErr(t, err)
}

// Test case for missing Number field which is required when AdjustmentType is specified
func TestResizeClusterInvalidParamsMissingNumber(t *testing.T) {
	maxSize := 5
	minSize := 1
	strict := true
	opts := clusters.ResizeOpts{
		AdjustmentType: "CHANGE_IN_CAPACITY",
		MaxSize:        &maxSize,
		MinSize:        &minSize,
		Strict:         &strict,
	}

	_, err := opts.ToClusterResizeMap()
	isValid := err == nil
	th.AssertEquals(t, false, isValid)
}

// Test case for float Number field which is only valid for CHANGE_IN_PERCENTAGE.
func TestResizeClusterInvalidParamsNumberFloat(t *testing.T) {
	maxSize := 5
	minSize := 1
	number := 100.0
	strict := true
	opts := clusters.ResizeOpts{
		AdjustmentType: "CHANGE_IN_CAPACITY",
		MaxSize:        &maxSize,
		MinSize:        &minSize,
		Number:         number,
		Strict:         &strict,
	}

	_, err := opts.ToClusterResizeMap()
	isValid := err == nil
	th.AssertEquals(t, false, isValid)
}

func TestClusterScaleIn(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleScaleInSuccessfully(t)

	count := 5
	scaleOpts := clusters.ScaleInOpts{
		Count: &count,
	}
	actionID, err := clusters.ScaleIn(fake.ServiceClient(), "edce3528-864f-41fb-8759-f4707925cc09", scaleOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, ExpectedActionID, actionID)
}

func TestListClusterPolicies(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleListPoliciesSuccessfully(t)

	pageCount := 0
	err := clusters.ListPolicies(fake.ServiceClient(), ExpectedClusterPolicy.ClusterID, clusters.ListPoliciesOpts{Name: "Test"}).EachPage(func(page pagination.Page) (bool, error) {
		pageCount++
		actual, err := clusters.ExtractClusterPolicies(page)
		th.AssertNoErr(t, err)
		th.AssertDeepEquals(t, ExpectedListPolicies, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, pageCount, 1)
}

func TestGetClusterPolicies(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGetPolicySuccessfully(t)

	actual, err := clusters.GetPolicy(fake.ServiceClient(), "7d85f602-a948-4a30-afd4-e84f47471c15", "714fe676-a08f-4196-b7af-61d52eeded15").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedClusterPolicy, *actual)
}

func TestClusterRecover(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleRecoverSuccessfully(t)

	recoverOpts := clusters.RecoverOpts{
		Operation:     clusters.RebuildRecovery,
		Check:         new(bool),
		CheckCapacity: new(bool),
	}
	actionID, err := clusters.Recover(fake.ServiceClient(), "edce3528-864f-41fb-8759-f4707925cc09", recoverOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, ExpectedActionID, actionID)
}

func TestAttachPolicy(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleAttachPolicySuccessfully(t)

	enabled := true
	opts := clusters.AttachPolicyOpts{
		PolicyID: "policy1",
		Enabled:  &enabled,
	}
	actionID, err := clusters.AttachPolicy(fake.ServiceClient(), "7d85f602-a948-4a30-afd4-e84f47471c15", opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, ExpectedActionID, actionID)
}

func TestDetachPolicy(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleDetachPolicySuccessfully(t)

	opts := clusters.DetachPolicyOpts{
		PolicyID: "policy1",
	}
	actionID, err := clusters.DetachPolicy(fake.ServiceClient(), "7d85f602-a948-4a30-afd4-e84f47471c15", opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, ExpectedActionID, actionID)
}

func TestUpdatePolicy(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleUpdatePolicySuccessfully(t)

	enabled := true
	opts := clusters.UpdatePolicyOpts{
		PolicyID: "policy1",
		Enabled:  &enabled,
	}
	actionID, err := clusters.UpdatePolicy(fake.ServiceClient(), "7d85f602-a948-4a30-afd4-e84f47471c15", opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, ExpectedActionID, actionID)
}

func TestClusterScaleOut(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleScaleOutSuccessfully(t)

	scaleOutOpts := clusters.ScaleOutOpts{
		Count: 5,
	}
	actionID, err := clusters.ScaleOut(fake.ServiceClient(), "edce3528-864f-41fb-8759-f4707925cc09", scaleOutOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, ExpectedActionID, actionID)
}

func TestClusterCheck(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCheckSuccessfully(t)

	actionID, err := clusters.Check(fake.ServiceClient(), "edce3528-864f-41fb-8759-f4707925cc09").Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, ExpectedActionID, actionID)
}

func TestLifecycle(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleLifecycleSuccessfully(t)

	opts := clusters.CompleteLifecycleOpts{
		LifecycleActionTokenID: "976528c6-dcf6-4d8d-9f4c-588f4e675f29",
	}

	res := clusters.CompleteLifecycle(fake.ServiceClient(), "edce3528-864f-41fb-8759-f4707925cc09", opts)
	location := res.Header.Get("Location")
	th.AssertEquals(t, "http://senlin.cloud.blizzard.net:8778/v1/actions/2a0ff107-e789-4660-a122-3816c43af703", location)

	actionID, err := res.Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, "2a0ff107-e789-4660-a122-3816c43af703", actionID)
}

func TestAddNodes(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleAddNodesSuccessfully(t)

	opts := clusters.AddNodesOpts{
		Nodes: []string{"node1", "node2", "node3"},
	}
	result, err := clusters.AddNodes(fake.ServiceClient(), "7d85f602-a948-4a30-afd4-e84f47471c15", opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, result, "2a0ff107-e789-4660-a122-3816c43af703")
}

func TestRemoveNodes(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleRemoveNodesSuccessfully(t)
	opts := clusters.RemoveNodesOpts{
		Nodes: []string{"node1", "node2", "node3"},
	}
	err := clusters.RemoveNodes(fake.ServiceClient(), "7d85f602-a948-4a30-afd4-e84f47471c15", opts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestReplaceNodes(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleReplaceNodeSuccessfully(t)
	opts := clusters.ReplaceNodesOpts{
		Nodes: map[string]string{"node-1234": "node-5678"},
	}
	actionID, err := clusters.ReplaceNodes(fake.ServiceClient(), "7d85f602-a948-4a30-afd4-e84f47471c15", opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, actionID, "2a0ff107-e789-4660-a122-3816c43af703")
}

func TestClusterCollect(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleClusterCollectSuccessfully(t)
	opts := clusters.CollectOpts{
		Path: "foo.bar",
	}
	attributes, err := clusters.Collect(fake.ServiceClient(), "7d85f602-a948-4a30-afd4-e84f47471c15", opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedCollectAttributes, attributes)
}

func TestOperation(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleOpsSuccessfully(t)

	clusterOpts := clusters.OperationOpts{
		Operation: clusters.PauseOperation,
		Filters:   clusters.OperationFilters{"role": "slave"},
		Params:    clusters.OperationParams{"type": "soft"},
	}
	actual, err := clusters.Ops(fake.ServiceClient(), "7d85f602-a948-4a30-afd4-e84f47471c15", clusterOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, OperationExpectedActionID, actual)
}
