// +build acceptance clustering policies

package v1

import (
	"sort"
	"strings"
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/clustering/v1/actions"
	"github.com/gophercloud/gophercloud/openstack/clustering/v1/clusters"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestClustersCRUD(t *testing.T) {
	client, err := clients.NewClusteringV1Client()
	th.AssertNoErr(t, err)

	profile, err := CreateProfile(t, client)
	th.AssertNoErr(t, err)
	defer DeleteProfile(t, client, profile.ID)

	cluster, err := CreateCluster(t, client, profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteCluster(t, client, cluster.ID)

	// Test clusters list
	allPages, err := clusters.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allClusters, err := clusters.ExtractClusters(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, v := range allClusters {
		if v.ID == cluster.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)

	// Test cluster update
	updateOpts := clusters.UpdateOpts{
		Name: cluster.Name + "-UPDATED",
	}

	res := clusters.Update(client, cluster.ID, updateOpts)
	th.AssertNoErr(t, res.Err)

	actionID, err := GetActionID(res.Header)
	th.AssertNoErr(t, err)

	err = WaitForAction(client, actionID)
	th.AssertNoErr(t, err)

	newCluster, err := clusters.Get(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, newCluster.Name, cluster.Name+"-UPDATED")

	tools.PrintResource(t, newCluster)

	// Test cluster health
	actionID, err = clusters.Check(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)

	err = WaitForAction(client, actionID)
	th.AssertNoErr(t, err)
}

func TestClustersResize(t *testing.T) {
	client, err := clients.NewClusteringV1Client()
	th.AssertNoErr(t, err)

	profile, err := CreateProfile(t, client)
	th.AssertNoErr(t, err)
	defer DeleteProfile(t, client, profile.ID)

	cluster, err := CreateCluster(t, client, profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteCluster(t, client, cluster.ID)

	iTrue := true
	resizeOpts := clusters.ResizeOpts{
		AdjustmentType: clusters.ChangeInCapacityAdjustment,
		Number:         1,
		Strict:         &iTrue,
	}

	actionID, err := clusters.Resize(client, cluster.ID, resizeOpts).Extract()
	th.AssertNoErr(t, err)

	err = WaitForAction(client, actionID)
	th.AssertNoErr(t, err)

	newCluster, err := clusters.Get(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, newCluster.DesiredCapacity, 2)

	tools.PrintResource(t, newCluster)
}

func TestClustersScale(t *testing.T) {
	client, err := clients.NewClusteringV1Client()
	th.AssertNoErr(t, err)

	profile, err := CreateProfile(t, client)
	th.AssertNoErr(t, err)
	defer DeleteProfile(t, client, profile.ID)

	cluster, err := CreateCluster(t, client, profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteCluster(t, client, cluster.ID)

	// increase cluster size to 2
	scaleOutOpts := clusters.ScaleOutOpts{
		Count: 1,
	}
	actionID, err := clusters.ScaleOut(client, cluster.ID, scaleOutOpts).Extract()
	th.AssertNoErr(t, err)

	err = WaitForAction(client, actionID)
	th.AssertNoErr(t, err)

	newCluster, err := clusters.Get(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, newCluster.DesiredCapacity, 2)

	// reduce cluster size to 0
	count := 2
	scaleInOpts := clusters.ScaleInOpts{
		Count: &count,
	}

	actionID, err = clusters.ScaleIn(client, cluster.ID, scaleInOpts).Extract()
	th.AssertNoErr(t, err)

	err = WaitForAction(client, actionID)
	th.AssertNoErr(t, err)

	newCluster, err = clusters.Get(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, newCluster.DesiredCapacity, 0)

	tools.PrintResource(t, newCluster)
}

func TestClustersPolicies(t *testing.T) {
	client, err := clients.NewClusteringV1Client()
	th.AssertNoErr(t, err)
	client.Microversion = "1.5"

	profile, err := CreateProfile(t, client)
	th.AssertNoErr(t, err)
	defer DeleteProfile(t, client, profile.ID)

	cluster, err := CreateCluster(t, client, profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteCluster(t, client, cluster.ID)

	policy, err := CreatePolicy(t, client)
	th.AssertNoErr(t, err)
	defer DeletePolicy(t, client, policy.ID)

	iTrue := true
	attachPolicyOpts := clusters.AttachPolicyOpts{
		PolicyID: policy.ID,
		Enabled:  &iTrue,
	}

	actionID, err := clusters.AttachPolicy(client, cluster.ID, attachPolicyOpts).Extract()
	th.AssertNoErr(t, err)

	err = WaitForAction(client, actionID)
	th.AssertNoErr(t, err)

	// List all policies in the cluster to see if the policy was
	// successfully attached.
	allPages, err := clusters.ListPolicies(client, cluster.ID, nil).AllPages()
	th.AssertNoErr(t, err)

	allPolicies, err := clusters.ExtractClusterPolicies(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, v := range allPolicies {
		tools.PrintResource(t, v)
		if v.PolicyID == policy.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)

	// Set the policy to disabled
	iFalse := false
	updatePolicyOpts := clusters.UpdatePolicyOpts{
		PolicyID: policy.ID,
		Enabled:  &iFalse,
	}

	actionID, err = clusters.UpdatePolicy(client, cluster.ID, updatePolicyOpts).Extract()
	th.AssertNoErr(t, err)

	err = WaitForAction(client, actionID)
	th.AssertNoErr(t, err)

	clusterPolicy, err := clusters.GetPolicy(client, cluster.ID, policy.ID).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, clusterPolicy.Enabled, false)

	// Detach the policy
	detachPolicyOpts := clusters.DetachPolicyOpts{
		PolicyID: policy.ID,
	}

	actionID, err = clusters.DetachPolicy(client, cluster.ID, detachPolicyOpts).Extract()
	th.AssertNoErr(t, err)

	err = WaitForAction(client, actionID)
	th.AssertNoErr(t, err)

	// List all policies in the cluster to see if the policy was
	// successfully detached.
	allPages, err = clusters.ListPolicies(client, cluster.ID, nil).AllPages()
	th.AssertNoErr(t, err)

	allPolicies, err = clusters.ExtractClusterPolicies(allPages)
	th.AssertNoErr(t, err)

	found = false
	for _, v := range allPolicies {
		tools.PrintResource(t, v)
		if v.PolicyID == policy.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, false)
}

func TestClustersRecovery(t *testing.T) {
	client, err := clients.NewClusteringV1Client()
	th.AssertNoErr(t, err)

	profile, err := CreateProfile(t, client)
	th.AssertNoErr(t, err)
	defer DeleteProfile(t, client, profile.ID)

	cluster, err := CreateCluster(t, client, profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteCluster(t, client, cluster.ID)

	recoverOpts := clusters.RecoverOpts{
		Operation: clusters.RebuildRecovery,
	}

	actionID, err := clusters.Recover(client, cluster.ID, recoverOpts).Extract()
	th.AssertNoErr(t, err)

	err = WaitForAction(client, actionID)
	th.AssertNoErr(t, err)

	newCluster, err := clusters.Get(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newCluster)
}

func TestClustersAddNode(t *testing.T) {
	client, err := clients.NewClusteringV1Client()
	th.AssertNoErr(t, err)

	profile, err := CreateProfile(t, client)
	th.AssertNoErr(t, err)
	defer DeleteProfile(t, client, profile.ID)

	cluster, err := CreateCluster(t, client, profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteCluster(t, client, cluster.ID)

	node1, err := CreateNode(t, client, "", profile.ID)
	th.AssertNoErr(t, err)
	// Even tho deleting the cluster will delete the nodes but only if added into cluster successfully.
	defer DeleteNode(t, client, node1.ID)

	node2, err := CreateNode(t, client, "", profile.ID)
	th.AssertNoErr(t, err)
	// Even tho deleting the cluster will delete the nodes but only if added into cluster successfully.
	defer DeleteNode(t, client, node2.ID)

	cluster, err = clusters.Get(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)

	nodeIDs := []string{node1.ID, node2.ID}
	nodeIDs = append(nodeIDs, cluster.Nodes...)

	nodeNames := []string{node1.Name, node2.Name}
	addNodesOpts := clusters.AddNodesOpts{
		Nodes: nodeNames,
	}
	actionID, err := clusters.AddNodes(client, cluster.ID, addNodesOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to add nodes to cluster: %v", err)
	}

	err = WaitForAction(client, actionID)
	th.AssertNoErr(t, err)

	cluster, err = clusters.Get(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)

	sort.Strings(nodeIDs)
	sort.Strings(cluster.Nodes)

	tools.PrintResource(t, nodeIDs)
	tools.PrintResource(t, cluster.Nodes)

	th.AssertDeepEquals(t, nodeIDs, cluster.Nodes)

	tools.PrintResource(t, cluster)
}

func TestClustersRemoveNodeFromCluster(t *testing.T) {
	client, err := clients.NewClusteringV1Client()
	th.AssertNoErr(t, err)

	profile, err := CreateProfile(t, client)
	th.AssertNoErr(t, err)
	defer DeleteProfile(t, client, profile.ID)

	cluster, err := CreateCluster(t, client, profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteCluster(t, client, cluster.ID)

	cluster, err = clusters.Get(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, cluster)

	opt := clusters.RemoveNodesOpts{Nodes: cluster.Nodes}
	res := clusters.RemoveNodes(client, cluster.ID, opt)
	err = res.ExtractErr()
	th.AssertNoErr(t, err)

	for _, n := range cluster.Nodes {
		defer DeleteNode(t, client, n)
	}

	actionID, err := GetActionID(res.Header)
	th.AssertNoErr(t, err)

	err = WaitForAction(client, actionID)
	th.AssertNoErr(t, err)

	cluster, err = clusters.Get(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, 0, len(cluster.Nodes))

	tools.PrintResource(t, cluster)
}

func TestClustersReplaceNode(t *testing.T) {
	client, err := clients.NewClusteringV1Client()
	th.AssertNoErr(t, err)
	client.Microversion = "1.3"

	profile, err := CreateProfile(t, client)
	th.AssertNoErr(t, err)
	defer DeleteProfile(t, client, profile.ID)

	cluster, err := CreateCluster(t, client, profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteCluster(t, client, cluster.ID)

	node1, err := CreateNode(t, client, "", profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteNode(t, client, node1.ID)

	cluster, err = clusters.Get(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, true, len(cluster.Nodes) > 0)
	for _, n := range cluster.Nodes {
		defer DeleteNode(t, client, n)
	}

	nodeIDToBeReplaced := cluster.Nodes[0]
	opts := clusters.ReplaceNodesOpts{Nodes: map[string]string{nodeIDToBeReplaced: node1.ID}}
	actionID, err := clusters.ReplaceNodes(client, cluster.ID, opts).Extract()
	th.AssertNoErr(t, err)
	err = WaitForAction(client, actionID)
	th.AssertNoErr(t, err)

	cluster, err = clusters.Get(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)

	clusterNodes := strings.Join(cluster.Nodes, ",")
	th.AssertEquals(t, true, strings.Contains(clusterNodes, node1.ID))
	th.AssertEquals(t, false, strings.Contains(clusterNodes, nodeIDToBeReplaced))
	tools.PrintResource(t, cluster)
}

func TestClustersCollectAttributes(t *testing.T) {
	client, err := clients.NewClusteringV1Client()
	th.AssertNoErr(t, err)
	client.Microversion = "1.2"

	profile, err := CreateProfile(t, client)
	th.AssertNoErr(t, err)
	defer DeleteProfile(t, client, profile.ID)

	cluster, err := CreateCluster(t, client, profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteCluster(t, client, cluster.ID)

	cluster, err = clusters.Get(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, true, len(cluster.Nodes) > 0)

	_, err = CreateNode(t, client, cluster.ID, profile.ID)
	th.AssertNoErr(t, err)

	cluster, err = clusters.Get(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, true, len(cluster.Nodes) > 0)

	for _, n := range cluster.Nodes {
		defer DeleteNode(t, client, n)
	}

	opts := clusters.CollectOpts{
		Path: "status",
	}
	attrs, err := clusters.Collect(client, cluster.ID, opts).Extract()
	th.AssertNoErr(t, err)
	for _, attr := range attrs {
		th.AssertEquals(t, attr.Value, "ACTIVE")
	}

	opts = clusters.CollectOpts{
		Path: "data.placement.zone",
	}
	attrs, err = clusters.Collect(client, cluster.ID, opts).Extract()
	th.AssertNoErr(t, err)
	for _, attr := range attrs {
		th.AssertEquals(t, attr.Value, "nova")
	}

}

// Performs an operation on a cluster
func TestClustersOps(t *testing.T) {
	choices, err := clients.AcceptanceTestChoicesFromEnv()
	th.AssertNoErr(t, err)

	client, err := clients.NewClusteringV1Client()
	th.AssertNoErr(t, err)
	client.Microversion = "1.4"

	profile, err := CreateProfile(t, client)
	th.AssertNoErr(t, err)
	defer DeleteProfile(t, client, profile.ID)

	cluster, err := CreateCluster(t, client, profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteCluster(t, client, cluster.ID)

	node, err := CreateNode(t, client, cluster.ID, profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteNode(t, client, node.ID)

	ops := []clusters.OperationOpts{
		// TODO: Commented out due to backend returns error, as of 2019-01-09
		//{Operation: clusters.RebuildOperation},					// Error in set_admin_password in nova log
		//{Operation: clusters.EvacuateOperation, Params: clusters.OperationParams{"host": cluster.ID, "force": "True"}},
		{Operation: clusters.RebootOperation, Params: clusters.OperationParams{"type": "SOFT"}},
		{Operation: clusters.ChangePasswordOperation, Params: clusters.OperationParams{"admin_pass": "test"}},
		{Operation: clusters.LockOperation},
		{Operation: clusters.UnlockOperation},
		{Operation: clusters.SuspendOperation},
		{Operation: clusters.ResumeOperation},
		{Operation: clusters.RescueOperation, Params: clusters.OperationParams{"image_ref": choices.ImageID}},
		{Operation: clusters.PauseOperation},
		{Operation: clusters.UnpauseOperation},
		{Operation: clusters.StopOperation},
		{Operation: clusters.StartOperation},
	}

	for _, op := range ops {
		opName := string(op.Operation)
		t.Logf("Attempting to perform '%s' on cluster: %s", opName, cluster.ID)
		actionID, res := clusters.Ops(client, cluster.ID, op).Extract()
		th.AssertNoErr(t, res)

		err = WaitForAction(client, actionID)
		th.AssertNoErr(t, err)

		action, err := actions.Get(client, actionID).Extract()
		th.AssertNoErr(t, err)
		th.AssertEquals(t, "SUCCEEDED", action.Status)

		t.Logf("Successfully performed '%s' on cluster: %s", opName, cluster.ID)
	}

	cluster, err = clusters.Get(client, cluster.ID).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, cluster)
}
