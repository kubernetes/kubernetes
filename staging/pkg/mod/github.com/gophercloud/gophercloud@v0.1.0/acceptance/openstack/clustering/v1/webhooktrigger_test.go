// +build acceptance clustering webhooks

package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack/clustering/v1/nodes"
	"github.com/gophercloud/gophercloud/openstack/clustering/v1/webhooks"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestClusteringWebhookTrigger(t *testing.T) {

	client, err := clients.NewClusteringV1Client()
	if err != nil {
		t.Fatalf("Unable to create clustering client: %v", err)
	}

	opts := webhooks.TriggerOpts{
		V: "1",
	}

	// create profile, cluster and receiver first
	profile, err := CreateProfile(t, client)
	th.AssertNoErr(t, err)
	defer DeleteProfile(t, client, profile.ID)

	cluster, err := CreateCluster(t, client, profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteCluster(t, client, cluster.ID)

	receiver, err := CreateWebhookReceiver(t, client, cluster.ID)
	th.AssertNoErr(t, err)
	defer DeleteReceiver(t, client, receiver.ID)

	// trigger webhook
	actionID, err := webhooks.Trigger(client, receiver.ID, opts).Extract()
	if err != nil {
		t.Fatalf("Unable to extract webhooks trigger: %v", err)
	} else {
		t.Logf("Webhook trigger action id %s", actionID)
	}

	err = WaitForAction(client, actionID)
	if err != nil {
		t.Fatalf("Error scaling out cluster %s as a result from webhook trigger: %s:", cluster.ID, err)
	}

	// check that new node was created
	nodelistopts := nodes.ListOpts{
		ClusterID: cluster.ID,
	}

	allPages, err := nodes.List(client, nodelistopts).AllPages()
	th.AssertNoErr(t, err)

	allNodes, err := nodes.ExtractNodes(allPages)
	th.AssertNoErr(t, err)

	// there should be 2 nodes in the cluster after triggering webhook
	th.AssertEquals(t, len(allNodes), 2)
}
