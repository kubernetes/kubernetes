// +build acceptance clustering policies

package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/clustering/v1/receivers"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestReceiversCRUD(t *testing.T) {
	client, err := clients.NewClusteringV1Client()
	th.AssertNoErr(t, err)

	profile, err := CreateProfile(t, client)
	th.AssertNoErr(t, err)
	defer DeleteProfile(t, client, profile.ID)

	cluster, err := CreateCluster(t, client, profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteCluster(t, client, cluster.ID)

	receiver, err := CreateWebhookReceiver(t, client, cluster.ID)
	th.AssertNoErr(t, err)
	defer DeleteReceiver(t, client, receiver.ID)

	// Test listing receivers
	allPages, err := receivers.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allReceivers, err := receivers.ExtractReceivers(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, v := range allReceivers {
		if v.ID == receiver.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)

	// Test updating receivers
	newName := receiver.Name + "-UPDATED"
	updateOpts := receivers.UpdateOpts{
		Name: newName,
	}

	receiver, err = receivers.Update(client, receiver.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, receiver)
	tools.PrintResource(t, receiver.UpdatedAt)

	th.AssertEquals(t, receiver.Name, newName)
}

func TestReceiversNotify(t *testing.T) {
	t.Parallel()
	client, err := clients.NewClusteringV1Client()
	th.AssertNoErr(t, err)

	profile, err := CreateProfile(t, client)
	th.AssertNoErr(t, err)
	defer DeleteProfile(t, client, profile.ID)

	cluster, err := CreateCluster(t, client, profile.ID)
	th.AssertNoErr(t, err)
	defer DeleteCluster(t, client, cluster.ID)

	receiver, err := CreateMessageReceiver(t, client, cluster.ID)
	th.AssertNoErr(t, err)
	defer DeleteReceiver(t, client, receiver.ID)
	t.Logf("Created Mesage Receiver Name:[%s] Message Receiver ID:[%s]", receiver.Name, receiver.ID)

	requestID, err := receivers.Notify(client, receiver.ID).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Receiver Notify Service Request ID: %s", requestID)
}
