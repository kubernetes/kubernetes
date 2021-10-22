// +build acceptance messaging claims

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/messaging/v2/claims"
)

func TestCRUDClaim(t *testing.T) {
	clientID := "3381af92-2b9e-11e3-b191-71861300734c"

	client, err := clients.NewMessagingV2Client(clientID)
	if err != nil {
		t.Fatalf("Unable to create a messaging service client: %v", err)
	}

	createdQueueName, err := CreateQueue(t, client)
	defer DeleteQueue(t, client, createdQueueName)

	clientID = "3381af92-2b9e-11e3-b191-71861300734d"

	client, err = clients.NewMessagingV2Client(clientID)
	if err != nil {
		t.Fatalf("Unable to create a messaging service client: %v", err)
	}
	for i := 0; i < 3; i++ {
		CreateMessage(t, client, createdQueueName)
	}

	clientID = "3381af92-2b9e-11e3-b191-7186130073dd"
	claimedMessages, err := CreateClaim(t, client, createdQueueName)
	claimIDs, _ := ExtractIDs(claimedMessages)

	tools.PrintResource(t, claimedMessages)

	updateOpts := claims.UpdateOpts{
		TTL:   600,
		Grace: 500,
	}

	for _, claimID := range claimIDs {
		t.Logf("Attempting to update claim: %s", claimID)
		updateErr := claims.Update(client, createdQueueName, claimID, updateOpts).ExtractErr()

		if updateErr != nil {
			t.Fatalf("Unable to update claim %s: %v", claimID, err)
		} else {
			t.Logf("Successfully updated claim: %s", claimID)
		}

		updatedClaim, getErr := GetClaim(t, client, createdQueueName, claimID)
		if getErr != nil {
			t.Fatalf("Unable to retrieve claim %s: %v", claimID, getErr)
		}

		tools.PrintResource(t, updatedClaim)
		DeleteClaim(t, client, createdQueueName, claimID)
	}
}
