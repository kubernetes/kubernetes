/*
Package claims provides information and interaction with the Zaqar API
claims resource for the OpenStack Messaging service.

Example to Create a Claim on a specified Zaqar queue

	createOpts := claims.CreateOpts{
		TTL:		60,
		Grace:		120,
		Limit: 		20,
	}

	queueName := "my_queue"

	messages, err := claims.Create(messagingClient, queueName, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to get a claim for a specified Zaqar queue

	queueName := "my_queue"
	claimID := "123456789012345678"

	claim, err := claims.Get(messagingClient, queueName, claimID).Extract()
	if err != nil {
		panic(err)
	}

Example to update a claim for a specified Zaqar queue

	updateOpts := claims.UpdateOpts{
		TTL: 600
		Grace: 1200
	}

	queueName := "my_queue"

	err := claims.Update(messagingClient, queueName, claimID, updateOpts).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to delete a claim for a specified Zaqar queue

	queueName := "my_queue"

	err := claims.Delete(messagingClient, queueName, claimID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package claims
