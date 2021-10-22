/*
Package messages provides information and interaction with the messages through
the OpenStack Messaging(Zaqar) service.

Example to List Messages

	listOpts := messages.ListOpts{
		Limit: 10,
	}

	queueName := "my_queue"

	pager := messages.List(client, queueName, listOpts)

	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		allMessages, err := queues.ExtractQueues(page)
		if err != nil {
			panic(err)
		}

		for _, message := range allMessages {
			fmt.Printf("%+v\n", message)
		}

		return true, nil
	})

Example to Create Messages

	queueName = "my_queue"

	createOpts := messages.CreateOpts{
		Messages:     []messages.Messages{
			{
				TTL:   300,
				Delay: 20,
				Body: map[string]interface{}{
					"event": "BackupStarted",
					"backup_id": "c378813c-3f0b-11e2-ad92-7823d2b0f3ce",
				},
			},
			{
				Body: map[string]interface{}{
					"event": "BackupProgress",
					"current_bytes": "0",
					"total_bytes": "99614720",
				},
			},
		},
	}

	resources, err := messages.Create(client, queueName, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Get a set of Messages

	queueName := "my_queue"

	getMessageOpts := messages.GetMessagesOpts{
		IDs: "123456",
	}

	messagesList, err := messages.GetMessages(client, createdQueueName, getMessageOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to get a singular Message

	queueName := "my_queue"
	messageID := "123456"

	message, err := messages.Get(client, queueName, messageID).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a set of Messages

	queueName := "my_queue"

	deleteMessagesOpts := messages.DeleteMessagesOpts{
		IDs: []string{"9988776655"},
	}

	err := messages.DeleteMessages(client, queueName, deleteMessagesOpts).ExtractErr()
	if err != nil {
		panic(err)
	}

Example to Pop a set of Messages

	queueName := "my_queue"

	popMessagesOpts := messages.PopMessagesOpts{
		Pop: 5,
	}

	resources, err := messages.PopMessages(client, queueName, popMessagesOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a singular Message

	clientID := "3381af92-2b9e-11e3-b191-71861300734d"
	queueName := "my_queue"
	messageID := "123456"

	deleteOpts := messages.DeleteOpts{
		ClaimID: "12345",
	}

	err := messages.Delete(client), queueName, messageID, deleteOpts).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package messages
