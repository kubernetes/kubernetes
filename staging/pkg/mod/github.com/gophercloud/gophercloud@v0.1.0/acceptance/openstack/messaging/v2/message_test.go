// +build acceptance messaging messages

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/messaging/v2/messages"
	"github.com/gophercloud/gophercloud/pagination"
)

func TestListMessages(t *testing.T) {
	clientID := "3381af92-2b9e-11e3-b191-718613007343"

	client, err := clients.NewMessagingV2Client(clientID)
	if err != nil {
		t.Fatalf("Unable to create a messaging service client: %v", err)
	}

	createdQueueName, err := CreateQueue(t, client)
	defer DeleteQueue(t, client, createdQueueName)

	for i := 0; i < 3; i++ {
		CreateMessage(t, client, createdQueueName)
	}

	// Use a different client/clientID in order to see messages on the Queue
	clientID = "3381af92-2b9e-11e3-b191-71861300734d"
	client, err = clients.NewMessagingV2Client(clientID)

	listOpts := messages.ListOpts{}

	pager := messages.List(client, createdQueueName, listOpts)
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		allMessages, err := messages.ExtractMessages(page)
		if err != nil {
			t.Fatalf("Unable to extract messages: %v", err)
		}

		for _, message := range allMessages {
			tools.PrintResource(t, message)
		}

		return true, nil
	})
}

func TestCreateMessages(t *testing.T) {
	clientID := "3381af92-2b9e-11e3-b191-71861300734c"

	client, err := clients.NewMessagingV2Client(clientID)
	if err != nil {
		t.Fatalf("Unable to create a messaging service client: %v", err)
	}

	createdQueueName, err := CreateQueue(t, client)
	defer DeleteQueue(t, client, createdQueueName)

	CreateMessage(t, client, createdQueueName)
}

func TestGetMessages(t *testing.T) {
	clientID := "3381af92-2b9e-11e3-b191-718613007343"

	client, err := clients.NewMessagingV2Client(clientID)
	if err != nil {
		t.Fatalf("Unable to create a messaging service client: %v", err)
	}

	createdQueueName, err := CreateQueue(t, client)
	defer DeleteQueue(t, client, createdQueueName)

	CreateMessage(t, client, createdQueueName)
	CreateMessage(t, client, createdQueueName)

	// Use a different client/clientID in order to see messages on the Queue
	clientID = "3381af92-2b9e-11e3-b191-71861300734d"
	client, err = clients.NewMessagingV2Client(clientID)

	listOpts := messages.ListOpts{}

	var messageIDs []string

	pager := messages.List(client, createdQueueName, listOpts)
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		allMessages, err := messages.ExtractMessages(page)
		if err != nil {
			t.Fatalf("Unable to extract messages: %v", err)
		}

		for _, message := range allMessages {
			messageIDs = append(messageIDs, message.ID)
		}

		return true, nil
	})

	getMessageOpts := messages.GetMessagesOpts{
		IDs: messageIDs,
	}
	t.Logf("Attempting to get messages from queue %s with ids: %v", createdQueueName, messageIDs)
	messagesList, err := messages.GetMessages(client, createdQueueName, getMessageOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to get messages from queue: %s", createdQueueName)
	}

	tools.PrintResource(t, messagesList)
}

func TestGetMessage(t *testing.T) {
	clientID := "3381af92-2b9e-11e3-b191-718613007343"

	client, err := clients.NewMessagingV2Client(clientID)
	if err != nil {
		t.Fatalf("Unable to create a messaging service client: %v", err)
	}

	createdQueueName, err := CreateQueue(t, client)
	defer DeleteQueue(t, client, createdQueueName)

	CreateMessage(t, client, createdQueueName)

	// Use a different client/clientID in order to see messages on the Queue
	clientID = "3381af92-2b9e-11e3-b191-71861300734d"
	client, err = clients.NewMessagingV2Client(clientID)

	listOpts := messages.ListOpts{}

	var messageIDs []string

	pager := messages.List(client, createdQueueName, listOpts)
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		allMessages, err := messages.ExtractMessages(page)
		if err != nil {
			t.Fatalf("Unable to extract messages: %v", err)
		}

		for _, message := range allMessages {
			messageIDs = append(messageIDs, message.ID)
		}

		return true, nil
	})

	for _, messageID := range messageIDs {
		t.Logf("Attempting to get message from queue %s: %s", createdQueueName, messageID)
		message, getErr := messages.Get(client, createdQueueName, messageID).Extract()
		if getErr != nil {
			t.Fatalf("Unable to get message from queue %s: %s", createdQueueName, messageID)
		}
		tools.PrintResource(t, message)
	}
}

func TestDeleteMessagesIDs(t *testing.T) {
	clientID := "3381af92-2b9e-11e3-b191-718613007343"

	client, err := clients.NewMessagingV2Client(clientID)
	if err != nil {
		t.Fatalf("Unable to create a messaging service client: %v", err)
	}

	createdQueueName, err := CreateQueue(t, client)
	defer DeleteQueue(t, client, createdQueueName)

	CreateMessage(t, client, createdQueueName)
	CreateMessage(t, client, createdQueueName)

	// Use a different client/clientID in order to see messages on the Queue
	clientID = "3381af92-2b9e-11e3-b191-71861300734d"

	client, err = clients.NewMessagingV2Client(clientID)

	listOpts := messages.ListOpts{}

	var messageIDs []string

	pager := messages.List(client, createdQueueName, listOpts)
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		allMessages, err := messages.ExtractMessages(page)
		if err != nil {
			t.Fatalf("Unable to extract messages: %v", err)
		}

		for _, message := range allMessages {
			messageIDs = append(messageIDs, message.ID)
			tools.PrintResource(t, message)
		}

		return true, nil
	})

	deleteOpts := messages.DeleteMessagesOpts{
		IDs: messageIDs,
	}

	t.Logf("Attempting to delete messages: %v", messageIDs)
	deleteErr := messages.DeleteMessages(client, createdQueueName, deleteOpts).ExtractErr()
	if deleteErr != nil {
		t.Fatalf("Unable to delete messages: %v", deleteErr)
	}

	t.Logf("Attempting to list messages.")
	messageList, err := ListMessages(t, client, createdQueueName)

	if len(messageList) > 0 {
		t.Fatalf("Did not delete all specified messages in the queue.")
	}
}

func TestDeleteMessagesPop(t *testing.T) {
	clientID := "3381af92-2b9e-11e3-b191-718613007343"

	client, err := clients.NewMessagingV2Client(clientID)
	if err != nil {
		t.Fatalf("Unable to create a messaging service client: %v", err)
	}

	createdQueueName, err := CreateQueue(t, client)
	defer DeleteQueue(t, client, createdQueueName)

	for i := 0; i < 5; i++ {
		CreateMessage(t, client, createdQueueName)
	}

	// Use a different client/clientID in order to see messages on the Queue
	clientID = "3381af92-2b9e-11e3-b191-71861300734d"

	client, err = clients.NewMessagingV2Client(clientID)

	messageList, err := ListMessages(t, client, createdQueueName)

	messagesNumber := len(messageList)
	popNumber := 3

	PopOpts := messages.PopMessagesOpts{
		Pop: popNumber,
	}

	t.Logf("Attempting to Pop last %v messages.", popNumber)
	popMessages, deleteErr := messages.PopMessages(client, createdQueueName, PopOpts).Extract()
	if deleteErr != nil {
		t.Fatalf("Unable to Pop messages: %v", deleteErr)
	}

	tools.PrintResource(t, popMessages)

	messageList, err = ListMessages(t, client, createdQueueName)
	if len(messageList) != messagesNumber-popNumber {
		t.Fatalf("Unable to Pop specified number of messages.")
	}
}

func TestDeleteMessage(t *testing.T) {
	clientID := "3381af92-2b9e-11e3-b191-718613007343"

	client, err := clients.NewMessagingV2Client(clientID)
	if err != nil {
		t.Fatalf("Unable to create a messaging service client: %v", err)
	}

	createdQueueName, err := CreateQueue(t, client)
	defer DeleteQueue(t, client, createdQueueName)

	CreateMessage(t, client, createdQueueName)

	// Use a different client/clientID in order to see messages on the Queue
	clientID = "3381af92-2b9e-11e3-b191-71861300734d"
	client, err = clients.NewMessagingV2Client(clientID)

	listOpts := messages.ListOpts{}

	var messageIDs []string

	pager := messages.List(client, createdQueueName, listOpts)
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		allMessages, err := messages.ExtractMessages(page)
		if err != nil {
			t.Fatalf("Unable to extract messages: %v", err)
		}

		for _, message := range allMessages {
			messageIDs = append(messageIDs, message.ID)
		}

		return true, nil
	})

	for _, messageID := range messageIDs {
		t.Logf("Attempting to delete message from queue %s: %s", createdQueueName, messageID)
		deleteOpts := messages.DeleteOpts{}
		deleteErr := messages.Delete(client, createdQueueName, messageID, deleteOpts).ExtractErr()
		if deleteErr != nil {
			t.Fatalf("Unable to delete message from queue %s: %s", createdQueueName, messageID)
		} else {
			t.Logf("Successfully deleted message: %s", messageID)
		}
	}
}
