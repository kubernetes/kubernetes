/*
Package receivers provides information and interaction with the receivers through
the OpenStack Clustering service.

Example to Create a Receiver

	createOpts := receivers.CreateOpts{
		Action:     "CLUSTER_DEL_NODES",
		ClusterID:  "b7b870ee-d3c5-4a93-b9d7-846c53b2c2dc",
		Name:       "test_receiver",
		Type:       receivers.WebhookReceiver,
	}

	receiver, err := receivers.Create(serviceClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%v\n", receiver)

Example to Get a Receiver

	receiver, err := receivers.Get(serviceClient, "receiver-name").Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%v\n", receiver)

Example to Delete receiver

	receiverID := "6dc6d336e3fc4c0a951b5698cd1236ee"
	err := receivers.Delete(serviceClient, receiverID).ExtractErr()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%v\n", receiver)

Example to Update Receiver

	updateOpts := receivers.UpdateOpts{
		Name: "new-name",
	}

	receiverID := "6dc6d336e3fc4c0a951b5698cd1236ee"
	receiver, err := receivers.Update(serviceClient, receiverID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%v\n", receiver)

Example to List Receivers

	listOpts := receivers.ListOpts{
		Limit: 2,
	}

	receivers.List(serviceClient, listOpts).EachPage(func(page pagination.Page) (bool, error) {
		allReceivers, err := receivers.ExtractReceivers(page)
		if err != nil {
			panic(err)
		}

		for _, receiver := range allReceivers {
			fmt.Printf("%+v\n", receiver)
		}
		return true, nil
	})

Example to Notify a Receiver

	receiverID := "6dc6d336e3fc4c0a951b5698cd1236ee"
	requestID, err := receivers.Notify(serviceClient, receiverID).Extract()
	if err != nil {
		panic(err)
	}

*/
package receivers
