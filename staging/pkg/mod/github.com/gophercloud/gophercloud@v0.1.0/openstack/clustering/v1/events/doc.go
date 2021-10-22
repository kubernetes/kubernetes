/*
Package events provides listing and retrieving of senlin events for the
OpenStack Clustering Service.

Example to List Events

	opts := events.ListOpts{
		Limit: 5,
	}

	err = events.List(serviceClient, opts).EachPage(func(page pagination.Page) (bool, error) {
		eventInfos, err := events.ExtractEvents(page)
		if err != nil {
			return false, err
		}

		for _, eventInfo := range eventInfos {
			fmt.Println("%+v\n", eventInfo)
		}
		return true, nil
	})

Example to Get an Event

	eventID := "edce3528-864f-41fb-8759-f4707925cc09"
	event, err := events.Get(serviceClient, eventID).Extract()
	if err != nil {
		panic(err)
	}

	fmt.Printf("Event %+v: ", event)
*/
package events
