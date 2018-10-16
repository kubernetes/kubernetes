/*
Package monitors provides information and interaction with Monitors
of the LBaaS v2 extension for the OpenStack Networking service.

Example to List Monitors

	listOpts := monitors.ListOpts{
		PoolID: "c79a4468-d788-410c-bf79-9a8ef6354852",
	}

	allPages, err := monitors.List(networkClient, listOpts).AllPages()
	if err != nil {
		panic(err)
	}

	allMonitors, err := monitors.ExtractMonitors(allPages)
	if err != nil {
		panic(err)
	}

	for _, monitor := range allMonitors {
		fmt.Printf("%+v\n", monitor)
	}

Example to Create a Monitor

	createOpts := monitors.CreateOpts{
		Type:          "HTTP",
		Name:          "db",
		PoolID:        "84f1b61f-58c4-45bf-a8a9-2dafb9e5214d",
		Delay:         20,
		Timeout:       10,
		MaxRetries:    5,
		URLPath:       "/check",
		ExpectedCodes: "200-299",
	}

	monitor, err := monitors.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Monitor

	monitorID := "d67d56a6-4a86-4688-a282-f46444705c64"

	updateOpts := monitors.UpdateOpts{
		Name:          "NewHealthmonitorName",
		Delay:         3,
		Timeout:       20,
		MaxRetries:    10,
		URLPath:       "/another_check",
		ExpectedCodes: "301",
	}

	monitor, err := monitors.Update(networkClient, monitorID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Monitor

	monitorID := "d67d56a6-4a86-4688-a282-f46444705c64"
	err := monitors.Delete(networkClient, monitorID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package monitors
