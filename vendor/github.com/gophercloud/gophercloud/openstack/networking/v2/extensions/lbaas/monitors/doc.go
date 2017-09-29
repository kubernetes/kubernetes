/*
Package monitors provides information and interaction with the Monitors
of the Load Balancer as a Service extension for the OpenStack Networking
Service.

Example to List Monitors

	listOpts: monitors.ListOpts{
		Type: "HTTP",
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
		Delay:         20,
		Timeout:       20,
		MaxRetries:    5,
		URLPath:       "/check",
		ExpectedCodes: "200-299",
	}

	monitor, err := monitors.Create(networkClient, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Update a Monitor

	monitorID := "681aed03-aadb-43ae-aead-b9016375650a"

	updateOpts := monitors.UpdateOpts{
		Timeout: 30,
	}

	monitor, err := monitors.Update(networkClient, monitorID, updateOpts).Extract()
	if err != nil {
		panic(err)
	}

Example to Delete a Member

	monitorID := "681aed03-aadb-43ae-aead-b9016375650a"
	err := monitors.Delete(networkClient, monitorID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package monitors
