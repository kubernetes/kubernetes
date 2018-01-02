/*
Package availabilityzones provides the ability to extend a server result with
availability zone information. Example:

	type ServerWithAZ struct {
		servers.Server
		availabilityzones.ServerAvailabilityZoneExt
	}

	var allServers []ServerWithAZ

	allPages, err := servers.List(client, nil).AllPages()
	if err != nil {
		panic("Unable to retrieve servers: %s", err)
	}

	err = servers.ExtractServersInto(allPages, &allServers)
	if err != nil {
		panic("Unable to extract servers: %s", err)
	}

	for _, server := range allServers {
		fmt.Println(server.AvailabilityZone)
	}
*/
package availabilityzones
