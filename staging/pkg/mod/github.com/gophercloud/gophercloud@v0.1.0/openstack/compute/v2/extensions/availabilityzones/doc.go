/*
Package availabilityzones provides the ability to get lists and detailed
availability zone information and to extend a server result with
availability zone information.

Example of Extend server result with Availability Zone Information:

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

Example of Get Availability Zone Information

	allPages, err := availabilityzones.List(computeClient).AllPages()
	if err != nil {
		panic(err)
	}

	availabilityZoneInfo, err := availabilityzones.ExtractAvailabilityZones(allPages)
	if err != nil {
		panic(err)
	}

	for _, zoneInfo := range availabilityZoneInfo {
  		fmt.Printf("%+v\n", zoneInfo)
	}

Example of Get Detailed Availability Zone Information

	allPages, err := availabilityzones.ListDetail(computeClient).AllPages()
	if err != nil {
		panic(err)
	}

	availabilityZoneInfo, err := availabilityzones.ExtractAvailabilityZones(allPages)
	if err != nil {
		panic(err)
	}

	for _, zoneInfo := range availabilityZoneInfo {
  		fmt.Printf("%+v\n", zoneInfo)
	}
*/
package availabilityzones
