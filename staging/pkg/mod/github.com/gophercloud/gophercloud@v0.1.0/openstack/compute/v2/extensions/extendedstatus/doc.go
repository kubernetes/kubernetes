/*
Package extendedstatus provides the ability to extend a server result with
the extended status information. Example:

	type ServerWithExt struct {
		servers.Server
		extendedstatus.ServerExtendedStatusExt
	}

	var allServers []ServerWithExt

	allPages, err := servers.List(client, nil).AllPages()
	if err != nil {
		panic("Unable to retrieve servers: %s", err)
	}

	err = servers.ExtractServersInto(allPages, &allServers)
	if err != nil {
		panic("Unable to extract servers: %s", err)
	}

	for _, server := range allServers {
		fmt.Println(server.TaskState)
		fmt.Println(server.VmState)
		fmt.Println(server.PowerState)
	}
*/
package extendedstatus
