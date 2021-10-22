/*
Package diskconfig provides information and interaction with the Disk Config
extension that works with the OpenStack Compute service.

Example of Obtaining the Disk Config of a Server

	type ServerWithDiskConfig {
		servers.Server
		diskconfig.ServerDiskConfigExt
	}

	var allServers []ServerWithDiskConfig

	allPages, err := servers.List(client, nil).AllPages()
	if err != nil {
		panic("Unable to retrieve servers: %s", err)
	}

	err = servers.ExtractServersInto(allPages, &allServers)
	if err != nil {
		panic("Unable to extract servers: %s", err)
	}

	for _, server := range allServers {
		fmt.Println(server.DiskConfig)
	}

Example of Creating a Server with Disk Config

	serverCreateOpts := servers.CreateOpts{
		Name:      "server_name",
		ImageRef:  "image-uuid",
		FlavorRef: "flavor-uuid",
	}

	createOpts := diskconfig.CreateOptsExt{
		CreateOptsBuilder: serverCreateOpts,
		DiskConfig:        diskconfig.Manual,
	}

	server, err := servers.Create(computeClient, createOpts).Extract()
	if err != nil {
		panic("Unable to create server: %s", err)
	}
*/
package diskconfig
