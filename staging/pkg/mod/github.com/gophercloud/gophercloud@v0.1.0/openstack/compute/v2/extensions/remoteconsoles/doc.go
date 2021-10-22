/*
Package remoteconsoles provides the ability to create server remote consoles
through the Compute API.
You need to specify at least "2.6" microversion for the ComputeClient to use
that API.

Example of Creating a new RemoteConsole

  computeClient, err := openstack.NewComputeV2(providerClient, endpointOptions)
  computeClient.Microversion = "2.6"

  createOpts := remoteconsoles.CreateOpts{
    Protocol: remoteconsoles.ConsoleProtocolVNC,
    Type:     remoteconsoles.ConsoleTypeNoVNC,
  }
  serverID := "b16ba811-199d-4ffd-8839-ba96c1185a67"

  remtoteConsole, err := remoteconsoles.Create(computeClient, serverID, createOpts).Extract()
  if err != nil {
    panic(err)
  }

  fmt.Printf("Console URL: %s\n", remtoteConsole.URL)
*/
package remoteconsoles
