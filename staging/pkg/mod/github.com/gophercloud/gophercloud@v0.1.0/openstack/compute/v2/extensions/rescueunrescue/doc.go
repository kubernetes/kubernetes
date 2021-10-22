/*
Package rescueunrescue provides the ability to place a server into rescue mode
and to return it back.

Example to Rescue a server

  rescueOpts := rescueunrescue.RescueOpts{
    AdminPass:      "aUPtawPzE9NU",
    RescueImageRef: "115e5c5b-72f0-4a0a-9067-60706545248c",
  }
  serverID := "3f54d05f-3430-4d80-aa07-63e6af9e2488"

  adminPass, err := rescueunrescue.Rescue(computeClient, serverID, rescueOpts).Extract()
  if err != nil {
    panic(err)
  }

  fmt.Printf("adminPass of the rescued server %s: %s\n", serverID, adminPass)

Example to Unrescue a server

  serverID := "3f54d05f-3430-4d80-aa07-63e6af9e2488"

  if err := rescueunrescue.Unrescue(computeClient, serverID).ExtractErr(); err != nil {
    panic(err)
  }
*/
package rescueunrescue
