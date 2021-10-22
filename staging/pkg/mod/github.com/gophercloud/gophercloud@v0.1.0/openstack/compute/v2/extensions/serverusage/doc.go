/*
Package serverusage provides the ability the ability to extend a server result
with the extended usage information.

Example to Get an extended information:

  type serverUsageExt struct {
    servers.Server
    serverusage.UsageExt
  }
  var serverWithUsageExt serverUsageExt

  err := servers.Get(computeClient, "d650a0ce-17c3-497d-961a-43c4af80998a").ExtractInto(&serverWithUsageExt)
  if err != nil {
    panic(err)
  }

  fmt.Printf("%+v\n", serverWithUsageExt)
*/
package serverusage
