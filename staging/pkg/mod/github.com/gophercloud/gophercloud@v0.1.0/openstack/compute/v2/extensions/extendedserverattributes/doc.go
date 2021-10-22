/*
Package extendedserverattributes provides the ability to extend a
server result with the extended usage information.

Example to Get basic extended information:

  type serverAttributesExt struct {
    servers.Server
    extendedserverattributes.ServerAttributesExt
  }
  var serverWithAttributesExt serverAttributesExt

  err := servers.Get(computeClient, "d650a0ce-17c3-497d-961a-43c4af80998a").ExtractInto(&serverWithAttributesExt)
  if err != nil {
    panic(err)
  }

  fmt.Printf("%+v\n", serverWithAttributesExt)

Example to get additional fields with microversion 2.3 or later

  computeClient.Microversion = "2.3"
  result := servers.Get(computeClient, "d650a0ce-17c3-497d-961a-43c4af80998a")

  reservationID, err := extendedserverattributes.ExtractReservationID(result.Result)
  if err != nil {
    panic(err)
  }
  fmt.Printf("%s\n", reservationID)

  launchIndex, err := extendedserverattributes.ExtractLaunchIndex(result.Result)
  if err != nil {
    panic(err)
  }
  fmt.Printf("%d\n", launchIndex)

  ramdiskID, err := extendedserverattributes.ExtractRamdiskID(result.Result)
  if err != nil {
    panic(err)
  }
  fmt.Printf("%s\n", ramdiskID)

  kernelID, err := extendedserverattributes.ExtractKernelID(result.Result)
  if err != nil {
    panic(err)
  }
  fmt.Printf("%s\n", kernelID)

  hostname, err := extendedserverattributes.ExtractHostname(result.Result)
  if err != nil {
    panic(err)
  }
  fmt.Printf("%s\n", hostname)

  rootDeviceName, err := extendedserverattributes.ExtractRootDeviceName(result.Result)
  if err != nil {
    panic(err)
  }
  fmt.Printf("%s\n", rootDeviceName)

  userData, err := extendedserverattributes.ExtractUserData(result.Result)
  if err != nil {
    panic(err)
  }
  fmt.Printf("%s\n", userData)
*/
package extendedserverattributes
