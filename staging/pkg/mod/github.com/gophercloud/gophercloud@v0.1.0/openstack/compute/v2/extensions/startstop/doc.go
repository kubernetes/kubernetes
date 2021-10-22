/*
Package startstop provides functionality to start and stop servers that have
been provisioned by the OpenStack Compute service.

Example to Stop and Start a Server

	serverID := "47b6b7b7-568d-40e4-868c-d5c41735532e"

	err := startstop.Stop(computeClient, serverID).ExtractErr()
	if err != nil {
		panic(err)
	}

	err := startstop.Start(computeClient, serverID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package startstop
