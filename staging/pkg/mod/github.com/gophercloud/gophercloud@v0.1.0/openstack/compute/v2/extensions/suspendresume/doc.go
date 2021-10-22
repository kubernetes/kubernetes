/*
Package suspendresume provides functionality to suspend and resume servers that have
been provisioned by the OpenStack Compute service.

Example to Suspend and Resume a Server

	serverID := "47b6b7b7-568d-40e4-868c-d5c41735532e"

	err := suspendresume.Suspend(computeClient, serverID).ExtractErr()
	if err != nil {
		panic(err)
	}

	err := suspendresume.Resume(computeClient, serverID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package suspendresume
