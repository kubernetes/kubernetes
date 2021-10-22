/*
Package pauseunpause provides functionality to pause and unpause servers that
have been provisioned by the OpenStack Compute service.

Example to Pause and Unpause a Server

	serverID := "32c8baf7-1cdb-4cc2-bc31-c3a55b89f56b"
	err := pauseunpause.Pause(computeClient, serverID).ExtractErr()
	if err != nil {
		panic(err)
	}

	err = pauseunpause.Unpause(computeClient, serverID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package pauseunpause
