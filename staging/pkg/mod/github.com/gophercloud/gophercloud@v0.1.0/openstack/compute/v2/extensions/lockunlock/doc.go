/*
Package lockunlock provides functionality to lock and unlock servers that
have been provisioned by the OpenStack Compute service.

Example to Lock and Unlock a Server

	serverID := "47b6b7b7-568d-40e4-868c-d5c41735532e"

	err := lockunlock.Lock(computeClient, serverID).ExtractErr()
	if err != nil {
		panic(err)
	}

	err = lockunlock.Unlock(computeClient, serverID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package lockunlock
