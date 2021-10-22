/*
Package resetstate provides functionality to reset the state of a server that has
been provisioned by the OpenStack Compute service.

Example to Reset a Server

	serverID := "47b6b7b7-568d-40e4-868c-d5c41735532e"
	err := resetstate.ResetState(client, id, resetstate.StateActive).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package resetstate
