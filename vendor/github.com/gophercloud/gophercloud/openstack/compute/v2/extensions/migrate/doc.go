/*
Package migrate provides functionality to migrate servers that have been
provisioned by the OpenStack Compute service.

Example to Migrate a Server

	serverID := "b16ba811-199d-4ffd-8839-ba96c1185a67"
	err := migrate.Migrate(computeClient, serverID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package migrate
