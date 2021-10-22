/*
Package migrate provides functionality to migrate servers that have been
provisioned by the OpenStack Compute service.

Example of Migrate Server (migrate Action)

	serverID := "b16ba811-199d-4ffd-8839-ba96c1185a67"
	err := migrate.Migrate(computeClient, serverID).ExtractErr()
	if err != nil {
		panic(err)
	}

Example of Live-Migrate Server (os-migrateLive Action)

	serverID := "b16ba811-199d-4ffd-8839-ba96c1185a67"
	host := "01c0cadef72d47e28a672a76060d492c"
	blockMigration := false

	migrationOpts := migrate.LiveMigrateOpts{
		Host: &host,
		BlockMigration: &blockMigration,
	}

	err := migrate.LiveMigrate(computeClient, serverID, migrationOpts).ExtractErr()
	if err != nil {
		panic(err)
	}

*/
package migrate
