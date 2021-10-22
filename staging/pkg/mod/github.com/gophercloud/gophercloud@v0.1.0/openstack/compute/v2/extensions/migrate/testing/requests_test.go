package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/migrate"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

const serverID = "b16ba811-199d-4ffd-8839-ba96c1185a67"

func TestMigrate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockMigrateResponse(t, serverID)

	err := migrate.Migrate(client.ServiceClient(), serverID).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestLiveMigrate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockLiveMigrateResponse(t, serverID)

	host := "01c0cadef72d47e28a672a76060d492c"
	blockMigration := false
	diskOverCommit := true

	migrationOpts := migrate.LiveMigrateOpts{
		Host:           &host,
		BlockMigration: &blockMigration,
		DiskOverCommit: &diskOverCommit,
	}

	err := migrate.LiveMigrate(client.ServiceClient(), serverID, migrationOpts).ExtractErr()
	th.AssertNoErr(t, err)
}
