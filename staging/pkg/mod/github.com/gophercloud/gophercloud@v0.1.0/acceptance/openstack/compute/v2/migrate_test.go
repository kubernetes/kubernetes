// +build acceptance compute servers

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/migrate"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestMigrate(t *testing.T) {
	t.Skip("This is not passing in OpenLab. Works locally")

	clients.RequireLong(t)
	clients.RequireAdmin(t)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	server, err := CreateServer(t, client)
	th.AssertNoErr(t, err)
	defer DeleteServer(t, client, server)

	t.Logf("Attempting to migrate server %s", server.ID)

	err = migrate.Migrate(client, server.ID).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestLiveMigrate(t *testing.T) {
	clients.RequireLong(t)
	clients.RequireAdmin(t)
	clients.RequireLiveMigration(t)

	client, err := clients.NewComputeV2Client()
	th.AssertNoErr(t, err)

	server, err := CreateServer(t, client)
	th.AssertNoErr(t, err)
	defer DeleteServer(t, client, server)

	t.Logf("Attempting to migrate server %s", server.ID)

	blockMigration := false
	diskOverCommit := false

	liveMigrateOpts := migrate.LiveMigrateOpts{
		BlockMigration: &blockMigration,
		DiskOverCommit: &diskOverCommit,
	}

	err = migrate.LiveMigrate(client, server.ID, liveMigrateOpts).ExtractErr()
	th.AssertNoErr(t, err)
}
