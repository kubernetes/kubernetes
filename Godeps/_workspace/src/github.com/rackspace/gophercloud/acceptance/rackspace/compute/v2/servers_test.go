// +build acceptance

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/diskconfig"
	oskey "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/keypairs"
	os "github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/keypairs"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/servers"
	th "github.com/rackspace/gophercloud/testhelper"
)

func createServerKeyPair(t *testing.T, client *gophercloud.ServiceClient) *oskey.KeyPair {
	name := tools.RandomString("importedkey-", 8)
	pubkey := "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDlIQ3r+zd97kb9Hzmujd3V6pbO53eb3Go4q2E8iqVGWQfZTrFdL9KACJnqJIm9HmncfRkUTxE37hqeGCCv8uD+ZPmPiZG2E60OX1mGDjbbzAyReRwYWXgXHopggZTLak5k4mwZYaxwaufbVBDRn847e01lZnaXaszEToLM37NLw+uz29sl3TwYy2R0RGHPwPc160aWmdLjSyd1Nd4c9pvvOP/EoEuBjIC6NJJwg2Rvg9sjjx9jYj0QUgc8CqKLN25oMZ69kNJzlFylKRUoeeVr89txlR59yehJWk6Uw6lYFTdJmcmQOFVAJ12RMmS1hLWCM8UzAgtw+EDa0eqBxBDl smash@winter"

	k, err := keypairs.Create(client, oskey.CreateOpts{
		Name:      name,
		PublicKey: pubkey,
	}).Extract()
	th.AssertNoErr(t, err)

	return k
}

func createServer(t *testing.T, client *gophercloud.ServiceClient, keyName string) *os.Server {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	options, err := optionsFromEnv()
	th.AssertNoErr(t, err)

	name := tools.RandomString("Gophercloud-", 8)

	pwd := tools.MakeNewPassword("")

	opts := &servers.CreateOpts{
		Name:       name,
		ImageRef:   options.imageID,
		FlavorRef:  options.flavorID,
		DiskConfig: diskconfig.Manual,
		AdminPass:  pwd,
	}

	if keyName != "" {
		opts.KeyPair = keyName
	}

	t.Logf("Creating server [%s].", name)
	s, err := servers.Create(client, opts).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Creating server.")

	err = servers.WaitForStatus(client, s.ID, "ACTIVE", 300)
	th.AssertNoErr(t, err)
	t.Logf("Server created successfully.")

	th.CheckEquals(t, pwd, s.AdminPass)

	return s
}

func logServer(t *testing.T, server *os.Server, index int) {
	if index == -1 {
		t.Logf("             id=[%s]", server.ID)
	} else {
		t.Logf("[%02d]             id=[%s]", index, server.ID)
	}
	t.Logf("           name=[%s]", server.Name)
	t.Logf("      tenant ID=[%s]", server.TenantID)
	t.Logf("        user ID=[%s]", server.UserID)
	t.Logf("        updated=[%s]", server.Updated)
	t.Logf("        created=[%s]", server.Created)
	t.Logf("        host ID=[%s]", server.HostID)
	t.Logf("    access IPv4=[%s]", server.AccessIPv4)
	t.Logf("    access IPv6=[%s]", server.AccessIPv6)
	t.Logf("          image=[%v]", server.Image)
	t.Logf("         flavor=[%v]", server.Flavor)
	t.Logf("      addresses=[%v]", server.Addresses)
	t.Logf("       metadata=[%v]", server.Metadata)
	t.Logf("          links=[%v]", server.Links)
	t.Logf("        keyname=[%s]", server.KeyName)
	t.Logf(" admin password=[%s]", server.AdminPass)
	t.Logf("         status=[%s]", server.Status)
	t.Logf("       progress=[%d]", server.Progress)
}

func getServer(t *testing.T, client *gophercloud.ServiceClient, server *os.Server) {
	t.Logf("> servers.Get")

	details, err := servers.Get(client, server.ID).Extract()
	th.AssertNoErr(t, err)
	logServer(t, details, -1)
}

func updateServer(t *testing.T, client *gophercloud.ServiceClient, server *os.Server) {
	t.Logf("> servers.Get")

	opts := os.UpdateOpts{
		Name: "updated-server",
	}
	updatedServer, err := servers.Update(client, server.ID, opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, "updated-server", updatedServer.Name)
	logServer(t, updatedServer, -1)
}

func listServers(t *testing.T, client *gophercloud.ServiceClient) {
	t.Logf("> servers.List")

	count := 0
	err := servers.List(client, nil).EachPage(func(page pagination.Page) (bool, error) {
		count++
		t.Logf("--- Page %02d ---", count)

		s, err := servers.ExtractServers(page)
		th.AssertNoErr(t, err)
		for index, server := range s {
			logServer(t, &server, index)
		}

		return true, nil
	})
	th.AssertNoErr(t, err)
}

func changeAdminPassword(t *testing.T, client *gophercloud.ServiceClient, server *os.Server) {
	t.Logf("> servers.ChangeAdminPassword")

	original := server.AdminPass

	t.Logf("Changing server password.")
	err := servers.ChangeAdminPassword(client, server.ID, tools.MakeNewPassword(original)).ExtractErr()
	th.AssertNoErr(t, err)

	err = servers.WaitForStatus(client, server.ID, "ACTIVE", 300)
	th.AssertNoErr(t, err)
	t.Logf("Password changed successfully.")
}

func rebootServer(t *testing.T, client *gophercloud.ServiceClient, server *os.Server) {
	t.Logf("> servers.Reboot")

	err := servers.Reboot(client, server.ID, os.HardReboot).ExtractErr()
	th.AssertNoErr(t, err)

	err = servers.WaitForStatus(client, server.ID, "ACTIVE", 300)
	th.AssertNoErr(t, err)

	t.Logf("Server successfully rebooted.")
}

func rebuildServer(t *testing.T, client *gophercloud.ServiceClient, server *os.Server) {
	t.Logf("> servers.Rebuild")

	options, err := optionsFromEnv()
	th.AssertNoErr(t, err)

	opts := servers.RebuildOpts{
		Name:       tools.RandomString("RenamedGopher", 16),
		AdminPass:  tools.MakeNewPassword(server.AdminPass),
		ImageID:    options.imageID,
		DiskConfig: diskconfig.Manual,
	}
	after, err := servers.Rebuild(client, server.ID, opts).Extract()
	th.AssertNoErr(t, err)
	th.CheckEquals(t, after.ID, server.ID)

	err = servers.WaitForStatus(client, after.ID, "ACTIVE", 300)
	th.AssertNoErr(t, err)

	t.Logf("Server successfully rebuilt.")
	logServer(t, after, -1)
}

func deleteServer(t *testing.T, client *gophercloud.ServiceClient, server *os.Server) {
	t.Logf("> servers.Delete")

	res := servers.Delete(client, server.ID)
	th.AssertNoErr(t, res.Err)

	t.Logf("Server deleted successfully.")
}

func deleteServerKeyPair(t *testing.T, client *gophercloud.ServiceClient, k *oskey.KeyPair) {
	t.Logf("> keypairs.Delete")

	err := keypairs.Delete(client, k.Name).ExtractErr()
	th.AssertNoErr(t, err)

	t.Logf("Keypair deleted successfully.")
}

func TestServerOperations(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	kp := createServerKeyPair(t, client)
	defer deleteServerKeyPair(t, client, kp)

	server := createServer(t, client, kp.Name)
	defer deleteServer(t, client, server)

	getServer(t, client, server)
	updateServer(t, client, server)
	listServers(t, client)
	changeAdminPassword(t, client, server)
	rebootServer(t, client, server)
	rebuildServer(t, client, server)
}
