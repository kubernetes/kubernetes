// +build acceptance compute servers

package v2

import (
	"os"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/openstack"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	"github.com/rackspace/gophercloud/openstack/networking/v2/networks"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestListServers(t *testing.T) {
	client, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	t.Logf("ID\tRegion\tName\tStatus\tIPv4\tIPv6")

	pager := servers.List(client, servers.ListOpts{})
	count, pages := 0, 0
	pager.EachPage(func(page pagination.Page) (bool, error) {
		pages++
		t.Logf("---")

		servers, err := servers.ExtractServers(page)
		if err != nil {
			return false, err
		}

		for _, s := range servers {
			t.Logf("%s\t%s\t%s\t%s\t%s\t\n", s.ID, s.Name, s.Status, s.AccessIPv4, s.AccessIPv6)
			count++
		}

		return true, nil
	})

	t.Logf("--------\n%d servers listed on %d pages.\n", count, pages)
}

func networkingClient() (*gophercloud.ServiceClient, error) {
	opts, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	provider, err := openstack.AuthenticatedClient(opts)
	if err != nil {
		return nil, err
	}

	return openstack.NewNetworkV2(provider, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

func createServer(t *testing.T, client *gophercloud.ServiceClient, choices *ComputeChoices) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	var network networks.Network

	networkingClient, err := networkingClient()
	if err != nil {
		t.Fatalf("Unable to create a networking client: %v", err)
	}

	pager := networks.List(networkingClient, networks.ListOpts{
		Name:  choices.NetworkName,
		Limit: 1,
	})
	pager.EachPage(func(page pagination.Page) (bool, error) {
		networks, err := networks.ExtractNetworks(page)
		if err != nil {
			t.Errorf("Failed to extract networks: %v", err)
			return false, err
		}

		if len(networks) == 0 {
			t.Fatalf("No networks to attach to server")
			return false, err
		}

		network = networks[0]

		return false, nil
	})

	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create server: %s\n", name)

	pwd := tools.MakeNewPassword("")

	server, err := servers.Create(client, servers.CreateOpts{
		Name:      name,
		FlavorRef: choices.FlavorID,
		ImageRef:  choices.ImageID,
		Networks: []servers.Network{
			servers.Network{UUID: network.ID},
		},
		AdminPass: pwd,
		Personality: servers.Personality{
			&servers.File{
				Path:     "/etc/test",
				Contents: []byte("hello world"),
			},
		},
	}).Extract()
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}

	th.AssertEquals(t, pwd, server.AdminPass)

	return server, err
}

func TestCreateDestroyServer(t *testing.T) {
	choices, err := ComputeChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	client, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	server, err := createServer(t, client, choices)
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}
	defer func() {
		servers.Delete(client, server.ID)
		t.Logf("Server deleted.")
	}()

	if err = waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatalf("Unable to wait for server: %v", err)
	}

	pager := servers.ListAddresses(client, server.ID)
	pager.EachPage(func(page pagination.Page) (bool, error) {
		networks, err := servers.ExtractAddresses(page)
		if err != nil {
			return false, err
		}

		for n, a := range networks {
			t.Logf("%s: %+v\n", n, a)
		}
		return true, nil
	})

	pager = servers.ListAddressesByNetwork(client, server.ID, choices.NetworkName)
	pager.EachPage(func(page pagination.Page) (bool, error) {
		addresses, err := servers.ExtractNetworkAddresses(page)
		if err != nil {
			return false, err
		}

		for _, a := range addresses {
			t.Logf("%+v\n", a)
		}
		return true, nil
	})
}

func TestUpdateServer(t *testing.T) {
	client, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	choices, err := ComputeChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	server, err := createServer(t, client, choices)
	if err != nil {
		t.Fatal(err)
	}
	defer servers.Delete(client, server.ID)

	if err = waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}

	alternateName := tools.RandomString("ACPTTEST", 16)
	for alternateName == server.Name {
		alternateName = tools.RandomString("ACPTTEST", 16)
	}

	t.Logf("Attempting to rename the server to %s.", alternateName)

	updated, err := servers.Update(client, server.ID, servers.UpdateOpts{Name: alternateName}).Extract()
	if err != nil {
		t.Fatalf("Unable to rename server: %v", err)
	}

	if updated.ID != server.ID {
		t.Errorf("Updated server ID [%s] didn't match original server ID [%s]!", updated.ID, server.ID)
	}

	err = tools.WaitFor(func() (bool, error) {
		latest, err := servers.Get(client, updated.ID).Extract()
		if err != nil {
			return false, err
		}

		return latest.Name == alternateName, nil
	})
}

func TestActionChangeAdminPassword(t *testing.T) {
	t.Parallel()

	client, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	choices, err := ComputeChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	server, err := createServer(t, client, choices)
	if err != nil {
		t.Fatal(err)
	}
	defer servers.Delete(client, server.ID)

	if err = waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}

	randomPassword := tools.MakeNewPassword(server.AdminPass)
	res := servers.ChangeAdminPassword(client, server.ID, randomPassword)
	if res.Err != nil {
		t.Fatal(err)
	}

	if err = waitForStatus(client, server, "PASSWORD"); err != nil {
		t.Fatal(err)
	}

	if err = waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}
}

func TestActionReboot(t *testing.T) {
	t.Parallel()

	client, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	choices, err := ComputeChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	server, err := createServer(t, client, choices)
	if err != nil {
		t.Fatal(err)
	}
	defer servers.Delete(client, server.ID)

	if err = waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}

	res := servers.Reboot(client, server.ID, "aldhjflaskhjf")
	if res.Err == nil {
		t.Fatal("Expected the SDK to provide an ArgumentError here")
	}

	t.Logf("Attempting reboot of server %s", server.ID)
	res = servers.Reboot(client, server.ID, servers.OSReboot)
	if res.Err != nil {
		t.Fatalf("Unable to reboot server: %v", err)
	}

	if err = waitForStatus(client, server, "REBOOT"); err != nil {
		t.Fatal(err)
	}

	if err = waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}
}

func TestActionRebuild(t *testing.T) {
	t.Parallel()

	client, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	choices, err := ComputeChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	server, err := createServer(t, client, choices)
	if err != nil {
		t.Fatal(err)
	}
	defer servers.Delete(client, server.ID)

	if err = waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}

	t.Logf("Attempting to rebuild server %s", server.ID)

	rebuildOpts := servers.RebuildOpts{
		Name:      tools.RandomString("ACPTTEST", 16),
		AdminPass: tools.MakeNewPassword(server.AdminPass),
		ImageID:   choices.ImageID,
	}

	rebuilt, err := servers.Rebuild(client, server.ID, rebuildOpts).Extract()
	if err != nil {
		t.Fatal(err)
	}

	if rebuilt.ID != server.ID {
		t.Errorf("Expected rebuilt server ID of [%s]; got [%s]", server.ID, rebuilt.ID)
	}

	if err = waitForStatus(client, rebuilt, "REBUILD"); err != nil {
		t.Fatal(err)
	}

	if err = waitForStatus(client, rebuilt, "ACTIVE"); err != nil {
		t.Fatal(err)
	}
}

func resizeServer(t *testing.T, client *gophercloud.ServiceClient, server *servers.Server, choices *ComputeChoices) {
	if err := waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}

	t.Logf("Attempting to resize server [%s]", server.ID)

	opts := &servers.ResizeOpts{
		FlavorRef: choices.FlavorIDResize,
	}
	if res := servers.Resize(client, server.ID, opts); res.Err != nil {
		t.Fatal(res.Err)
	}

	if err := waitForStatus(client, server, "VERIFY_RESIZE"); err != nil {
		t.Fatal(err)
	}
}

func TestActionResizeConfirm(t *testing.T) {
	t.Parallel()

	choices, err := ComputeChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	client, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	server, err := createServer(t, client, choices)
	if err != nil {
		t.Fatal(err)
	}
	defer servers.Delete(client, server.ID)
	resizeServer(t, client, server, choices)

	t.Logf("Attempting to confirm resize for server %s", server.ID)

	if res := servers.ConfirmResize(client, server.ID); res.Err != nil {
		t.Fatal(err)
	}

	if err = waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}
}

func TestActionResizeRevert(t *testing.T) {
	t.Parallel()

	choices, err := ComputeChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	client, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	server, err := createServer(t, client, choices)
	if err != nil {
		t.Fatal(err)
	}
	defer servers.Delete(client, server.ID)
	resizeServer(t, client, server, choices)

	t.Logf("Attempting to revert resize for server %s", server.ID)

	if res := servers.RevertResize(client, server.ID); res.Err != nil {
		t.Fatal(err)
	}

	if err = waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}
}

func TestServerMetadata(t *testing.T) {
	t.Parallel()

	choices, err := ComputeChoicesFromEnv()
	th.AssertNoErr(t, err)

	client, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	server, err := createServer(t, client, choices)
	if err != nil {
		t.Fatal(err)
	}
	defer servers.Delete(client, server.ID)
	if err = waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}

	metadata, err := servers.UpdateMetadata(client, server.ID, servers.MetadataOpts{
		"foo":  "bar",
		"this": "that",
	}).Extract()
	th.AssertNoErr(t, err)
	t.Logf("UpdateMetadata result: %+v\n", metadata)

	err = servers.DeleteMetadatum(client, server.ID, "foo").ExtractErr()
	th.AssertNoErr(t, err)

	metadata, err = servers.CreateMetadatum(client, server.ID, servers.MetadatumOpts{
		"foo": "baz",
	}).Extract()
	th.AssertNoErr(t, err)
	t.Logf("CreateMetadatum result: %+v\n", metadata)

	metadata, err = servers.Metadatum(client, server.ID, "foo").Extract()
	th.AssertNoErr(t, err)
	t.Logf("Metadatum result: %+v\n", metadata)
	th.AssertEquals(t, "baz", metadata["foo"])

	metadata, err = servers.Metadata(client, server.ID).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Metadata result: %+v\n", metadata)

	metadata, err = servers.ResetMetadata(client, server.ID, servers.MetadataOpts{}).Extract()
	th.AssertNoErr(t, err)
	t.Logf("ResetMetadata result: %+v\n", metadata)
	th.AssertDeepEquals(t, map[string]string{}, metadata)
}
