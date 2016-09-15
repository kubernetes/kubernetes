// +build acceptance compute servers

package v2

import (
	"strings"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestServersList(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	allPages, err := servers.List(client, servers.ListOpts{}).AllPages()
	if err != nil {
		t.Fatalf("Unable to retrieve servers: %v", err)
	}

	allServers, err := servers.ExtractServers(allPages)
	if err != nil {
		t.Fatalf("Unable to extract servers: %v", err)
	}

	for _, server := range allServers {
		PrintServer(t, &server)
	}
}

func TestServersCreateDestroy(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	server, err := CreateServer(t, client, choices)
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}

	defer DeleteServer(t, client, server)

	newServer, err := servers.Get(client, server.ID).Extract()
	if err != nil {
		t.Errorf("Unable to retrieve server: %v", err)
	}
	PrintServer(t, newServer)

	allAddressPages, err := servers.ListAddresses(client, server.ID).AllPages()
	if err != nil {
		t.Errorf("Unable to list server addresses: %v", err)
	}

	allAddresses, err := servers.ExtractAddresses(allAddressPages)
	if err != nil {
		t.Errorf("Unable to extract server addresses: %v", err)
	}

	for network, address := range allAddresses {
		t.Logf("Addresses on %s: %+v", network, address)
	}

	allNetworkAddressPages, err := servers.ListAddressesByNetwork(client, server.ID, choices.NetworkName).AllPages()
	if err != nil {
		t.Errorf("Unable to list server addresses: %v", err)
	}

	allNetworkAddresses, err := servers.ExtractNetworkAddresses(allNetworkAddressPages)
	if err != nil {
		t.Errorf("Unable to extract server addresses: %v", err)
	}

	t.Logf("Addresses on %s:", choices.NetworkName)
	for _, address := range allNetworkAddresses {
		t.Logf("%+v", address)
	}
}

func TestServersWithoutImageRef(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	server, err := CreateServerWithoutImageRef(t, client, choices)
	if err != nil {
		if err400, ok := err.(*gophercloud.ErrUnexpectedResponseCode); ok {
			if !strings.Contains("Missing imageRef attribute", string(err400.Body)) {
				defer DeleteServer(t, client, server)
			}
		}
	}
}

func TestServersUpdate(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	server, err := CreateServer(t, client, choices)
	if err != nil {
		t.Fatal(err)
	}
	defer DeleteServer(t, client, server)

	alternateName := tools.RandomString("ACPTTEST", 16)
	for alternateName == server.Name {
		alternateName = tools.RandomString("ACPTTEST", 16)
	}

	t.Logf("Attempting to rename the server to %s.", alternateName)

	updateOpts := servers.UpdateOpts{
		Name: alternateName,
	}

	updated, err := servers.Update(client, server.ID, updateOpts).Extract()
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

func TestServersMetadata(t *testing.T) {
	t.Parallel()

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	server, err := CreateServer(t, client, choices)
	if err != nil {
		t.Fatal(err)
	}
	defer DeleteServer(t, client, server)

	metadata, err := servers.UpdateMetadata(client, server.ID, servers.MetadataOpts{
		"foo":  "bar",
		"this": "that",
	}).Extract()
	if err != nil {
		t.Fatalf("Unable to update metadata: %v", err)
	}
	t.Logf("UpdateMetadata result: %+v\n", metadata)

	err = servers.DeleteMetadatum(client, server.ID, "foo").ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete metadatum: %v", err)
	}

	metadata, err = servers.CreateMetadatum(client, server.ID, servers.MetadatumOpts{
		"foo": "baz",
	}).Extract()
	if err != nil {
		t.Fatalf("Unable to create metadatum: %v", err)
	}
	t.Logf("CreateMetadatum result: %+v\n", metadata)

	metadata, err = servers.Metadatum(client, server.ID, "foo").Extract()
	if err != nil {
		t.Fatalf("Unable to get metadatum: %v", err)
	}
	t.Logf("Metadatum result: %+v\n", metadata)
	th.AssertEquals(t, "baz", metadata["foo"])

	metadata, err = servers.Metadata(client, server.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get metadata: %v", err)
	}
	t.Logf("Metadata result: %+v\n", metadata)

	metadata, err = servers.ResetMetadata(client, server.ID, servers.MetadataOpts{}).Extract()
	if err != nil {
		t.Fatalf("Unable to reset metadata: %v", err)
	}
	t.Logf("ResetMetadata result: %+v\n", metadata)
	th.AssertDeepEquals(t, map[string]string{}, metadata)
}

func TestServersActionChangeAdminPassword(t *testing.T) {
	t.Parallel()

	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	server, err := CreateServer(t, client, choices)
	if err != nil {
		t.Fatal(err)
	}
	defer DeleteServer(t, client, server)

	randomPassword := tools.MakeNewPassword(server.AdminPass)
	res := servers.ChangeAdminPassword(client, server.ID, randomPassword)
	if res.Err != nil {
		t.Fatal(res.Err)
	}

	if err = WaitForComputeStatus(client, server, "PASSWORD"); err != nil {
		t.Fatal(err)
	}

	if err = WaitForComputeStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}
}

func TestServersActionReboot(t *testing.T) {
	t.Parallel()

	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	server, err := CreateServer(t, client, choices)
	if err != nil {
		t.Fatal(err)
	}
	defer DeleteServer(t, client, server)

	rebootOpts := &servers.RebootOpts{
		Type: servers.SoftReboot,
	}

	t.Logf("Attempting reboot of server %s", server.ID)
	res := servers.Reboot(client, server.ID, rebootOpts)
	if res.Err != nil {
		t.Fatalf("Unable to reboot server: %v", res.Err)
	}

	if err = WaitForComputeStatus(client, server, "REBOOT"); err != nil {
		t.Fatal(err)
	}

	if err = WaitForComputeStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}
}

func TestServersActionRebuild(t *testing.T) {
	t.Parallel()

	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	server, err := CreateServer(t, client, choices)
	if err != nil {
		t.Fatal(err)
	}
	defer DeleteServer(t, client, server)

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

	if err = WaitForComputeStatus(client, rebuilt, "REBUILD"); err != nil {
		t.Fatal(err)
	}

	if err = WaitForComputeStatus(client, rebuilt, "ACTIVE"); err != nil {
		t.Fatal(err)
	}
}

func TestServersActionResizeConfirm(t *testing.T) {
	t.Parallel()

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	server, err := CreateServer(t, client, choices)
	if err != nil {
		t.Fatal(err)
	}
	defer DeleteServer(t, client, server)

	t.Logf("Attempting to resize server %s", server.ID)
	ResizeServer(t, client, server, choices)

	t.Logf("Attempting to confirm resize for server %s", server.ID)
	if res := servers.ConfirmResize(client, server.ID); res.Err != nil {
		t.Fatal(res.Err)
	}

	if err = WaitForComputeStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}
}

func TestServersActionResizeRevert(t *testing.T) {
	t.Parallel()

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	server, err := CreateServer(t, client, choices)
	if err != nil {
		t.Fatal(err)
	}
	defer DeleteServer(t, client, server)

	t.Logf("Attempting to resize server %s", server.ID)
	ResizeServer(t, client, server, choices)

	t.Logf("Attempting to revert resize for server %s", server.ID)
	if res := servers.RevertResize(client, server.ID); res.Err != nil {
		t.Fatal(res.Err)
	}

	if err = WaitForComputeStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}
}
