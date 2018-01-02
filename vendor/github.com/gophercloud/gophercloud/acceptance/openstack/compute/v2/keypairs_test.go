// +build acceptance compute keypairs

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/keypairs"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
)

const keyName = "gophercloud_test_key_pair"

func TestKeypairsList(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	allPages, err := keypairs.List(client).AllPages()
	if err != nil {
		t.Fatalf("Unable to retrieve keypairs: %s", err)
	}

	allKeys, err := keypairs.ExtractKeyPairs(allPages)
	if err != nil {
		t.Fatalf("Unable to extract keypairs results: %s", err)
	}

	for _, keypair := range allKeys {
		tools.PrintResource(t, keypair)
	}
}

func TestKeypairsCreate(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	keyPair, err := CreateKeyPair(t, client)
	if err != nil {
		t.Fatalf("Unable to create key pair: %v", err)
	}
	defer DeleteKeyPair(t, client, keyPair)

	tools.PrintResource(t, keyPair)
}

func TestKeypairsImportPublicKey(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	publicKey, err := createKey()
	if err != nil {
		t.Fatalf("Unable to create public key: %s", err)
	}

	keyPair, err := ImportPublicKey(t, client, publicKey)
	if err != nil {
		t.Fatalf("Unable to create keypair: %s", err)
	}
	defer DeleteKeyPair(t, client, keyPair)

	tools.PrintResource(t, keyPair)
}

func TestKeypairsServerCreateWithKey(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	publicKey, err := createKey()
	if err != nil {
		t.Fatalf("Unable to create public key: %s", err)
	}

	keyPair, err := ImportPublicKey(t, client, publicKey)
	if err != nil {
		t.Fatalf("Unable to create keypair: %s", err)
	}
	defer DeleteKeyPair(t, client, keyPair)

	server, err := CreateServerWithPublicKey(t, client, keyPair.Name)
	if err != nil {
		t.Fatalf("Unable to create server: %s", err)
	}
	defer DeleteServer(t, client, server)

	server, err = servers.Get(client, server.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to retrieve server: %s", err)
	}

	if server.KeyName != keyPair.Name {
		t.Fatalf("key name of server %s is %s, not %s", server.ID, server.KeyName, keyPair.Name)
	}
}
