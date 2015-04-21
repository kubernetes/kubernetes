// +build acceptance

package v2

import (
	"crypto/rand"
	"crypto/rsa"
	"testing"

	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/keypairs"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	th "github.com/rackspace/gophercloud/testhelper"

	"golang.org/x/crypto/ssh"
)

const keyName = "gophercloud_test_key_pair"

func TestCreateServerWithKeyPair(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	publicKey := privateKey.PublicKey
	pub, err := ssh.NewPublicKey(&publicKey)
	th.AssertNoErr(t, err)
	pubBytes := ssh.MarshalAuthorizedKey(pub)
	pk := string(pubBytes)

	kp, err := keypairs.Create(client, keypairs.CreateOpts{
		Name:      keyName,
		PublicKey: pk,
	}).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created key pair: %s\n", kp)

	choices, err := ComputeChoicesFromEnv()
	th.AssertNoErr(t, err)

	name := tools.RandomString("Gophercloud-", 8)
	t.Logf("Creating server [%s] with key pair.", name)

	serverCreateOpts := servers.CreateOpts{
		Name:      name,
		FlavorRef: choices.FlavorID,
		ImageRef:  choices.ImageID,
	}

	server, err := servers.Create(client, keypairs.CreateOptsExt{
		serverCreateOpts,
		keyName,
	}).Extract()
	th.AssertNoErr(t, err)
	defer servers.Delete(client, server.ID)
	if err = waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatalf("Unable to wait for server: %v", err)
	}

	server, err = servers.Get(client, server.ID).Extract()
	t.Logf("Created server: %+v\n", server)
	th.AssertNoErr(t, err)
	th.AssertEquals(t, server.KeyName, keyName)

	t.Logf("Deleting key pair [%s]...", kp.Name)
	err = keypairs.Delete(client, keyName).ExtractErr()
	th.AssertNoErr(t, err)

	t.Logf("Deleting server [%s]...", name)
}
