// +build acceptance rackspace

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	os "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/keypairs"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/keypairs"
	th "github.com/rackspace/gophercloud/testhelper"
)

func deleteKeyPair(t *testing.T, client *gophercloud.ServiceClient, name string) {
	err := keypairs.Delete(client, name).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Successfully deleted key [%s].", name)
}

func TestCreateKeyPair(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	name := tools.RandomString("createdkey-", 8)
	k, err := keypairs.Create(client, os.CreateOpts{Name: name}).Extract()
	th.AssertNoErr(t, err)
	defer deleteKeyPair(t, client, name)

	t.Logf("Created a new keypair:")
	t.Logf("        name=[%s]", k.Name)
	t.Logf(" fingerprint=[%s]", k.Fingerprint)
	t.Logf("   publickey=[%s]", tools.Elide(k.PublicKey))
	t.Logf("  privatekey=[%s]", tools.Elide(k.PrivateKey))
	t.Logf("      userid=[%s]", k.UserID)
}

func TestImportKeyPair(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	name := tools.RandomString("importedkey-", 8)
	pubkey := "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDlIQ3r+zd97kb9Hzmujd3V6pbO53eb3Go4q2E8iqVGWQfZTrFdL9KACJnqJIm9HmncfRkUTxE37hqeGCCv8uD+ZPmPiZG2E60OX1mGDjbbzAyReRwYWXgXHopggZTLak5k4mwZYaxwaufbVBDRn847e01lZnaXaszEToLM37NLw+uz29sl3TwYy2R0RGHPwPc160aWmdLjSyd1Nd4c9pvvOP/EoEuBjIC6NJJwg2Rvg9sjjx9jYj0QUgc8CqKLN25oMZ69kNJzlFylKRUoeeVr89txlR59yehJWk6Uw6lYFTdJmcmQOFVAJ12RMmS1hLWCM8UzAgtw+EDa0eqBxBDl smash@winter"

	k, err := keypairs.Create(client, os.CreateOpts{
		Name:      name,
		PublicKey: pubkey,
	}).Extract()
	th.AssertNoErr(t, err)
	defer deleteKeyPair(t, client, name)

	th.CheckEquals(t, pubkey, k.PublicKey)
	th.CheckEquals(t, "", k.PrivateKey)

	t.Logf("Imported an existing keypair:")
	t.Logf("        name=[%s]", k.Name)
	t.Logf(" fingerprint=[%s]", k.Fingerprint)
	t.Logf("   publickey=[%s]", tools.Elide(k.PublicKey))
	t.Logf("  privatekey=[%s]", tools.Elide(k.PrivateKey))
	t.Logf("      userid=[%s]", k.UserID)
}

func TestListKeyPairs(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	count := 0
	err = keypairs.List(client).EachPage(func(page pagination.Page) (bool, error) {
		count++
		t.Logf("--- %02d ---", count)

		ks, err := keypairs.ExtractKeyPairs(page)
		th.AssertNoErr(t, err)

		for i, keypair := range ks {
			t.Logf("[%02d]    name=[%s]", i, keypair.Name)
			t.Logf(" fingerprint=[%s]", keypair.Fingerprint)
			t.Logf("   publickey=[%s]", tools.Elide(keypair.PublicKey))
			t.Logf("  privatekey=[%s]", tools.Elide(keypair.PrivateKey))
			t.Logf("      userid=[%s]", keypair.UserID)
		}

		return true, nil
	})
	th.AssertNoErr(t, err)
}
