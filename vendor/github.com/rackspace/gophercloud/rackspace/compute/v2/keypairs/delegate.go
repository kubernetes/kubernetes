package keypairs

import (
	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/keypairs"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns a Pager that allows you to iterate over a collection of KeyPairs.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	return os.List(client)
}

// Create requests the creation of a new keypair on the server, or to import a pre-existing
// keypair.
func Create(client *gophercloud.ServiceClient, opts os.CreateOptsBuilder) os.CreateResult {
	return os.Create(client, opts)
}

// Get returns public data about a previously uploaded KeyPair.
func Get(client *gophercloud.ServiceClient, name string) os.GetResult {
	return os.Get(client, name)
}

// Delete requests the deletion of a previous stored KeyPair from the server.
func Delete(client *gophercloud.ServiceClient, name string) os.DeleteResult {
	return os.Delete(client, name)
}

// ExtractKeyPairs interprets a page of results as a slice of KeyPairs.
func ExtractKeyPairs(page pagination.Page) ([]os.KeyPair, error) {
	return os.ExtractKeyPairs(page)
}
