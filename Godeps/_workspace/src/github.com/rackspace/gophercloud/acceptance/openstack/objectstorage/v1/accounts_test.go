// +build acceptance

package v1

import (
	"strings"
	"testing"

	"github.com/rackspace/gophercloud/openstack/objectstorage/v1/accounts"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestAccounts(t *testing.T) {
	// Create a provider client for making the HTTP requests.
	// See common.go in this directory for more information.
	client := newClient(t)

	// Update an account's metadata.
	updateres := accounts.Update(client, accounts.UpdateOpts{Metadata: metadata})
	th.AssertNoErr(t, updateres.Err)

	// Defer the deletion of the metadata set above.
	defer func() {
		tempMap := make(map[string]string)
		for k := range metadata {
			tempMap[k] = ""
		}
		updateres = accounts.Update(client, accounts.UpdateOpts{Metadata: tempMap})
		th.AssertNoErr(t, updateres.Err)
	}()

	// Retrieve account metadata.
	getres := accounts.Get(client, nil)
	th.AssertNoErr(t, getres.Err)
	// Extract the custom metadata from the 'Get' response.
	am, err := getres.ExtractMetadata()
	th.AssertNoErr(t, err)
	for k := range metadata {
		if am[k] != metadata[strings.Title(k)] {
			t.Errorf("Expected custom metadata with key: %s", k)
			return
		}
	}
}
