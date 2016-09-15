// +build acceptance

package v1

import (
	"strings"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/objectstorage/v1/accounts"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestAccounts(t *testing.T) {
	// Create a provider client for making the HTTP requests.
	// See common.go in this directory for more information.
	client := newClient(t)

	// Update an account's metadata.
	updateres := accounts.Update(client, accounts.UpdateOpts{Metadata: metadata})
	t.Logf("Update Account Response: %+v\n", updateres)
	updateHeaders, err := updateres.Extract()
	th.AssertNoErr(t, err)
	t.Logf("Update Account Response Headers: %+v\n", updateHeaders)

	// Defer the deletion of the metadata set above.
	defer func() {
		tempMap := make(map[string]string)
		for k := range metadata {
			tempMap[k] = ""
		}
		updateres = accounts.Update(client, accounts.UpdateOpts{Metadata: tempMap})
		th.AssertNoErr(t, updateres.Err)
	}()

	// Extract the custom metadata from the 'Get' response.
	res := accounts.Get(client, nil)

	h, err := res.Extract()
	th.AssertNoErr(t, err)
	t.Logf("Get Account Response Headers: %+v\n", h)

	am, err := res.ExtractMetadata()
	th.AssertNoErr(t, err)
	for k := range metadata {
		if am[k] != metadata[strings.Title(k)] {
			t.Errorf("Expected custom metadata with key: %s", k)
			return
		}
	}
}
