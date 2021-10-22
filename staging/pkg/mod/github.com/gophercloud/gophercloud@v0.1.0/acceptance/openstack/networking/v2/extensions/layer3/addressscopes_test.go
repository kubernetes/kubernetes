// +build acceptance networking layer3 addressscopes

package layer3

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/addressscopes"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestAddressScopesCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create an address-scope
	addressScope, err := CreateAddressScope(t, client)
	th.AssertNoErr(t, err)
	defer DeleteAddressScope(t, client, addressScope.ID)

	tools.PrintResource(t, addressScope)

	newName := tools.RandomString("TESTACC-", 8)
	updateOpts := &addressscopes.UpdateOpts{
		Name: &newName,
	}

	_, err = addressscopes.Update(client, addressScope.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	newAddressScope, err := addressscopes.Get(client, addressScope.ID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newAddressScope)
	th.AssertEquals(t, newAddressScope.Name, newName)

	allPages, err := addressscopes.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allAddressScopes, err := addressscopes.ExtractAddressScopes(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, addressScope := range allAddressScopes {
		if addressScope.ID == newAddressScope.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}
