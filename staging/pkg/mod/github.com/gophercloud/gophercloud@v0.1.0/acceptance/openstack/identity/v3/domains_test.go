// +build acceptance

package v3

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/domains"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestDomainsList(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	var iTrue bool = true
	listOpts := domains.ListOpts{
		Enabled: &iTrue,
	}

	allPages, err := domains.List(client, listOpts).AllPages()
	th.AssertNoErr(t, err)

	allDomains, err := domains.ExtractDomains(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, domain := range allDomains {
		tools.PrintResource(t, domain)

		if domain.Name == "Default" {
			found = true
		}
	}

	th.AssertEquals(t, found, true)
}

func TestDomainsGet(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	p, err := domains.Get(client, "default").Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, p)

	th.AssertEquals(t, p.Name, "Default")
}

func TestDomainsCRUD(t *testing.T) {
	clients.RequireAdmin(t)

	client, err := clients.NewIdentityV3Client()
	th.AssertNoErr(t, err)

	var iTrue bool = true
	var description = "Testing Domain"
	createOpts := domains.CreateOpts{
		Description: description,
		Enabled:     &iTrue,
	}

	domain, err := CreateDomain(t, client, &createOpts)
	th.AssertNoErr(t, err)
	defer DeleteDomain(t, client, domain.ID)

	tools.PrintResource(t, domain)

	th.AssertEquals(t, domain.Description, description)

	var iFalse bool = false
	description = ""
	updateOpts := domains.UpdateOpts{
		Description: &description,
		Enabled:     &iFalse,
	}

	newDomain, err := domains.Update(client, domain.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, newDomain)

	th.AssertEquals(t, newDomain.Description, description)
}
