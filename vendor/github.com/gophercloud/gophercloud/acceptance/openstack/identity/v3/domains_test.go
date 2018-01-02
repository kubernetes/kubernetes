// +build acceptance

package v3

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/domains"
)

func TestDomainsList(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	var iTrue bool = true
	listOpts := domains.ListOpts{
		Enabled: &iTrue,
	}

	allPages, err := domains.List(client, listOpts).AllPages()
	if err != nil {
		t.Fatalf("Unable to list domains: %v", err)
	}

	allDomains, err := domains.ExtractDomains(allPages)
	if err != nil {
		t.Fatalf("Unable to extract domains: %v", err)
	}

	for _, domain := range allDomains {
		tools.PrintResource(t, domain)
	}
}

func TestDomainsGet(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	allPages, err := domains.List(client, nil).AllPages()
	if err != nil {
		t.Fatalf("Unable to list domains: %v", err)
	}

	allDomains, err := domains.ExtractDomains(allPages)
	if err != nil {
		t.Fatalf("Unable to extract domains: %v", err)
	}

	domain := allDomains[0]
	p, err := domains.Get(client, domain.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get domain: %v", err)
	}

	tools.PrintResource(t, p)
}

func TestDomainsCRUD(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	var iTrue bool = true
	createOpts := domains.CreateOpts{
		Description: "Testing Domain",
		Enabled:     &iTrue,
	}

	domain, err := CreateDomain(t, client, &createOpts)
	if err != nil {
		t.Fatalf("Unable to create domain: %v", err)
	}
	defer DeleteDomain(t, client, domain.ID)

	tools.PrintResource(t, domain)

	var iFalse bool = false
	updateOpts := domains.UpdateOpts{
		Description: "Staging Test Domain",
		Enabled:     &iFalse,
	}

	newDomain, err := domains.Update(client, domain.ID, updateOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update domain: %v", err)
	}

	tools.PrintResource(t, newDomain)
}
