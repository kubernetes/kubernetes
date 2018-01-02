// +build acceptance

package v3

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/tokens"
)

func TestGetToken(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v")
	}

	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		t.Fatalf("Unable to obtain environment auth options: %v", err)
	}

	authOptions := tokens.AuthOptions{
		Username:   ao.Username,
		Password:   ao.Password,
		DomainName: "default",
	}

	token, err := tokens.Create(client, &authOptions).Extract()
	if err != nil {
		t.Fatalf("Unable to get token: %v", err)
	}
	tools.PrintResource(t, token)

	catalog, err := tokens.Get(client, token.ID).ExtractServiceCatalog()
	if err != nil {
		t.Fatalf("Unable to get catalog from token: %v", err)
	}
	tools.PrintResource(t, catalog)

	user, err := tokens.Get(client, token.ID).ExtractUser()
	if err != nil {
		t.Fatalf("Unable to get user from token: %v", err)
	}
	tools.PrintResource(t, user)

	roles, err := tokens.Get(client, token.ID).ExtractRoles()
	if err != nil {
		t.Fatalf("Unable to get roles from token: %v", err)
	}
	tools.PrintResource(t, roles)

	project, err := tokens.Get(client, token.ID).ExtractProject()
	if err != nil {
		t.Fatalf("Unable to get project from token: %v", err)
	}
	tools.PrintResource(t, project)
}
