// +build acceptance identity

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack"
	"github.com/gophercloud/gophercloud/openstack/identity/v2/tokens"
)

func TestTokenAuthenticate(t *testing.T) {
	client, err := clients.NewIdentityV2UnauthenticatedClient()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	authOptions, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		t.Fatalf("Unable to obtain authentication options: %v", err)
	}

	result := tokens.Create(client, authOptions)
	token, err := result.ExtractToken()
	if err != nil {
		t.Fatalf("Unable to extract token: %v", err)
	}

	PrintToken(t, token)

	catalog, err := result.ExtractServiceCatalog()
	if err != nil {
		t.Fatalf("Unable to extract service catalog: %v", err)
	}

	for _, entry := range catalog.Entries {
		PrintCatalogEntry(t, &entry)
	}
}

func TestTokenValidate(t *testing.T) {
	client, err := clients.NewIdentityV2Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	authOptions, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		t.Fatalf("Unable to obtain authentication options: %v", err)
	}

	result := tokens.Create(client, authOptions)
	token, err := result.ExtractToken()
	if err != nil {
		t.Fatalf("Unable to extract token: %v", err)
	}

	PrintToken(t, token)

	getResult := tokens.Get(client, token.ID)
	user, err := getResult.ExtractUser()
	if err != nil {
		t.Fatalf("Unable to extract user: %v", err)
	}

	PrintTokenUser(t, user)
}
