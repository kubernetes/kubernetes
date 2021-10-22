// +build acceptance

package openstack

import (
	"os"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"
)

func TestAuthenticatedClient(t *testing.T) {
	// Obtain credentials from the environment.
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		t.Fatalf("Unable to acquire credentials: %v", err)
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		t.Fatalf("Unable to authenticate: %v", err)
	}

	if client.TokenID == "" {
		t.Errorf("No token ID assigned to the client")
	}

	t.Logf("Client successfully acquired a token: %v", client.TokenID)

	// Find the storage service in the service catalog.
	storage, err := openstack.NewObjectStorageV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
	if err != nil {
		t.Errorf("Unable to locate a storage service: %v", err)
	} else {
		t.Logf("Located a storage service at endpoint: [%s]", storage.Endpoint)
	}
}

func TestReauth(t *testing.T) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		t.Fatalf("Unable to obtain environment auth options: %v", err)
	}

	// Allow reauth
	ao.AllowReauth = true

	provider, err := openstack.NewClient(ao.IdentityEndpoint)
	if err != nil {
		t.Fatalf("Unable to create provider: %v", err)
	}

	err = openstack.Authenticate(provider, ao)
	if err != nil {
		t.Fatalf("Unable to authenticate: %v", err)
	}

	t.Logf("Creating a compute client")
	_, err = openstack.NewComputeV2(provider, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
	if err != nil {
		t.Fatalf("Unable to create compute client: %v", err)
	}

	t.Logf("Sleeping for 1 second")
	time.Sleep(1 * time.Second)
	t.Logf("Attempting to reauthenticate")

	err = provider.ReauthFunc()
	if err != nil {
		t.Fatalf("Unable to reauthenticate: %v", err)
	}

	t.Logf("Creating a compute client")
	_, err = openstack.NewComputeV2(provider, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
	if err != nil {
		t.Fatalf("Unable to create compute client: %v", err)
	}
}
