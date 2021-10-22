package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack"
	"github.com/gophercloud/gophercloud/openstack/objectstorage/v1/swauth"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestAuth(t *testing.T) {
	authOpts := swauth.AuthOpts{
		User: "test:tester",
		Key:  "testing",
	}

	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleAuthSuccessfully(t, authOpts)

	providerClient, err := openstack.NewClient(th.Endpoint())
	th.AssertNoErr(t, err)

	swiftClient, err := swauth.NewObjectStorageV1(providerClient, authOpts)
	th.AssertNoErr(t, err)
	th.AssertEquals(t, swiftClient.TokenID, AuthResult.Token)
}

func TestBadAuth(t *testing.T) {
	authOpts := swauth.AuthOpts{}
	_, err := authOpts.ToAuthOptsMap()
	if err == nil {
		t.Fatalf("Expected an error due to missing auth options")
	}
}
