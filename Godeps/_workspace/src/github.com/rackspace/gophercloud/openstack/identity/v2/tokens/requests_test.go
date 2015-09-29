package tokens

import (
	"fmt"
	"testing"

	"github.com/rackspace/gophercloud"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func tokenPost(t *testing.T, options gophercloud.AuthOptions, requestJSON string) CreateResult {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleTokenPost(t, requestJSON)

	return Create(client.ServiceClient(), AuthOptions{options})
}

func tokenPostErr(t *testing.T, options gophercloud.AuthOptions, expectedErr error) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleTokenPost(t, "")

	actualErr := Create(client.ServiceClient(), AuthOptions{options}).Err
	th.CheckDeepEquals(t, expectedErr, actualErr)
}

func TestCreateWithPassword(t *testing.T) {
	options := gophercloud.AuthOptions{
		Username: "me",
		Password: "swordfish",
	}

	IsSuccessful(t, tokenPost(t, options, `
    {
      "auth": {
        "passwordCredentials": {
          "username": "me",
          "password": "swordfish"
        }
      }
    }
  `))
}

func TestCreateTokenWithTenantID(t *testing.T) {
	options := gophercloud.AuthOptions{
		Username: "me",
		Password: "opensesame",
		TenantID: "fc394f2ab2df4114bde39905f800dc57",
	}

	IsSuccessful(t, tokenPost(t, options, `
    {
      "auth": {
        "tenantId": "fc394f2ab2df4114bde39905f800dc57",
        "passwordCredentials": {
          "username": "me",
          "password": "opensesame"
        }
      }
    }
  `))
}

func TestCreateTokenWithTenantName(t *testing.T) {
	options := gophercloud.AuthOptions{
		Username:   "me",
		Password:   "opensesame",
		TenantName: "demo",
	}

	IsSuccessful(t, tokenPost(t, options, `
    {
      "auth": {
        "tenantName": "demo",
        "passwordCredentials": {
          "username": "me",
          "password": "opensesame"
        }
      }
    }
  `))
}

func TestProhibitUserID(t *testing.T) {
	options := gophercloud.AuthOptions{
		Username: "me",
		UserID:   "1234",
		Password: "thing",
	}

	tokenPostErr(t, options, ErrUserIDProvided)
}

func TestProhibitAPIKey(t *testing.T) {
	options := gophercloud.AuthOptions{
		Username: "me",
		Password: "thing",
		APIKey:   "123412341234",
	}

	tokenPostErr(t, options, ErrAPIKeyProvided)
}

func TestProhibitDomainID(t *testing.T) {
	options := gophercloud.AuthOptions{
		Username: "me",
		Password: "thing",
		DomainID: "1234",
	}

	tokenPostErr(t, options, ErrDomainIDProvided)
}

func TestProhibitDomainName(t *testing.T) {
	options := gophercloud.AuthOptions{
		Username:   "me",
		Password:   "thing",
		DomainName: "wat",
	}

	tokenPostErr(t, options, ErrDomainNameProvided)
}

func TestRequireUsername(t *testing.T) {
	options := gophercloud.AuthOptions{
		Password: "thing",
	}

	tokenPostErr(t, options, fmt.Errorf("You must provide either username/password or tenantID/token values."))
}

func TestRequirePassword(t *testing.T) {
	options := gophercloud.AuthOptions{
		Username: "me",
	}

	tokenPostErr(t, options, ErrPasswordRequired)
}
