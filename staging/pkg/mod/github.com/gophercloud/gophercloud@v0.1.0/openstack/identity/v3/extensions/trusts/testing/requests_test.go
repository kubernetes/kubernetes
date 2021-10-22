package testing

import (
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/extensions/trusts"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/tokens"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestCreateUserIDPasswordTrustID(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	ao := trusts.AuthOptsExt{
		TrustID: "de0945a",
		AuthOptionsBuilder: &tokens.AuthOptions{
			UserID:   "me",
			Password: "squirrel!",
		},
	}
	HandleCreateTokenWithTrustID(t, ao, `
		{
			"auth": {
				"identity": {
					"methods": ["password"],
					"password": {
						"user": { "id": "me", "password": "squirrel!" }
					}
				},
        "scope": {
            "OS-TRUST:trust": {
                "id": "de0945a"
            }
        }
			}
		}
	`)

	var actual struct {
		tokens.Token
		trusts.TokenExt
	}
	err := tokens.Create(client.ServiceClient(), ao).ExtractInto(&actual)
	if err != nil {
		t.Errorf("Create returned an error: %v", err)
	}
	expected := struct {
		tokens.Token
		trusts.TokenExt
	}{
		tokens.Token{
			ExpiresAt: time.Date(2013, 02, 27, 18, 30, 59, 999999000, time.UTC),
		},
		trusts.TokenExt{
			Trust: trusts.Trust{
				ID:            "fe0aef",
				Impersonation: false,
				TrusteeUser: trusts.TrusteeUser{
					ID: "0ca8f6",
				},
				TrustorUser: trusts.TrustorUser{
					ID: "bd263c",
				},
				RedelegatedTrustID: "3ba234",
				RedelegationCount:  2,
			},
		},
	}

	th.AssertDeepEquals(t, expected, actual)
}
