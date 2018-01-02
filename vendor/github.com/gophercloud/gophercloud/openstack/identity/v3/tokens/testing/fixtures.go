package testing

import (
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/tokens"
	"github.com/gophercloud/gophercloud/testhelper"
)

const testTokenID = "130f6c17-420e-4a0b-97b0-0c9cf2a05f30"

// TokenOutput is a sample response to a Token call.
const TokenOutput = `
{
   "token":{
      "is_domain":false,
      "methods":[
         "password"
      ],
      "roles":[
         {
            "id":"434426788d5a451faf763b0e6db5aefb",
            "name":"admin"
         }
      ],
      "expires_at":"2017-06-03T02:19:49.000000Z",
      "project":{
         "domain":{
            "id":"default",
            "name":"Default"
         },
         "id":"a99e9b4e620e4db09a2dfb6e42a01e66",
         "name":"admin"
      },
      "catalog":[
         {
            "endpoints":[
               {
                  "url":"http://127.0.0.1:8774/v2.1/a99e9b4e620e4db09a2dfb6e42a01e66",
                  "interface":"admin",
                  "region":"RegionOne",
                  "region_id":"RegionOne",
                  "id":"3eac9e7588eb4eb2a4650cf5e079505f"
               },
               {
                  "url":"http://127.0.0.1:8774/v2.1/a99e9b4e620e4db09a2dfb6e42a01e66",
                  "interface":"internal",
                  "region":"RegionOne",
                  "region_id":"RegionOne",
                  "id":"6b33fabc69c34ea782a3f6282582b59f"
               },
               {
                  "url":"http://127.0.0.1:8774/v2.1/a99e9b4e620e4db09a2dfb6e42a01e66",
                  "interface":"public",
                  "region":"RegionOne",
                  "region_id":"RegionOne",
                  "id":"dae63c71bee24070a71f5425e7a916b5"
               }
            ],
            "type":"compute",
            "id":"17e0fa04647d4155a7933ee624dd66da",
            "name":"nova"
         },
         {
            "endpoints":[
               {
                  "url":"http://127.0.0.1:35357/v3",
                  "interface":"admin",
                  "region":"RegionOne",
                  "region_id":"RegionOne",
                  "id":"0539aeff80954a0bb756cec496768d3d"
               },
               {
                  "url":"http://127.0.0.1:5000/v3",
                  "interface":"public",
                  "region":"RegionOne",
                  "region_id":"RegionOne",
                  "id":"15bdf2d0853e4c939993d29548b1b56f"
               },
               {
                  "url":"http://127.0.0.1:5000/v3",
                  "interface":"internal",
                  "region":"RegionOne",
                  "region_id":"RegionOne",
                  "id":"3b4423c54ba343c58226bc424cb11c4b"
               }
            ],
            "type":"identity",
            "id":"1cde0ea8cb3c49d8928cb172ca825ca5",
            "name":"keystone"
         }
      ],
      "user":{
         "domain":{
            "id":"default",
            "name":"Default"
         },
         "password_expires_at":null,
         "name":"admin",
         "id":"0fe36e73809d46aeae6705c39077b1b3"
      },
      "audit_ids":[
         "ysSI0bEWR0Gmrp4LHL9LFw"
      ],
      "issued_at":"2017-06-03T01:19:49.000000Z"
   }
}`

var expectedTokenTime, _ = time.Parse(gophercloud.RFC3339Milli,
	"2017-06-03T02:19:49.000000Z")
var ExpectedToken = tokens.Token{
	ID:        testTokenID,
	ExpiresAt: expectedTokenTime,
}

var catalogEntry1 = tokens.CatalogEntry{
	ID:   "17e0fa04647d4155a7933ee624dd66da",
	Name: "nova",
	Type: "compute",
	Endpoints: []tokens.Endpoint{
		tokens.Endpoint{
			ID:        "3eac9e7588eb4eb2a4650cf5e079505f",
			Region:    "RegionOne",
			Interface: "admin",
			URL:       "http://127.0.0.1:8774/v2.1/a99e9b4e620e4db09a2dfb6e42a01e66",
		},
		tokens.Endpoint{
			ID:        "6b33fabc69c34ea782a3f6282582b59f",
			Region:    "RegionOne",
			Interface: "internal",
			URL:       "http://127.0.0.1:8774/v2.1/a99e9b4e620e4db09a2dfb6e42a01e66",
		},
		tokens.Endpoint{
			ID:        "dae63c71bee24070a71f5425e7a916b5",
			Region:    "RegionOne",
			Interface: "public",
			URL:       "http://127.0.0.1:8774/v2.1/a99e9b4e620e4db09a2dfb6e42a01e66",
		},
	},
}
var catalogEntry2 = tokens.CatalogEntry{
	ID:   "1cde0ea8cb3c49d8928cb172ca825ca5",
	Name: "keystone",
	Type: "identity",
	Endpoints: []tokens.Endpoint{
		tokens.Endpoint{
			ID:        "0539aeff80954a0bb756cec496768d3d",
			Region:    "RegionOne",
			Interface: "admin",
			URL:       "http://127.0.0.1:35357/v3",
		},
		tokens.Endpoint{
			ID:        "15bdf2d0853e4c939993d29548b1b56f",
			Region:    "RegionOne",
			Interface: "public",
			URL:       "http://127.0.0.1:5000/v3",
		},
		tokens.Endpoint{
			ID:        "3b4423c54ba343c58226bc424cb11c4b",
			Region:    "RegionOne",
			Interface: "internal",
			URL:       "http://127.0.0.1:5000/v3",
		},
	},
}

// ExpectedServiceCatalog contains expected service extracted from token response.
var ExpectedServiceCatalog = tokens.ServiceCatalog{
	Entries: []tokens.CatalogEntry{catalogEntry1, catalogEntry2},
}

var domain = tokens.Domain{
	ID:   "default",
	Name: "Default",
}

// ExpectedUser contains expected user extracted from token response.
var ExpectedUser = tokens.User{
	Domain: domain,
	ID:     "0fe36e73809d46aeae6705c39077b1b3",
	Name:   "admin",
}

var role = tokens.Role{
	ID:   "434426788d5a451faf763b0e6db5aefb",
	Name: "admin",
}

// ExpectedRoles contains expected roles extracted from token response.
var ExpectedRoles = []tokens.Role{role}

// ExpectedProject contains expected project extracted from token response.
var ExpectedProject = tokens.Project{
	Domain: domain,
	ID:     "a99e9b4e620e4db09a2dfb6e42a01e66",
	Name:   "admin",
}

func getGetResult(t *testing.T) tokens.GetResult {
	result := tokens.GetResult{}
	result.Header = http.Header{
		"X-Subject-Token": []string{testTokenID},
	}
	err := json.Unmarshal([]byte(TokenOutput), &result.Body)
	testhelper.AssertNoErr(t, err)
	return result
}
