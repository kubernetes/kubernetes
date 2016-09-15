package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"
	th "github.com/gophercloud/gophercloud/testhelper"
)

const ID = "0123456789"

func TestAuthenticatedClientV3(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `
			{
				"versions": {
					"values": [
						{
							"status": "stable",
							"id": "v3.0",
							"links": [
								{ "href": "%s", "rel": "self" }
							]
						},
						{
							"status": "stable",
							"id": "v2.0",
							"links": [
								{ "href": "%s", "rel": "self" }
							]
						}
					]
				}
			}
		`, th.Endpoint()+"v3/", th.Endpoint()+"v2.0/")
	})

	th.Mux.HandleFunc("/v3/auth/tokens", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("X-Subject-Token", ID)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, `{ "token": { "expires_at": "2013-02-02T18:30:59.000000Z" } }`)
	})

	options := gophercloud.AuthOptions{
		Username:         "me",
		Password:         "secret",
		DomainName:       "default",
		TenantName:       "project",
		IdentityEndpoint: th.Endpoint(),
	}
	client, err := openstack.AuthenticatedClient(options)
	th.AssertNoErr(t, err)
	th.CheckEquals(t, ID, client.TokenID)
}

func TestAuthenticatedClientV2(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `
			{
				"versions": {
					"values": [
						{
							"status": "experimental",
							"id": "v3.0",
							"links": [
								{ "href": "%s", "rel": "self" }
							]
						},
						{
							"status": "stable",
							"id": "v2.0",
							"links": [
								{ "href": "%s", "rel": "self" }
							]
						}
					]
				}
			}
		`, th.Endpoint()+"v3/", th.Endpoint()+"v2.0/")
	})

	th.Mux.HandleFunc("/v2.0/tokens", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `
			{
				"access": {
					"token": {
						"id": "01234567890",
						"expires": "2014-10-01T10:00:00.000000Z"
					},
					"serviceCatalog": [
						{
							"name": "Cloud Servers",
							"type": "compute",
							"endpoints": [
								{
									"tenantId": "t1000",
									"publicURL": "https://compute.north.host.com/v1/t1000",
									"internalURL": "https://compute.north.internal/v1/t1000",
									"region": "North",
									"versionId": "1",
									"versionInfo": "https://compute.north.host.com/v1/",
									"versionList": "https://compute.north.host.com/"
								},
								{
									"tenantId": "t1000",
									"publicURL": "https://compute.north.host.com/v1.1/t1000",
									"internalURL": "https://compute.north.internal/v1.1/t1000",
									"region": "North",
									"versionId": "1.1",
									"versionInfo": "https://compute.north.host.com/v1.1/",
									"versionList": "https://compute.north.host.com/"
								}
							],
							"endpoints_links": []
						},
						{
							"name": "Cloud Files",
							"type": "object-store",
							"endpoints": [
								{
									"tenantId": "t1000",
									"publicURL": "https://storage.north.host.com/v1/t1000",
									"internalURL": "https://storage.north.internal/v1/t1000",
									"region": "North",
									"versionId": "1",
									"versionInfo": "https://storage.north.host.com/v1/",
									"versionList": "https://storage.north.host.com/"
								},
								{
									"tenantId": "t1000",
									"publicURL": "https://storage.south.host.com/v1/t1000",
									"internalURL": "https://storage.south.internal/v1/t1000",
									"region": "South",
									"versionId": "1",
									"versionInfo": "https://storage.south.host.com/v1/",
									"versionList": "https://storage.south.host.com/"
								}
							]
						}
					]
				}
			}
		`)
	})

	options := gophercloud.AuthOptions{
		Username:         "me",
		Password:         "secret",
		IdentityEndpoint: th.Endpoint(),
	}
	client, err := openstack.AuthenticatedClient(options)
	th.AssertNoErr(t, err)
	th.CheckEquals(t, "01234567890", client.TokenID)
}

func TestIdentityAdminV3Client(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `
			{
				"versions": {
					"values": [
						{
							"status": "stable",
							"id": "v3.0",
							"links": [
								{ "href": "%s", "rel": "self" }
							]
						},
						{
							"status": "stable",
							"id": "v2.0",
							"links": [
								{ "href": "%s", "rel": "self" }
							]
						}
					]
				}
			}
		`, th.Endpoint()+"v3/", th.Endpoint()+"v2.0/")
	})

	th.Mux.HandleFunc("/v3/auth/tokens", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("X-Subject-Token", ID)

		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, `
	{
    "token": {
        "audit_ids": ["VcxU2JYqT8OzfUVvrjEITQ", "qNUTIJntTzO1-XUk5STybw"],
        "catalog": [
            {
                "endpoints": [
                    {
                        "id": "39dc322ce86c4111b4f06c2eeae0841b",
                        "interface": "public",
                        "region": "RegionOne",
                        "url": "http://localhost:5000"
                    },
                    {
                        "id": "ec642f27474842e78bf059f6c48f4e99",
                        "interface": "internal",
                        "region": "RegionOne",
                        "url": "http://localhost:5000"
                    },
                    {
                        "id": "c609fc430175452290b62a4242e8a7e8",
                        "interface": "admin",
                        "region": "RegionOne",
                        "url": "http://localhost:35357"
                    }
                ],
                "id": "4363ae44bdf34a3981fde3b823cb9aa2",
                "type": "identity",
                "name": "keystone"
            }
        ],
        "expires_at": "2013-02-27T18:30:59.999999Z",
        "is_domain": false,
        "issued_at": "2013-02-27T16:30:59.999999Z",
        "methods": [
            "password"
        ],
        "project": {
            "domain": {
                "id": "1789d1",
                "name": "example.com"
            },
            "id": "263fd9",
            "name": "project-x"
        },
        "roles": [
            {
                "id": "76e72a",
                "name": "admin"
            },
            {
                "id": "f4f392",
                "name": "member"
            }
        ],
        "service_providers": [
            {
                "auth_url":"https://example.com:5000/v3/OS-FEDERATION/identity_providers/acme/protocols/saml2/auth",
                "id": "sp1",
                "sp_url": "https://example.com:5000/Shibboleth.sso/SAML2/ECP"
            },
            {
                "auth_url":"https://other.example.com:5000/v3/OS-FEDERATION/identity_providers/acme/protocols/saml2/auth",
                "id": "sp2",
                "sp_url": "https://other.example.com:5000/Shibboleth.sso/SAML2/ECP"
            }
        ],
        "user": {
            "domain": {
                "id": "1789d1",
                "name": "example.com"
            },
            "id": "0ca8f6",
            "name": "Joe",
            "password_expires_at": "2016-11-06T15:32:17.000000"
        }
    }
}
	`)
	})

	options := gophercloud.AuthOptions{
		Username:         "me",
		Password:         "secret",
		DomainID:         "12345",
		IdentityEndpoint: th.Endpoint(),
	}
	pc, err := openstack.AuthenticatedClient(options)
	th.AssertNoErr(t, err)
	sc, err := openstack.NewIdentityV3(pc, gophercloud.EndpointOpts{
		Availability: gophercloud.AvailabilityAdmin,
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, "http://localhost:35357/", sc.Endpoint)
}

func testAuthenticatedClientFails(t *testing.T, endpoint string) {
	options := gophercloud.AuthOptions{
		Username:         "me",
		Password:         "secret",
		DomainName:       "default",
		TenantName:       "project",
		IdentityEndpoint: endpoint,
	}
	_, err := openstack.AuthenticatedClient(options)
	if err == nil {
		t.Fatal("expected error but call succeeded")
	}
}

func TestAuthenticatedClientV3Fails(t *testing.T) {
	testAuthenticatedClientFails(t, "http://bad-address.example.com/v3")
}

func TestAuthenticatedClientV2Fails(t *testing.T) {
	testAuthenticatedClientFails(t, "http://bad-address.example.com/v2.0")
}
