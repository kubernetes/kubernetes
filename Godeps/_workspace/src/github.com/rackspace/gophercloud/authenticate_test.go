package gophercloud

import (
	"net/http"
	"testing"
)

const SUCCESSFUL_RESPONSE = `{
	"access": {
		"serviceCatalog": [{
			"endpoints": [{
				"publicURL": "https://ord.servers.api.rackspacecloud.com/v2/12345",
				"region": "ORD",
				"tenantId": "12345",
				"versionId": "2",
				"versionInfo": "https://ord.servers.api.rackspacecloud.com/v2",
				"versionList": "https://ord.servers.api.rackspacecloud.com/"
			},{
				"publicURL": "https://dfw.servers.api.rackspacecloud.com/v2/12345",
				"region": "DFW",
				"tenantId": "12345",
				"versionId": "2",
				"versionInfo": "https://dfw.servers.api.rackspacecloud.com/v2",
				"versionList": "https://dfw.servers.api.rackspacecloud.com/"
			}],
			"name": "cloudServersOpenStack",
			"type": "compute"
		},{
			"endpoints": [{
				"publicURL": "https://ord.databases.api.rackspacecloud.com/v1.0/12345",
				"region": "ORD",
				"tenantId": "12345"
			}],
			"name": "cloudDatabases",
			"type": "rax:database"
		}],
		"token": {
			"expires": "2012-04-13T13:15:00.000-05:00",
			"id": "aaaaa-bbbbb-ccccc-dddd"
		},
		"user": {
			"RAX-AUTH:defaultRegion": "DFW",
			"id": "161418",
			"name": "demoauthor",
			"roles": [{
				"description": "User Admin Role.",
				"id": "3",
				"name": "identity:user-admin"
			}]
		}
	}
}
`

func TestAuthProvider(t *testing.T) {
	tt := newTransport().WithResponse(SUCCESSFUL_RESPONSE)
	c := TestContext().UseCustomClient(&http.Client{
		Transport: tt,
	})

	_, err := c.Authenticate("", AuthOptions{})
	if err == nil {
		t.Error("Expected error for empty provider string")
		return
	}
	_, err = c.Authenticate("unknown-provider", AuthOptions{Username: "u", Password: "p"})
	if err == nil {
		t.Error("Expected error for unknown service provider")
		return
	}

	err = c.RegisterProvider("provider", Provider{AuthEndpoint: "/"})
	if err != nil {
		t.Error(err)
		return
	}
	_, err = c.Authenticate("provider", AuthOptions{Username: "u", Password: "p"})
	if err != nil {
		t.Error(err)
		return
	}
	if tt.called != 1 {
		t.Error("Expected transport to be called once.")
		return
	}
}

func TestTenantIdEncoding(t *testing.T) {
	tt := newTransport().WithResponse(SUCCESSFUL_RESPONSE)
	c := TestContext().
		UseCustomClient(&http.Client{
		Transport: tt,
	}).
		WithProvider("provider", Provider{AuthEndpoint: "/"})

	tt.IgnoreTenantId()
	_, err := c.Authenticate("provider", AuthOptions{
		Username: "u",
		Password: "p",
	})
	if err != nil {
		t.Error(err)
		return
	}
	if tt.tenantIdFound {
		t.Error("Tenant ID should not have been encoded")
		return
	}

	tt.ExpectTenantId()
	_, err = c.Authenticate("provider", AuthOptions{
		Username: "u",
		Password: "p",
		TenantId: "t",
	})
	if err != nil {
		t.Error(err)
		return
	}
	if !tt.tenantIdFound {
		t.Error("Tenant ID should have been encoded")
		return
	}
}

func TestUserNameAndPassword(t *testing.T) {
	c := TestContext().
		WithProvider("provider", Provider{AuthEndpoint: "http://localhost/"}).
		UseCustomClient(&http.Client{Transport: newTransport().WithResponse(SUCCESSFUL_RESPONSE)})

	credentials := []AuthOptions{
		{},
		{Username: "u"},
		{Password: "p"},
	}
	for i, auth := range credentials {
		_, err := c.Authenticate("provider", auth)
		if err == nil {
			t.Error("Expected error from missing credentials (%d)", i)
			return
		}
	}

	_, err := c.Authenticate("provider", AuthOptions{Username: "u", Password: "p"})
	if err != nil {
		t.Error(err)
		return
	}
}

func TestUserNameAndApiKey(t *testing.T) {
	c := TestContext().
		WithProvider("provider", Provider{AuthEndpoint: "http://localhost/"}).
		UseCustomClient(&http.Client{Transport: newTransport().WithResponse(SUCCESSFUL_RESPONSE)})

	credentials := []AuthOptions{
		{},
		{Username: "u"},
		{ApiKey: "a"},
	}
	for i, auth := range credentials {
		_, err := c.Authenticate("provider", auth)
		if err == nil {
			t.Error("Expected error from missing credentials (%d)", i)
			return
		}
	}

	_, err := c.Authenticate("provider", AuthOptions{Username: "u", ApiKey: "a"})
	if err != nil {
		t.Error(err)
		return
	}
}

func TestTokenAcquisition(t *testing.T) {
	c := TestContext().
		UseCustomClient(&http.Client{Transport: newTransport().WithResponse(SUCCESSFUL_RESPONSE)}).
		WithProvider("provider", Provider{AuthEndpoint: "http://localhost/"})

	acc, err := c.Authenticate("provider", AuthOptions{Username: "u", Password: "p"})
	if err != nil {
		t.Error(err)
		return
	}

	tok := acc.Token
	if (tok.Id == "") || (tok.Expires == "") {
		t.Error("Expected a valid token for successful login; got %s, %s", tok.Id, tok.Expires)
		return
	}
}

func TestServiceCatalogAcquisition(t *testing.T) {
	c := TestContext().
		UseCustomClient(&http.Client{Transport: newTransport().WithResponse(SUCCESSFUL_RESPONSE)}).
		WithProvider("provider", Provider{AuthEndpoint: "http://localhost/"})

	acc, err := c.Authenticate("provider", AuthOptions{Username: "u", Password: "p"})
	if err != nil {
		t.Error(err)
		return
	}

	svcs := acc.ServiceCatalog
	if len(svcs) < 2 {
		t.Error("Expected 2 service catalog entries; got %d", len(svcs))
		return
	}

	types := map[string]bool{
		"compute":      true,
		"rax:database": true,
	}
	for _, entry := range svcs {
		if !types[entry.Type] {
			t.Error("Expected to find type %s.", entry.Type)
			return
		}
	}
}

func TestUserAcquisition(t *testing.T) {
	c := TestContext().
		UseCustomClient(&http.Client{Transport: newTransport().WithResponse(SUCCESSFUL_RESPONSE)}).
		WithProvider("provider", Provider{AuthEndpoint: "http://localhost/"})

	acc, err := c.Authenticate("provider", AuthOptions{Username: "u", Password: "p"})
	if err != nil {
		t.Error(err)
		return
	}

	u := acc.User
	if u.Id != "161418" {
		t.Error("Expected user ID of 16148; got", u.Id)
		return
	}
}

func TestAuthenticationNeverReauths(t *testing.T) {
	tt := newTransport().WithError(401)
	c := TestContext().
		UseCustomClient(&http.Client{Transport: tt}).
		WithProvider("provider", Provider{AuthEndpoint: "http://localhost"})

	_, err := c.Authenticate("provider", AuthOptions{Username: "u", Password: "p"})
	if err == nil {
		t.Error("Expected an error from a 401 Unauthorized response")
		return
	}

	rc, _ := ActualResponseCode(err)
	if rc != 401 {
		t.Error("Expected a 401 error code")
		return
	}

	err = tt.VerifyCalls(t, 1)
	if err != nil {
		// Test object already flagged.
		return
	}
}
