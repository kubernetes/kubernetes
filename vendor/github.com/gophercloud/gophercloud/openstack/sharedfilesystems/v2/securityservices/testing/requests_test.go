package testing

import (
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/securityservices"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// Verifies that a security service can be created correctly
func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockCreateResponse(t)

	options := &securityservices.CreateOpts{
		Name:        "SecServ1",
		Description: "Creating my first Security Service",
		DNSIP:       "10.0.0.0/24",
		User:        "demo",
		Password:    "***",
		Type:        "kerberos",
	}

	s, err := securityservices.Create(client.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "SecServ1")
	th.AssertEquals(t, s.Description, "Creating my first Security Service")
	th.AssertEquals(t, s.User, "demo")
	th.AssertEquals(t, s.DNSIP, "10.0.0.0/24")
	th.AssertEquals(t, s.Password, "supersecret")
	th.AssertEquals(t, s.Type, "kerberos")
}

// Verifies that a security service cannot be created without a type
func TestCreateFails(t *testing.T) {
	options := &securityservices.CreateOpts{
		Name:        "SecServ1",
		Description: "Creating my first Security Service",
		DNSIP:       "10.0.0.0/24",
		User:        "demo",
		Password:    "***",
	}

	_, err := securityservices.Create(client.ServiceClient(), options).Extract()
	if _, ok := err.(gophercloud.ErrMissingInput); !ok {
		t.Fatal("ErrMissingInput was expected to occur")
	}
}

// Verifies that security service deletion works
func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockDeleteResponse(t)

	res := securityservices.Delete(client.ServiceClient(), "securityServiceID")
	th.AssertNoErr(t, res.Err)
}

// Verifies that security services can be listed correctly
func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockListResponse(t)

	allPages, err := securityservices.List(client.ServiceClient(), &securityservices.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)
	actual, err := securityservices.ExtractSecurityServices(allPages)
	th.AssertNoErr(t, err)
	var nilTime time.Time
	expected := []securityservices.SecurityService{
		{
			Status:      "new",
			Domain:      "",
			ProjectID:   "16e1ab15c35a457e9c2b2aa189f544e1",
			Name:        "SecServ1",
			CreatedAt:   time.Date(2015, 9, 7, 12, 19, 10, 0, time.UTC),
			Description: "Creating my first Security Service",
			UpdatedAt:   nilTime,
			Server:      "",
			DNSIP:       "10.0.0.0/24",
			User:        "demo",
			Password:    "supersecret",
			Type:        "kerberos",
			ID:          "3c829734-0679-4c17-9637-801da48c0d5f",
		},
		{
			Status:      "new",
			Domain:      "",
			ProjectID:   "16e1ab15c35a457e9c2b2aa189f544e1",
			Name:        "SecServ2",
			CreatedAt:   time.Date(2015, 9, 7, 12, 25, 03, 0, time.UTC),
			Description: "Creating my second Security Service",
			UpdatedAt:   nilTime,
			Server:      "",
			DNSIP:       "10.0.0.0/24",
			User:        "",
			Password:    "",
			Type:        "ldap",
			ID:          "5a1d3a12-34a7-4087-8983-50e9ed03509a",
		},
	}

	th.CheckDeepEquals(t, expected, actual)
}

// Verifies that security services list can be called with query parameters
func TestFilteredList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockFilteredListResponse(t)

	options := &securityservices.ListOpts{
		Type: "kerberos",
	}

	allPages, err := securityservices.List(client.ServiceClient(), options).AllPages()
	th.AssertNoErr(t, err)
	actual, err := securityservices.ExtractSecurityServices(allPages)
	th.AssertNoErr(t, err)
	var nilTime time.Time
	expected := []securityservices.SecurityService{
		{
			Status:      "new",
			Domain:      "",
			ProjectID:   "16e1ab15c35a457e9c2b2aa189f544e1",
			Name:        "SecServ1",
			CreatedAt:   time.Date(2015, 9, 7, 12, 19, 10, 0, time.UTC),
			Description: "Creating my first Security Service",
			UpdatedAt:   nilTime,
			Server:      "",
			DNSIP:       "10.0.0.0/24",
			User:        "demo",
			Password:    "supersecret",
			Type:        "kerberos",
			ID:          "3c829734-0679-4c17-9637-801da48c0d5f",
		},
	}

	th.CheckDeepEquals(t, expected, actual)
}

// Verifies that it is possible to get a security service
func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockGetResponse(t)

	var nilTime time.Time
	expected := securityservices.SecurityService{
		ID:          "3c829734-0679-4c17-9637-801da48c0d5f",
		Name:        "SecServ1",
		CreatedAt:   time.Date(2015, 9, 7, 12, 19, 10, 0, time.UTC),
		Description: "Creating my first Security Service",
		Type:        "kerberos",
		UpdatedAt:   nilTime,
		ProjectID:   "16e1ab15c35a457e9c2b2aa189f544e1",
		Status:      "new",
		Domain:      "",
		Server:      "",
		DNSIP:       "10.0.0.0/24",
		User:        "demo",
		Password:    "supersecret",
	}

	n, err := securityservices.Get(client.ServiceClient(), "3c829734-0679-4c17-9637-801da48c0d5f").Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, &expected, n)
}

// Verifies that it is possible to update a security service
func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	MockUpdateResponse(t)
	expected := securityservices.SecurityService{
		ID:          "securityServiceID",
		Name:        "SecServ2",
		CreatedAt:   time.Date(2015, 9, 7, 12, 19, 10, 0, time.UTC),
		Description: "Updating my first Security Service",
		Type:        "kerberos",
		UpdatedAt:   time.Date(2015, 9, 7, 12, 20, 10, 0, time.UTC),
		ProjectID:   "16e1ab15c35a457e9c2b2aa189f544e1",
		Status:      "new",
		Domain:      "",
		Server:      "",
		DNSIP:       "10.0.0.0/24",
		User:        "demo",
		Password:    "supersecret",
	}

	options := securityservices.UpdateOpts{Name: "SecServ2"}
	s, err := securityservices.Update(client.ServiceClient(), "securityServiceID", options).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &expected, s)
}
