package testing

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/addressscopes"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/address-scopes", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, AddressScopesListResult)
	})

	count := 0

	addressscopes.List(fake.ServiceClient(), addressscopes.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := addressscopes.ExtractAddressScopes(page)
		if err != nil {
			t.Errorf("Failed to extract addressscopes: %v", err)
			return false, nil
		}

		expected := []addressscopes.AddressScope{
			AddressScope1,
			AddressScope2,
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/address-scopes/9cc35860-522a-4d35-974d-51d4b011801e", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, AddressScopesGetResult)
	})

	s, err := addressscopes.Get(fake.ServiceClient(), "9cc35860-522a-4d35-974d-51d4b011801e").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.ID, "9cc35860-522a-4d35-974d-51d4b011801e")
	th.AssertEquals(t, s.Name, "scopev4")
	th.AssertEquals(t, s.TenantID, "4a9807b773404e979b19633f38370643")
	th.AssertEquals(t, s.ProjectID, "4a9807b773404e979b19633f38370643")
	th.AssertEquals(t, s.IPVersion, 4)
	th.AssertEquals(t, s.Shared, false)
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/address-scopes", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, AddressScopeCreateRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, AddressScopeCreateResult)
	})

	opts := addressscopes.CreateOpts{
		IPVersion: 4,
		Shared:    true,
		Name:      "test0",
	}
	s, err := addressscopes.Create(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "test0")
	th.AssertEquals(t, s.Shared, true)
	th.AssertEquals(t, s.IPVersion, 4)
	th.AssertEquals(t, s.TenantID, "4a9807b773404e979b19633f38370643")
	th.AssertEquals(t, s.ProjectID, "4a9807b773404e979b19633f38370643")
	th.AssertEquals(t, s.ID, "9cc35860-522a-4d35-974d-51d4b011801e")
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/address-scopes/9cc35860-522a-4d35-974d-51d4b011801e", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, AddressScopeUpdateRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, AddressScopeUpdateResult)
	})

	shared := true
	newName := "test1"
	updateOpts := addressscopes.UpdateOpts{
		Name:   &newName,
		Shared: &shared,
	}
	s, err := addressscopes.Update(fake.ServiceClient(), "9cc35860-522a-4d35-974d-51d4b011801e", updateOpts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Name, "test1")
	th.AssertEquals(t, s.Shared, true)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/address-scopes/9cc35860-522a-4d35-974d-51d4b011801e", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := addressscopes.Delete(fake.ServiceClient(), "9cc35860-522a-4d35-974d-51d4b011801e")
	th.AssertNoErr(t, res.Err)
}
