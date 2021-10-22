package testing

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/attributestags"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestReplaceAll(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks/fakeid/tags", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, attributestagsReplaceAllRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, attributestagsReplaceAllResult)
	})

	opts := attributestags.ReplaceAllOpts{
		Tags: []string{"abc", "xyz"},
	}
	res, err := attributestags.ReplaceAll(fake.ServiceClient(), "networks", "fakeid", opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, res, []string{"abc", "xyz"})
}

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks/fakeid/tags", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, attributestagsListResult)
	})

	res, err := attributestags.List(fake.ServiceClient(), "networks", "fakeid").Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, res, []string{"abc", "xyz"})
}

func TestDeleteAll(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks/fakeid/tags", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusNoContent)
	})

	err := attributestags.DeleteAll(fake.ServiceClient(), "networks", "fakeid").ExtractErr()
	th.AssertNoErr(t, err)
}

func TestAdd(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks/fakeid/tags/atag", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
	})

	err := attributestags.Add(fake.ServiceClient(), "networks", "fakeid", "atag").ExtractErr()
	th.AssertNoErr(t, err)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks/fakeid/tags/atag", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusNoContent)
	})

	err := attributestags.Delete(fake.ServiceClient(), "networks", "fakeid", "atag").ExtractErr()
	th.AssertNoErr(t, err)
}

func TestConfirmTrue(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks/fakeid/tags/atag", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusNoContent)
	})

	exists, err := attributestags.Confirm(fake.ServiceClient(), "networks", "fakeid", "atag").Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, true, exists)
}

func TestConfirmFalse(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/networks/fakeid/tags/atag", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
	})

	exists, _ := attributestags.Confirm(fake.ServiceClient(), "networks", "fakeid", "atag").Extract()
	th.AssertEquals(t, false, exists)
}
