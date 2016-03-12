package servers

import (
	"fmt"
	"net/http"
	"testing"

	os "github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestListServers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/servers/detail", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, ListOutput)
	})

	count := 0
	err := List(client.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractServers(page)
		th.AssertNoErr(t, err)
		th.CheckDeepEquals(t, ExpectedServerSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, count)
}

func TestCreateServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleServerCreationSuccessfully(t, CreateOutput)

	actual, err := Create(client.ServiceClient(), os.CreateOpts{
		Name:      "derp",
		ImageRef:  "f90f6034-2570-4974-8351-6b49732ef2eb",
		FlavorRef: "1",
	}).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, &CreatedServer, actual)
}

func TestDeleteServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleServerDeletionSuccessfully(t)

	res := Delete(client.ServiceClient(), "asdfasdfasdf")
	th.AssertNoErr(t, res.Err)
}

func TestGetServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/servers/8c65cb68-0681-4c30-bc88-6b83a8a26aee", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, GetOutput)
	})

	actual, err := Get(client.ServiceClient(), "8c65cb68-0681-4c30-bc88-6b83a8a26aee").Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &GophercloudServer, actual)
}

func TestUpdateServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/servers/8c65cb68-0681-4c30-bc88-6b83a8a26aee", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{ "server": { "name": "test-server-updated" } }`)

		w.Header().Add("Content-Type", "application/json")

		fmt.Fprintf(w, UpdateOutput)
	})

	opts := os.UpdateOpts{
		Name: "test-server-updated",
	}
	actual, err := Update(client.ServiceClient(), "8c65cb68-0681-4c30-bc88-6b83a8a26aee", opts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &GophercloudUpdatedServer, actual)
}

func TestChangeAdminPassword(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleAdminPasswordChangeSuccessfully(t)

	res := ChangeAdminPassword(client.ServiceClient(), "1234asdf", "new-password")
	th.AssertNoErr(t, res.Err)
}

func TestReboot(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleRebootSuccessfully(t)

	res := Reboot(client.ServiceClient(), "1234asdf", os.SoftReboot)
	th.AssertNoErr(t, res.Err)
}

func TestRebuildServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleRebuildSuccessfully(t, GetOutput)

	opts := os.RebuildOpts{
		Name:       "new-name",
		AdminPass:  "swordfish",
		ImageID:    "http://104.130.131.164:8774/fcad67a6189847c4aecfa3c81a05783b/images/f90f6034-2570-4974-8351-6b49732ef2eb",
		AccessIPv4: "1.2.3.4",
	}
	actual, err := Rebuild(client.ServiceClient(), "1234asdf", opts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &GophercloudServer, actual)
}

func TestListAddresses(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleAddressListSuccessfully(t)

	expected := os.ListAddressesExpected
	pages := 0
	err := ListAddresses(client.ServiceClient(), "asdfasdfasdf").EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := ExtractAddresses(page)
		th.AssertNoErr(t, err)

		if len(actual) != 2 {
			t.Fatalf("Expected 2 networks, got %d", len(actual))
		}
		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, pages)
}

func TestListAddressesByNetwork(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleNetworkAddressListSuccessfully(t)

	expected := os.ListNetworkAddressesExpected
	pages := 0
	err := ListAddressesByNetwork(client.ServiceClient(), "asdfasdfasdf", "public").EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := ExtractNetworkAddresses(page)
		th.AssertNoErr(t, err)

		if len(actual) != 2 {
			t.Fatalf("Expected 2 addresses, got %d", len(actual))
		}
		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, pages)
}
