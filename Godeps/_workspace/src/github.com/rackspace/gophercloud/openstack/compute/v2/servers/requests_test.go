package servers

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestListServers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleServerListSuccessfully(t)

	pages := 0
	err := List(client.ServiceClient(), ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := ExtractServers(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 2 {
			t.Fatalf("Expected 2 servers, got %d", len(actual))
		}
		th.CheckDeepEquals(t, ServerHerp, actual[0])
		th.CheckDeepEquals(t, ServerDerp, actual[1])

		return true, nil
	})

	th.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestListAllServers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleServerListSuccessfully(t)

	allPages, err := List(client.ServiceClient(), ListOpts{}).AllPages()
	th.AssertNoErr(t, err)
	actual, err := ExtractServers(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ServerHerp, actual[0])
	th.CheckDeepEquals(t, ServerDerp, actual[1])
}

func TestCreateServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleServerCreationSuccessfully(t, SingleServerBody)

	actual, err := Create(client.ServiceClient(), CreateOpts{
		Name:      "derp",
		ImageRef:  "f90f6034-2570-4974-8351-6b49732ef2eb",
		FlavorRef: "1",
	}).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, ServerDerp, *actual)
}

func TestDeleteServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleServerDeletionSuccessfully(t)

	res := Delete(client.ServiceClient(), "asdfasdfasdf")
	th.AssertNoErr(t, res.Err)
}

func TestGetServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleServerGetSuccessfully(t)

	client := client.ServiceClient()
	actual, err := Get(client, "1234asdf").Extract()
	if err != nil {
		t.Fatalf("Unexpected Get error: %v", err)
	}

	th.CheckDeepEquals(t, ServerDerp, *actual)
}

func TestUpdateServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleServerUpdateSuccessfully(t)

	client := client.ServiceClient()
	actual, err := Update(client, "1234asdf", UpdateOpts{Name: "new-name"}).Extract()
	if err != nil {
		t.Fatalf("Unexpected Update error: %v", err)
	}

	th.CheckDeepEquals(t, ServerDerp, *actual)
}

func TestChangeServerAdminPassword(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleAdminPasswordChangeSuccessfully(t)

	res := ChangeAdminPassword(client.ServiceClient(), "1234asdf", "new-password")
	th.AssertNoErr(t, res.Err)
}

func TestRebootServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleRebootSuccessfully(t)

	res := Reboot(client.ServiceClient(), "1234asdf", SoftReboot)
	th.AssertNoErr(t, res.Err)
}

func TestRebuildServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleRebuildSuccessfully(t, SingleServerBody)

	opts := RebuildOpts{
		Name:       "new-name",
		AdminPass:  "swordfish",
		ImageID:    "http://104.130.131.164:8774/fcad67a6189847c4aecfa3c81a05783b/images/f90f6034-2570-4974-8351-6b49732ef2eb",
		AccessIPv4: "1.2.3.4",
	}

	actual, err := Rebuild(client.ServiceClient(), "1234asdf", opts).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, ServerDerp, *actual)
}

func TestResizeServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/servers/1234asdf/action", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{ "resize": { "flavorRef": "2" } }`)

		w.WriteHeader(http.StatusAccepted)
	})

	res := Resize(client.ServiceClient(), "1234asdf", ResizeOpts{FlavorRef: "2"})
	th.AssertNoErr(t, res.Err)
}

func TestConfirmResize(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/servers/1234asdf/action", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{ "confirmResize": null }`)

		w.WriteHeader(http.StatusNoContent)
	})

	res := ConfirmResize(client.ServiceClient(), "1234asdf")
	th.AssertNoErr(t, res.Err)
}

func TestRevertResize(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/servers/1234asdf/action", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)
		th.TestJSONRequest(t, r, `{ "revertResize": null }`)

		w.WriteHeader(http.StatusAccepted)
	})

	res := RevertResize(client.ServiceClient(), "1234asdf")
	th.AssertNoErr(t, res.Err)
}

func TestRescue(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleServerRescueSuccessfully(t)

	res := Rescue(client.ServiceClient(), "1234asdf", RescueOpts{
		AdminPass: "1234567890",
	})
	th.AssertNoErr(t, res.Err)
	adminPass, _ := res.Extract()
	th.AssertEquals(t, "1234567890", adminPass)
}

func TestGetMetadatum(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleMetadatumGetSuccessfully(t)

	expected := map[string]string{"foo": "bar"}
	actual, err := Metadatum(client.ServiceClient(), "1234asdf", "foo").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestCreateMetadatum(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleMetadatumCreateSuccessfully(t)

	expected := map[string]string{"foo": "bar"}
	actual, err := CreateMetadatum(client.ServiceClient(), "1234asdf", MetadatumOpts{"foo": "bar"}).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestDeleteMetadatum(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleMetadatumDeleteSuccessfully(t)

	err := DeleteMetadatum(client.ServiceClient(), "1234asdf", "foo").ExtractErr()
	th.AssertNoErr(t, err)
}

func TestGetMetadata(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleMetadataGetSuccessfully(t)

	expected := map[string]string{"foo": "bar", "this": "that"}
	actual, err := Metadata(client.ServiceClient(), "1234asdf").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestResetMetadata(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleMetadataResetSuccessfully(t)

	expected := map[string]string{"foo": "bar", "this": "that"}
	actual, err := ResetMetadata(client.ServiceClient(), "1234asdf", MetadataOpts{
		"foo":  "bar",
		"this": "that",
	}).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestUpdateMetadata(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleMetadataUpdateSuccessfully(t)

	expected := map[string]string{"foo": "baz", "this": "those"}
	actual, err := UpdateMetadata(client.ServiceClient(), "1234asdf", MetadataOpts{
		"foo":  "baz",
		"this": "those",
	}).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestListAddresses(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleAddressListSuccessfully(t)

	expected := ListAddressesExpected
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
	HandleNetworkAddressListSuccessfully(t)

	expected := ListNetworkAddressesExpected
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

func TestCreateServerImage(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateServerImageSuccessfully(t)

	_, err := CreateImage(client.ServiceClient(), "serverimage", CreateImageOpts{Name: "test"}).ExtractImageID()
	th.AssertNoErr(t, err)
}

func TestMarshalPersonality(t *testing.T) {
	name := "/etc/test"
	contents := []byte("asdfasdf")

	personality := Personality{
		&File{
			Path:     name,
			Contents: contents,
		},
	}

	data, err := json.Marshal(personality)
	if err != nil {
		t.Fatal(err)
	}

	var actual []map[string]string
	err = json.Unmarshal(data, &actual)
	if err != nil {
		t.Fatal(err)
	}

	if len(actual) != 1 {
		t.Fatal("expected personality length 1")
	}

	if actual[0]["path"] != name {
		t.Fatal("file path incorrect")
	}

	if actual[0]["contents"] != base64.StdEncoding.EncodeToString(contents) {
		t.Fatal("file contents incorrect")
	}
}
