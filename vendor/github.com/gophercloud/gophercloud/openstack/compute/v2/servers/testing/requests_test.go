package testing

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListServers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleServerListSuccessfully(t)

	pages := 0
	err := servers.List(client.ServiceClient(), servers.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := servers.ExtractServers(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 3 {
			t.Fatalf("Expected 3 servers, got %d", len(actual))
		}
		th.CheckDeepEquals(t, ServerHerp, actual[0])
		th.CheckDeepEquals(t, ServerDerp, actual[1])
		th.CheckDeepEquals(t, ServerMerp, actual[2])

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

	allPages, err := servers.List(client.ServiceClient(), servers.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)
	actual, err := servers.ExtractServers(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ServerHerp, actual[0])
	th.CheckDeepEquals(t, ServerDerp, actual[1])
}

func TestCreateServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleServerCreationSuccessfully(t, SingleServerBody)

	actual, err := servers.Create(client.ServiceClient(), servers.CreateOpts{
		Name:      "derp",
		ImageRef:  "f90f6034-2570-4974-8351-6b49732ef2eb",
		FlavorRef: "1",
	}).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, ServerDerp, *actual)
}

func TestCreateServerWithCustomField(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleServerCreationWithCustomFieldSuccessfully(t, SingleServerBody)

	actual, err := servers.Create(client.ServiceClient(), CreateOptsWithCustomField{
		CreateOpts: servers.CreateOpts{
			Name:      "derp",
			ImageRef:  "f90f6034-2570-4974-8351-6b49732ef2eb",
			FlavorRef: "1",
		},
		Foo: "bar",
	}).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, ServerDerp, *actual)
}

func TestCreateServerWithMetadata(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleServerCreationWithMetadata(t, SingleServerBody)

	actual, err := servers.Create(client.ServiceClient(), servers.CreateOpts{
		Name:      "derp",
		ImageRef:  "f90f6034-2570-4974-8351-6b49732ef2eb",
		FlavorRef: "1",
		Metadata: map[string]string{
			"abc": "def",
		},
	}).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, ServerDerp, *actual)
}

func TestCreateServerWithImageNameAndFlavorName(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleServerCreationSuccessfully(t, SingleServerBody)

	actual, err := servers.Create(client.ServiceClient(), servers.CreateOpts{
		Name:          "derp",
		ImageName:     "cirros-0.3.2-x86_64-disk",
		FlavorName:    "m1.tiny",
		ServiceClient: client.ServiceClient(),
	}).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, ServerDerp, *actual)
}

func TestDeleteServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleServerDeletionSuccessfully(t)

	res := servers.Delete(client.ServiceClient(), "asdfasdfasdf")
	th.AssertNoErr(t, res.Err)
}

func TestForceDeleteServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleServerForceDeletionSuccessfully(t)

	res := servers.ForceDelete(client.ServiceClient(), "asdfasdfasdf")
	th.AssertNoErr(t, res.Err)
}

func TestGetServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleServerGetSuccessfully(t)

	client := client.ServiceClient()
	actual, err := servers.Get(client, "1234asdf").Extract()
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
	actual, err := servers.Update(client, "1234asdf", servers.UpdateOpts{Name: "new-name"}).Extract()
	if err != nil {
		t.Fatalf("Unexpected Update error: %v", err)
	}

	th.CheckDeepEquals(t, ServerDerp, *actual)
}

func TestChangeServerAdminPassword(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleAdminPasswordChangeSuccessfully(t)

	res := servers.ChangeAdminPassword(client.ServiceClient(), "1234asdf", "new-password")
	th.AssertNoErr(t, res.Err)
}

func TestGetPassword(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePasswordGetSuccessfully(t)

	res := servers.GetPassword(client.ServiceClient(), "1234asdf")
	th.AssertNoErr(t, res.Err)
}

func TestRebootServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleRebootSuccessfully(t)

	res := servers.Reboot(client.ServiceClient(), "1234asdf", &servers.RebootOpts{
		Type: servers.SoftReboot,
	})
	th.AssertNoErr(t, res.Err)
}

func TestRebuildServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleRebuildSuccessfully(t, SingleServerBody)

	opts := servers.RebuildOpts{
		Name:       "new-name",
		AdminPass:  "swordfish",
		ImageID:    "http://104.130.131.164:8774/fcad67a6189847c4aecfa3c81a05783b/images/f90f6034-2570-4974-8351-6b49732ef2eb",
		AccessIPv4: "1.2.3.4",
	}

	actual, err := servers.Rebuild(client.ServiceClient(), "1234asdf", opts).Extract()
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

	res := servers.Resize(client.ServiceClient(), "1234asdf", servers.ResizeOpts{FlavorRef: "2"})
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

	res := servers.ConfirmResize(client.ServiceClient(), "1234asdf")
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

	res := servers.RevertResize(client.ServiceClient(), "1234asdf")
	th.AssertNoErr(t, res.Err)
}

func TestRescue(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleServerRescueSuccessfully(t)

	res := servers.Rescue(client.ServiceClient(), "1234asdf", servers.RescueOpts{
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
	actual, err := servers.Metadatum(client.ServiceClient(), "1234asdf", "foo").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestCreateMetadatum(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleMetadatumCreateSuccessfully(t)

	expected := map[string]string{"foo": "bar"}
	actual, err := servers.CreateMetadatum(client.ServiceClient(), "1234asdf", servers.MetadatumOpts{"foo": "bar"}).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestDeleteMetadatum(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleMetadatumDeleteSuccessfully(t)

	err := servers.DeleteMetadatum(client.ServiceClient(), "1234asdf", "foo").ExtractErr()
	th.AssertNoErr(t, err)
}

func TestGetMetadata(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleMetadataGetSuccessfully(t)

	expected := map[string]string{"foo": "bar", "this": "that"}
	actual, err := servers.Metadata(client.ServiceClient(), "1234asdf").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, actual)
}

func TestResetMetadata(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleMetadataResetSuccessfully(t)

	expected := map[string]string{"foo": "bar", "this": "that"}
	actual, err := servers.ResetMetadata(client.ServiceClient(), "1234asdf", servers.MetadataOpts{
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
	actual, err := servers.UpdateMetadata(client.ServiceClient(), "1234asdf", servers.MetadataOpts{
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
	err := servers.ListAddresses(client.ServiceClient(), "asdfasdfasdf").EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := servers.ExtractAddresses(page)
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
	err := servers.ListAddressesByNetwork(client.ServiceClient(), "asdfasdfasdf", "public").EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := servers.ExtractNetworkAddresses(page)
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

	_, err := servers.CreateImage(client.ServiceClient(), "serverimage", servers.CreateImageOpts{Name: "test"}).ExtractImageID()
	th.AssertNoErr(t, err)
}

func TestMarshalPersonality(t *testing.T) {
	name := "/etc/test"
	contents := []byte("asdfasdf")

	personality := servers.Personality{
		&servers.File{
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
