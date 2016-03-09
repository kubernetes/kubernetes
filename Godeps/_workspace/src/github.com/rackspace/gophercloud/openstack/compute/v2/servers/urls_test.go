package servers

import (
	"testing"

	"github.com/rackspace/gophercloud"
	th "github.com/rackspace/gophercloud/testhelper"
)

const endpoint = "http://localhost:57909"

func endpointClient() *gophercloud.ServiceClient {
	return &gophercloud.ServiceClient{Endpoint: endpoint}
}

func TestCreateURL(t *testing.T) {
	actual := createURL(endpointClient())
	expected := endpoint + "servers"
	th.CheckEquals(t, expected, actual)
}

func TestListURL(t *testing.T) {
	actual := listURL(endpointClient())
	expected := endpoint + "servers"
	th.CheckEquals(t, expected, actual)
}

func TestListDetailURL(t *testing.T) {
	actual := listDetailURL(endpointClient())
	expected := endpoint + "servers/detail"
	th.CheckEquals(t, expected, actual)
}

func TestDeleteURL(t *testing.T) {
	actual := deleteURL(endpointClient(), "foo")
	expected := endpoint + "servers/foo"
	th.CheckEquals(t, expected, actual)
}

func TestGetURL(t *testing.T) {
	actual := getURL(endpointClient(), "foo")
	expected := endpoint + "servers/foo"
	th.CheckEquals(t, expected, actual)
}

func TestUpdateURL(t *testing.T) {
	actual := updateURL(endpointClient(), "foo")
	expected := endpoint + "servers/foo"
	th.CheckEquals(t, expected, actual)
}

func TestActionURL(t *testing.T) {
	actual := actionURL(endpointClient(), "foo")
	expected := endpoint + "servers/foo/action"
	th.CheckEquals(t, expected, actual)
}

func TestMetadatumURL(t *testing.T) {
	actual := metadatumURL(endpointClient(), "foo", "bar")
	expected := endpoint + "servers/foo/metadata/bar"
	th.CheckEquals(t, expected, actual)
}

func TestMetadataURL(t *testing.T) {
	actual := metadataURL(endpointClient(), "foo")
	expected := endpoint + "servers/foo/metadata"
	th.CheckEquals(t, expected, actual)
}

func TestPasswordURL(t *testing.T) {
	actual := passwordURL(endpointClient(), "foo")
	expected := endpoint + "servers/foo/os-server-password"
	th.CheckEquals(t, expected, actual)
}
