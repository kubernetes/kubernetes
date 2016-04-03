package objects

import (
	"testing"

	"github.com/rackspace/gophercloud"
	th "github.com/rackspace/gophercloud/testhelper"
)

const endpoint = "http://localhost:57909/"

func endpointClient() *gophercloud.ServiceClient {
	return &gophercloud.ServiceClient{Endpoint: endpoint}
}

func TestListURL(t *testing.T) {
	actual := listURL(endpointClient(), "foo")
	expected := endpoint + "foo"
	th.CheckEquals(t, expected, actual)
}

func TestCopyURL(t *testing.T) {
	actual := copyURL(endpointClient(), "foo", "bar")
	expected := endpoint + "foo/bar"
	th.CheckEquals(t, expected, actual)
}

func TestCreateURL(t *testing.T) {
	actual := createURL(endpointClient(), "foo", "bar")
	expected := endpoint + "foo/bar"
	th.CheckEquals(t, expected, actual)
}

func TestGetURL(t *testing.T) {
	actual := getURL(endpointClient(), "foo", "bar")
	expected := endpoint + "foo/bar"
	th.CheckEquals(t, expected, actual)
}

func TestDeleteURL(t *testing.T) {
	actual := deleteURL(endpointClient(), "foo", "bar")
	expected := endpoint + "foo/bar"
	th.CheckEquals(t, expected, actual)
}

func TestDownloadURL(t *testing.T) {
	actual := downloadURL(endpointClient(), "foo", "bar")
	expected := endpoint + "foo/bar"
	th.CheckEquals(t, expected, actual)
}

func TestUpdateURL(t *testing.T) {
	actual := updateURL(endpointClient(), "foo", "bar")
	expected := endpoint + "foo/bar"
	th.CheckEquals(t, expected, actual)
}
