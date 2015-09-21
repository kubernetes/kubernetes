package base

import (
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestGetHomeDocument(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSuccessfully(t)

	actual, err := Get(fake.ServiceClient()).Extract()
	th.CheckNoErr(t, err)

	expected := HomeDocument{
		"rel/cdn": map[string]interface{}{
			"href-template": "services{?marker,limit}",
			"href-vars": map[string]interface{}{
				"marker": "param/marker",
				"limit":  "param/limit",
			},
			"hints": map[string]interface{}{
				"allow": []string{"GET"},
				"formats": map[string]interface{}{
					"application/json": map[string]interface{}{},
				},
			},
		},
	}
	th.CheckDeepEquals(t, expected, *actual)
}

func TestPing(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePingSuccessfully(t)

	err := Ping(fake.ServiceClient()).ExtractErr()
	th.CheckNoErr(t, err)
}
