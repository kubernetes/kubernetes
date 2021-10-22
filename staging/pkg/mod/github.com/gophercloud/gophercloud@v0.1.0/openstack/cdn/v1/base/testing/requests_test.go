package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/cdn/v1/base"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestGetHomeDocument(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSuccessfully(t)

	actual, err := base.Get(fake.ServiceClient()).Extract()
	th.CheckNoErr(t, err)

	expected := base.HomeDocument{
		"resources": map[string]interface{}{
			"rel/cdn": map[string]interface{}{
				"href-template": "services{?marker,limit}",
				"href-vars": map[string]interface{}{
					"marker": "param/marker",
					"limit":  "param/limit",
				},
				"hints": map[string]interface{}{
					"allow": []interface{}{"GET"},
					"formats": map[string]interface{}{
						"application/json": map[string]interface{}{},
					},
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

	err := base.Ping(fake.ServiceClient()).ExtractErr()
	th.CheckNoErr(t, err)
}
