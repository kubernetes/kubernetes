package testing

import (
	"fmt"
	"testing"

	"github.com/gophercloud/gophercloud/testhelper/fixture"
)

const flavor = `
{
	"id": %s,
	"links": [
		{
			"href": "https://openstack.example.com/v1.0/1234/flavors/%s",
			"rel": "self"
		},
		{
			"href": "https://openstack.example.com/flavors/%s",
			"rel": "bookmark"
		}
	],
	"name": "%s",
	"ram": %d,
	"str_id": "%s"
}
`

var (
	flavorID = "{flavorID}"
	_baseURL = "/flavors"
	resURL   = "/flavors/" + flavorID
)

var (
	flavor1 = fmt.Sprintf(flavor, "1", "1", "1", "m1.tiny", 512, "1")
	flavor2 = fmt.Sprintf(flavor, "2", "2", "2", "m1.small", 1024, "2")
	flavor3 = fmt.Sprintf(flavor, "3", "3", "3", "m1.medium", 2048, "3")
	flavor4 = fmt.Sprintf(flavor, "4", "4", "4", "m1.large", 4096, "4")
	flavor5 = fmt.Sprintf(flavor, "null", "d1", "d1", "ds512M", 512, "d1")

	listFlavorsResp = fmt.Sprintf(`{"flavors":[%s, %s, %s, %s, %s]}`, flavor1, flavor2, flavor3, flavor4, flavor5)
	getFlavorResp   = fmt.Sprintf(`{"flavor": %s}`, flavor1)
)

func HandleList(t *testing.T) {
	fixture.SetupHandler(t, _baseURL, "GET", "", listFlavorsResp, 200)
}

func HandleGet(t *testing.T) {
	fixture.SetupHandler(t, resURL, "GET", "", getFlavorResp, 200)
}
