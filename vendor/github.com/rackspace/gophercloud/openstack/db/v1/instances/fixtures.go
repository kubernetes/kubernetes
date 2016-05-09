package instances

import (
	"fmt"
	"testing"
	"time"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/db/v1/datastores"
	"github.com/rackspace/gophercloud/openstack/db/v1/flavors"
	"github.com/rackspace/gophercloud/testhelper/fixture"
)

var (
	timestamp  = "2015-11-12T14:22:42Z"
	timeVal, _ = time.Parse(time.RFC3339, timestamp)
)

var instance = `
{
  "created": "` + timestamp + `",
  "datastore": {
    "type": "mysql",
    "version": "5.6"
  },
  "flavor": {
    "id": "1",
    "links": [
      {
        "href": "https://my-openstack.com/v1.0/1234/flavors/1",
        "rel": "self"
      },
      {
        "href": "https://my-openstack.com/v1.0/1234/flavors/1",
        "rel": "bookmark"
      }
    ]
  },
  "links": [
    {
      "href": "https://my-openstack.com/v1.0/1234/instances/1",
      "rel": "self"
    }
  ],
  "hostname": "e09ad9a3f73309469cf1f43d11e79549caf9acf2.my-openstack.com",
  "id": "{instanceID}",
  "name": "json_rack_instance",
  "status": "BUILD",
  "updated": "` + timestamp + `",
  "volume": {
    "size": 2
  }
}
`

var createReq = `
{
	"instance": {
		"databases": [
			{
				"character_set": "utf8",
				"collate": "utf8_general_ci",
				"name": "sampledb"
			},
			{
				"name": "nextround"
			}
		],
		"flavorRef": "1",
		"name": "json_rack_instance",
		"users": [
			{
				"databases": [
					{
						"name": "sampledb"
					}
				],
				"name": "demouser",
				"password": "demopassword"
			}
		],
		"volume": {
			"size": 2
		}
	}
}
`

var (
	instanceID = "{instanceID}"
	rootURL    = "/instances"
	resURL     = rootURL + "/" + instanceID
	uRootURL   = resURL + "/root"
	aURL       = resURL + "/action"
)

var (
	restartReq   = `{"restart": {}}`
	resizeReq    = `{"resize": {"flavorRef": "2"}}`
	resizeVolReq = `{"resize": {"volume": {"size": 4}}}`
)

var (
	createResp        = fmt.Sprintf(`{"instance": %s}`, instance)
	listInstancesResp = fmt.Sprintf(`{"instances":[%s]}`, instance)
	getInstanceResp   = createResp
	enableUserResp    = `{"user":{"name":"root","password":"secretsecret"}}`
	isUserEnabledResp = `{"rootEnabled":true}`
)

var expectedInstance = Instance{
	Created: timeVal,
	Updated: timeVal,
	Flavor: flavors.Flavor{
		ID: "1",
		Links: []gophercloud.Link{
			gophercloud.Link{Href: "https://my-openstack.com/v1.0/1234/flavors/1", Rel: "self"},
			gophercloud.Link{Href: "https://my-openstack.com/v1.0/1234/flavors/1", Rel: "bookmark"},
		},
	},
	Hostname: "e09ad9a3f73309469cf1f43d11e79549caf9acf2.my-openstack.com",
	ID:       instanceID,
	Links: []gophercloud.Link{
		gophercloud.Link{Href: "https://my-openstack.com/v1.0/1234/instances/1", Rel: "self"},
	},
	Name:   "json_rack_instance",
	Status: "BUILD",
	Volume: Volume{Size: 2},
	Datastore: datastores.DatastorePartial{
		Type:    "mysql",
		Version: "5.6",
	},
}

func HandleCreate(t *testing.T) {
	fixture.SetupHandler(t, rootURL, "POST", createReq, createResp, 200)
}

func HandleList(t *testing.T) {
	fixture.SetupHandler(t, rootURL, "GET", "", listInstancesResp, 200)
}

func HandleGet(t *testing.T) {
	fixture.SetupHandler(t, resURL, "GET", "", getInstanceResp, 200)
}

func HandleDelete(t *testing.T) {
	fixture.SetupHandler(t, resURL, "DELETE", "", "", 202)
}

func HandleEnableRoot(t *testing.T) {
	fixture.SetupHandler(t, uRootURL, "POST", "", enableUserResp, 200)
}

func HandleIsRootEnabled(t *testing.T) {
	fixture.SetupHandler(t, uRootURL, "GET", "", isUserEnabledResp, 200)
}

func HandleRestart(t *testing.T) {
	fixture.SetupHandler(t, aURL, "POST", restartReq, "", 202)
}

func HandleResize(t *testing.T) {
	fixture.SetupHandler(t, aURL, "POST", resizeReq, "", 202)
}

func HandleResizeVol(t *testing.T) {
	fixture.SetupHandler(t, aURL, "POST", resizeVolReq, "", 202)
}
