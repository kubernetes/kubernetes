package testing

import (
	"fmt"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/db/v1/datastores"
	"github.com/gophercloud/gophercloud/openstack/db/v1/instances"
	"github.com/gophercloud/gophercloud/testhelper/fixture"
)

var (
	timestamp  = "2015-11-12T14:22:42"
	timeVal, _ = time.Parse(gophercloud.RFC3339NoZ, timestamp)
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
        "href": "https://openstack.example.com/v1.0/1234/flavors/1",
        "rel": "self"
      },
      {
        "href": "https://openstack.example.com/v1.0/1234/flavors/1",
        "rel": "bookmark"
      }
    ]
  },
  "links": [
    {
      "href": "https://openstack.example.com/v1.0/1234/instances/1",
      "rel": "self"
    }
  ],
  "hostname": "e09ad9a3f73309469cf1f43d11e79549caf9acf2.openstack.example.com",
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
	instanceID    = "{instanceID}"
	configGroupID = "00000000-0000-0000-0000-000000000000"
	rootURL       = "/instances"
	resURL        = rootURL + "/" + instanceID
	uRootURL      = resURL + "/root"
	aURL          = resURL + "/action"
)

var (
	restartReq                  = `{"restart": {}}`
	resizeReq                   = `{"resize": {"flavorRef": "2"}}`
	resizeVolReq                = `{"resize": {"volume": {"size": 4}}}`
	attachConfigurationGroupReq = `{"instance": {"configuration": "00000000-0000-0000-0000-000000000000"}}`
	detachConfigurationGroupReq = `{"instance": {}}`
)

var (
	createResp        = fmt.Sprintf(`{"instance": %s}`, instance)
	listInstancesResp = fmt.Sprintf(`{"instances":[%s]}`, instance)
	getInstanceResp   = createResp
	enableUserResp    = `{"user":{"name":"root","password":"secretsecret"}}`
	isUserEnabledResp = `{"rootEnabled":true}`
)

var expectedInstance = instances.Instance{
	Created: timeVal,
	Updated: timeVal,
	Flavor: instances.Flavor{
		ID: "1",
		Links: []gophercloud.Link{
			{Href: "https://openstack.example.com/v1.0/1234/flavors/1", Rel: "self"},
			{Href: "https://openstack.example.com/v1.0/1234/flavors/1", Rel: "bookmark"},
		},
	},
	Hostname: "e09ad9a3f73309469cf1f43d11e79549caf9acf2.openstack.example.com",
	ID:       instanceID,
	Links: []gophercloud.Link{
		{Href: "https://openstack.example.com/v1.0/1234/instances/1", Rel: "self"},
	},
	Name:   "json_rack_instance",
	Status: "BUILD",
	Volume: instances.Volume{Size: 2},
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

func HandleAttachConfigurationGroup(t *testing.T) {
	fixture.SetupHandler(t, resURL, "PUT", attachConfigurationGroupReq, "", 202)
}

func HandleDetachConfigurationGroup(t *testing.T) {
	fixture.SetupHandler(t, resURL, "PUT", detachConfigurationGroupReq, "", 202)
}
