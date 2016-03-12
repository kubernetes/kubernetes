package databases

import (
	"testing"

	"github.com/rackspace/gophercloud/testhelper/fixture"
)

var (
	instanceID = "{instanceID}"
	resURL     = "/instances/" + instanceID + "/databases"
)

var createDBsReq = `
{
	"databases": [
		{
			"character_set": "utf8",
			"collate": "utf8_general_ci",
			"name": "testingdb"
		},
		{
			"name": "sampledb"
		}
	]
}
`

var listDBsResp = `
{
	"databases": [
		{
			"name": "anotherexampledb"
		},
		{
			"name": "exampledb"
		},
		{
			"name": "nextround"
		},
		{
			"name": "sampledb"
		},
		{
			"name": "testingdb"
		}
	]
}
`

func HandleCreate(t *testing.T) {
	fixture.SetupHandler(t, resURL, "POST", createDBsReq, "", 202)
}

func HandleList(t *testing.T) {
	fixture.SetupHandler(t, resURL, "GET", "", listDBsResp, 200)
}

func HandleDelete(t *testing.T) {
	fixture.SetupHandler(t, resURL+"/{dbName}", "DELETE", "", "", 202)
}
