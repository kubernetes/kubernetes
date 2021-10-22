package testing

import (
	"fmt"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/db/v1/configurations"
)

var (
	timestamp  = "2015-11-12T14:22:42"
	timeVal, _ = time.Parse(gophercloud.RFC3339NoZ, timestamp)
)

var singleConfigJSON = `
{
  "created": "` + timestamp + `",
  "datastore_name": "mysql",
  "datastore_version_id": "b00000b0-00b0-0b00-00b0-000b000000bb",
  "datastore_version_name": "5.6",
  "description": "example_description",
  "id": "005a8bb7-a8df-40ee-b0b7-fc144641abc2",
  "name": "example-configuration-name",
  "updated": "` + timestamp + `"
}
`

var singleConfigWithValuesJSON = `
{
  "created": "` + timestamp + `",
  "datastore_name": "mysql",
  "datastore_version_id": "b00000b0-00b0-0b00-00b0-000b000000bb",
  "datastore_version_name": "5.6",
  "description": "example description",
  "id": "005a8bb7-a8df-40ee-b0b7-fc144641abc2",
  "instance_count": 0,
  "name": "example-configuration-name",
  "updated": "` + timestamp + `",
  "values": {
    "collation_server": "latin1_swedish_ci",
    "connect_timeout": 120
  }
}
`

var (
	ListConfigsJSON  = fmt.Sprintf(`{"configurations": [%s]}`, singleConfigJSON)
	GetConfigJSON    = fmt.Sprintf(`{"configuration": %s}`, singleConfigJSON)
	CreateConfigJSON = fmt.Sprintf(`{"configuration": %s}`, singleConfigWithValuesJSON)
)

var CreateReq = `
{
  "configuration": {
    "datastore": {
      "type": "a00000a0-00a0-0a00-00a0-000a000000aa",
      "version": "b00000b0-00b0-0b00-00b0-000b000000bb"
    },
    "description": "example description",
    "name": "example-configuration-name",
    "values": {
      "collation_server": "latin1_swedish_ci",
      "connect_timeout": 120
    }
  }
}
`

var UpdateReq = `
{
  "configuration": {
    "values": {
      "connect_timeout": 300
    }
  }
}
`

var ListInstancesJSON = `
{
  "instances": [
    {
      "id": "d4603f69-ec7e-4e9b-803f-600b9205576f",
      "name": "json_rack_instance"
    }
  ]
}
`

var ListParamsJSON = `
{
  "configuration-parameters": [
    {
      "max": 1,
      "min": 0,
      "name": "innodb_file_per_table",
      "restart_required": true,
      "type": "integer"
    },
    {
      "max": 4294967296,
      "min": 0,
      "name": "key_buffer_size",
      "restart_required": false,
      "type": "integer"
    },
    {
      "max": 65535,
      "min": 2,
      "name": "connect_timeout",
      "restart_required": false,
      "type": "integer"
    },
    {
      "max": 4294967296,
      "min": 0,
      "name": "join_buffer_size",
      "restart_required": false,
      "type": "integer"
    }
  ]
}
`

var GetParamJSON = `
{
  "max": 1,
  "min": 0,
  "name": "innodb_file_per_table",
  "restart_required": true,
  "type": "integer"
}
`

var ExampleConfig = configurations.Config{
	Created:              timeVal,
	DatastoreName:        "mysql",
	DatastoreVersionID:   "b00000b0-00b0-0b00-00b0-000b000000bb",
	DatastoreVersionName: "5.6",
	Description:          "example_description",
	ID:                   "005a8bb7-a8df-40ee-b0b7-fc144641abc2",
	Name:                 "example-configuration-name",
	Updated:              timeVal,
}

var ExampleConfigWithValues = configurations.Config{
	Created:              timeVal,
	DatastoreName:        "mysql",
	DatastoreVersionID:   "b00000b0-00b0-0b00-00b0-000b000000bb",
	DatastoreVersionName: "5.6",
	Description:          "example description",
	ID:                   "005a8bb7-a8df-40ee-b0b7-fc144641abc2",
	Name:                 "example-configuration-name",
	Updated:              timeVal,
	Values: map[string]interface{}{
		"collation_server": "latin1_swedish_ci",
		"connect_timeout":  float64(120),
	},
}
