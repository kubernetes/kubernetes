package bootfromvolume

import (
	"testing"

	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestCreateOpts(t *testing.T) {
	base := servers.CreateOpts{
		Name:      "createdserver",
		ImageRef:  "asdfasdfasdf",
		FlavorRef: "performance1-1",
	}

	ext := CreateOptsExt{
		CreateOptsBuilder: base,
		BlockDevice: []BlockDevice{
			BlockDevice{
				UUID:            "123456",
				SourceType:      Image,
				DestinationType: "volume",
				VolumeSize:      10,
			},
		},
	}

	expected := `
    {
      "server": {
        "name": "createdserver",
        "imageRef": "asdfasdfasdf",
        "flavorRef": "performance1-1",
	"flavorName": "",
	"imageName": "",
        "block_device_mapping_v2":[
          {
            "uuid":"123456",
            "source_type":"image",
            "destination_type":"volume",
            "boot_index": "0",
            "delete_on_termination": "false",
            "volume_size": "10"
          }
        ]
      }
    }
  `
	actual, err := ext.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, expected, actual)
}
