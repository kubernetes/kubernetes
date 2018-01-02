package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/bootfromvolume"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestBootFromNewVolume(t *testing.T) {
	base := servers.CreateOpts{
		Name:      "createdserver",
		FlavorRef: "performance1-1",
	}

	ext := bootfromvolume.CreateOptsExt{
		CreateOptsBuilder: base,
		BlockDevice: []bootfromvolume.BlockDevice{
			{
				UUID:                "123456",
				SourceType:          bootfromvolume.SourceImage,
				DestinationType:     bootfromvolume.DestinationVolume,
				VolumeSize:          10,
				DeleteOnTermination: true,
			},
		},
	}

	expected := `
    {
      "server": {
        "name":"createdserver",
        "flavorRef":"performance1-1",
        "imageRef":"",
        "block_device_mapping_v2":[
          {
            "uuid":"123456",
            "source_type":"image",
            "destination_type":"volume",
            "boot_index": 0,
            "delete_on_termination": true,
            "volume_size": 10
          }
        ]
      }
    }
  `
	actual, err := ext.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, expected, actual)
}

func TestBootFromExistingVolume(t *testing.T) {
	base := servers.CreateOpts{
		Name:      "createdserver",
		FlavorRef: "performance1-1",
	}

	ext := bootfromvolume.CreateOptsExt{
		CreateOptsBuilder: base,
		BlockDevice: []bootfromvolume.BlockDevice{
			{
				UUID:                "123456",
				SourceType:          bootfromvolume.SourceVolume,
				DestinationType:     bootfromvolume.DestinationVolume,
				DeleteOnTermination: true,
			},
		},
	}

	expected := `
    {
      "server": {
        "name":"createdserver",
        "flavorRef":"performance1-1",
        "imageRef":"",
        "block_device_mapping_v2":[
          {
            "uuid":"123456",
            "source_type":"volume",
            "destination_type":"volume",
            "boot_index": 0,
            "delete_on_termination": true
          }
        ]
      }
    }
  `
	actual, err := ext.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, expected, actual)
}

func TestBootFromImage(t *testing.T) {
	base := servers.CreateOpts{
		Name:      "createdserver",
		ImageRef:  "asdfasdfasdf",
		FlavorRef: "performance1-1",
	}

	ext := bootfromvolume.CreateOptsExt{
		CreateOptsBuilder: base,
		BlockDevice: []bootfromvolume.BlockDevice{
			{
				BootIndex:           0,
				DeleteOnTermination: true,
				DestinationType:     bootfromvolume.DestinationLocal,
				SourceType:          bootfromvolume.SourceImage,
				UUID:                "asdfasdfasdf",
			},
		},
	}

	expected := `
    {
      "server": {
        "name": "createdserver",
        "imageRef": "asdfasdfasdf",
        "flavorRef": "performance1-1",
        "block_device_mapping_v2":[
          {
            "boot_index": 0,
            "delete_on_termination": true,
            "destination_type":"local",
            "source_type":"image",
            "uuid":"asdfasdfasdf"
          }
        ]
      }
    }
  `
	actual, err := ext.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, expected, actual)
}

func TestCreateMultiEphemeralOpts(t *testing.T) {
	base := servers.CreateOpts{
		Name:      "createdserver",
		ImageRef:  "asdfasdfasdf",
		FlavorRef: "performance1-1",
	}

	ext := bootfromvolume.CreateOptsExt{
		CreateOptsBuilder: base,
		BlockDevice: []bootfromvolume.BlockDevice{
			{
				BootIndex:           0,
				DeleteOnTermination: true,
				DestinationType:     bootfromvolume.DestinationLocal,
				SourceType:          bootfromvolume.SourceImage,
				UUID:                "asdfasdfasdf",
			},
			{
				BootIndex:           -1,
				DeleteOnTermination: true,
				DestinationType:     bootfromvolume.DestinationLocal,
				GuestFormat:         "ext4",
				SourceType:          bootfromvolume.SourceBlank,
				VolumeSize:          1,
			},
			{
				BootIndex:           -1,
				DeleteOnTermination: true,
				DestinationType:     bootfromvolume.DestinationLocal,
				GuestFormat:         "ext4",
				SourceType:          bootfromvolume.SourceBlank,
				VolumeSize:          1,
			},
		},
	}

	expected := `
    {
      "server": {
        "name": "createdserver",
        "imageRef": "asdfasdfasdf",
        "flavorRef": "performance1-1",
        "block_device_mapping_v2":[
          {
            "boot_index": 0,
            "delete_on_termination": true,
            "destination_type":"local",
            "source_type":"image",
            "uuid":"asdfasdfasdf"
          },
          {
            "boot_index": -1,
            "delete_on_termination": true,
            "destination_type":"local",
            "guest_format":"ext4",
            "source_type":"blank",
            "volume_size": 1
          },
          {
            "boot_index": -1,
            "delete_on_termination": true,
            "destination_type":"local",
            "guest_format":"ext4",
            "source_type":"blank",
            "volume_size": 1
          }
        ]
      }
    }
  `
	actual, err := ext.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, expected, actual)
}

func TestAttachNewVolume(t *testing.T) {
	base := servers.CreateOpts{
		Name:      "createdserver",
		ImageRef:  "asdfasdfasdf",
		FlavorRef: "performance1-1",
	}

	ext := bootfromvolume.CreateOptsExt{
		CreateOptsBuilder: base,
		BlockDevice: []bootfromvolume.BlockDevice{
			{
				BootIndex:           0,
				DeleteOnTermination: true,
				DestinationType:     bootfromvolume.DestinationLocal,
				SourceType:          bootfromvolume.SourceImage,
				UUID:                "asdfasdfasdf",
			},
			{
				BootIndex:           1,
				DeleteOnTermination: true,
				DestinationType:     bootfromvolume.DestinationVolume,
				SourceType:          bootfromvolume.SourceBlank,
				VolumeSize:          1,
			},
		},
	}

	expected := `
    {
      "server": {
        "name": "createdserver",
        "imageRef": "asdfasdfasdf",
        "flavorRef": "performance1-1",
        "block_device_mapping_v2":[
          {
            "boot_index": 0,
            "delete_on_termination": true,
            "destination_type":"local",
            "source_type":"image",
            "uuid":"asdfasdfasdf"
          },
          {
            "boot_index": 1,
            "delete_on_termination": true,
            "destination_type":"volume",
            "source_type":"blank",
            "volume_size": 1
          }
        ]
      }
    }
  `
	actual, err := ext.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, expected, actual)
}

func TestAttachExistingVolume(t *testing.T) {
	base := servers.CreateOpts{
		Name:      "createdserver",
		ImageRef:  "asdfasdfasdf",
		FlavorRef: "performance1-1",
	}

	ext := bootfromvolume.CreateOptsExt{
		CreateOptsBuilder: base,
		BlockDevice: []bootfromvolume.BlockDevice{
			{
				BootIndex:           0,
				DeleteOnTermination: true,
				DestinationType:     bootfromvolume.DestinationLocal,
				SourceType:          bootfromvolume.SourceImage,
				UUID:                "asdfasdfasdf",
			},
			{
				BootIndex:           1,
				DeleteOnTermination: true,
				DestinationType:     bootfromvolume.DestinationVolume,
				SourceType:          bootfromvolume.SourceVolume,
				UUID:                "123456",
				VolumeSize:          1,
			},
		},
	}

	expected := `
    {
      "server": {
        "name": "createdserver",
        "imageRef": "asdfasdfasdf",
        "flavorRef": "performance1-1",
        "block_device_mapping_v2":[
          {
            "boot_index": 0,
            "delete_on_termination": true,
            "destination_type":"local",
            "source_type":"image",
            "uuid":"asdfasdfasdf"
          },
          {
            "boot_index": 1,
            "delete_on_termination": true,
            "destination_type":"volume",
            "source_type":"volume",
            "uuid":"123456",
            "volume_size": 1
          }
        ]
      }
    }
  `
	actual, err := ext.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, expected, actual)
}
