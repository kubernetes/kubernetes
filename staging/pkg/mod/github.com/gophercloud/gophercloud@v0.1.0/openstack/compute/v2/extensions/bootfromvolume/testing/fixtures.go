package testing

import (
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/bootfromvolume"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
)

var BaseCreateOpts = servers.CreateOpts{
	Name:      "createdserver",
	FlavorRef: "performance1-1",
}

var BaseCreateOptsWithImageRef = servers.CreateOpts{
	Name:      "createdserver",
	FlavorRef: "performance1-1",
	ImageRef:  "asdfasdfasdf",
}

const ExpectedNewVolumeRequest = `
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

var NewVolumeRequest = bootfromvolume.CreateOptsExt{
	CreateOptsBuilder: BaseCreateOpts,
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

const ExpectedExistingVolumeRequest = `
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

var ExistingVolumeRequest = bootfromvolume.CreateOptsExt{
	CreateOptsBuilder: BaseCreateOpts,
	BlockDevice: []bootfromvolume.BlockDevice{
		{
			UUID:                "123456",
			SourceType:          bootfromvolume.SourceVolume,
			DestinationType:     bootfromvolume.DestinationVolume,
			DeleteOnTermination: true,
		},
	},
}

const ExpectedImageRequest = `
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

var ImageRequest = bootfromvolume.CreateOptsExt{
	CreateOptsBuilder: BaseCreateOptsWithImageRef,
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

const ExpectedMultiEphemeralRequest = `
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

var MultiEphemeralRequest = bootfromvolume.CreateOptsExt{
	CreateOptsBuilder: BaseCreateOptsWithImageRef,
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

const ExpectedImageAndNewVolumeRequest = `
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
				"volume_size": 1,
				"device_type": "disk",
				"disk_bus": "scsi"
			}
		]
	}
}
`

var ImageAndNewVolumeRequest = bootfromvolume.CreateOptsExt{
	CreateOptsBuilder: BaseCreateOptsWithImageRef,
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
			DeviceType:          "disk",
			DiskBus:             "scsi",
		},
	},
}

const ExpectedImageAndExistingVolumeRequest = `
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

var ImageAndExistingVolumeRequest = bootfromvolume.CreateOptsExt{
	CreateOptsBuilder: BaseCreateOptsWithImageRef,
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
