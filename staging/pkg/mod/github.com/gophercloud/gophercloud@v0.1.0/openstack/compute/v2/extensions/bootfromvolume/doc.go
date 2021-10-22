/*
Package bootfromvolume extends a server create request with the ability to
specify block device options. This can be used to boot a server from a block
storage volume as well as specify multiple ephemeral disks upon creation.

It is recommended to refer to the Block Device Mapping documentation to see
all possible ways to configure a server's block devices at creation time:

https://docs.openstack.org/nova/latest/user/block-device-mapping.html

Note that this package implements `block_device_mapping_v2`.

Example of Creating a Server From an Image

This example will boot a server from an image and use a standard ephemeral
disk as the server's root disk. This is virtually no different than creating
a server without using block device mappings.

	blockDevices := []bootfromvolume.BlockDevice{
		bootfromvolume.BlockDevice{
			BootIndex:           0,
			DeleteOnTermination: true,
			DestinationType:     bootfromvolume.DestinationLocal,
			SourceType:          bootfromvolume.SourceImage,
			UUID:                "image-uuid",
		},
	}

	serverCreateOpts := servers.CreateOpts{
		Name:      "server_name",
		FlavorRef: "flavor-uuid",
		ImageRef:  "image-uuid",
	}

	createOpts := bootfromvolume.CreateOptsExt{
		CreateOptsBuilder: serverCreateOpts,
		BlockDevice:       blockDevices,
	}

	server, err := bootfromvolume.Create(client, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example of Creating a Server From a New Volume

This example will create a block storage volume based on the given Image. The
server will use this volume as its root disk.

	blockDevices := []bootfromvolume.BlockDevice{
		bootfromvolume.BlockDevice{
			DeleteOnTermination: true,
			DestinationType:     bootfromvolume.DestinationVolume,
			SourceType:          bootfromvolume.SourceImage,
			UUID:                "image-uuid",
			VolumeSize:          2,
		},
	}

	serverCreateOpts := servers.CreateOpts{
		Name:      "server_name",
		FlavorRef: "flavor-uuid",
	}

	createOpts := bootfromvolume.CreateOptsExt{
		CreateOptsBuilder: serverCreateOpts,
		BlockDevice:       blockDevices,
	}

	server, err := bootfromvolume.Create(client, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example of Creating a Server From an Existing Volume

This example will create a server with an existing volume as its root disk.

	blockDevices := []bootfromvolume.BlockDevice{
		bootfromvolume.BlockDevice{
			DeleteOnTermination: true,
			DestinationType:     bootfromvolume.DestinationVolume,
			SourceType:          bootfromvolume.SourceVolume,
			UUID:                "volume-uuid",
		},
	}

	serverCreateOpts := servers.CreateOpts{
		Name:      "server_name",
		FlavorRef: "flavor-uuid",
	}

	createOpts := bootfromvolume.CreateOptsExt{
		CreateOptsBuilder: serverCreateOpts,
		BlockDevice:       blockDevices,
	}

	server, err := bootfromvolume.Create(client, createOpts).Extract()
	if err != nil {
		panic(err)
	}

Example of Creating a Server with Multiple Ephemeral Disks

This example will create a server with multiple ephemeral disks. The first
block device will be based off of an existing Image. Each additional
ephemeral disks must have an index of -1.

	blockDevices := []bootfromvolume.BlockDevice{
		bootfromvolume.BlockDevice{
			BootIndex:           0,
			DestinationType:     bootfromvolume.DestinationLocal,
			DeleteOnTermination: true,
			SourceType:          bootfromvolume.SourceImage,
			UUID:                "image-uuid",
			VolumeSize:          5,
		},
		bootfromvolume.BlockDevice{
			BootIndex:           -1,
			DestinationType:     bootfromvolume.DestinationLocal,
			DeleteOnTermination: true,
			GuestFormat:         "ext4",
			SourceType:          bootfromvolume.SourceBlank,
			VolumeSize:          1,
		},
		bootfromvolume.BlockDevice{
			BootIndex:           -1,
			DestinationType:     bootfromvolume.DestinationLocal,
			DeleteOnTermination: true,
			GuestFormat:         "ext4",
			SourceType:          bootfromvolume.SourceBlank,
			VolumeSize:          1,
		},
	}

	serverCreateOpts := servers.CreateOpts{
		Name:      "server_name",
		FlavorRef: "flavor-uuid",
		ImageRef:  "image-uuid",
	}

	createOpts := bootfromvolume.CreateOptsExt{
		CreateOptsBuilder: serverCreateOpts,
		BlockDevice:       blockDevices,
	}

	server, err := bootfromvolume.Create(client, createOpts).Extract()
	if err != nil {
		panic(err)
	}
*/
package bootfromvolume
