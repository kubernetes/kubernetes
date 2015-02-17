// +build fixtures

package images

import (
	os "github.com/rackspace/gophercloud/openstack/compute/v2/images"
)

// ListOutput is an example response from an /images/detail request.
const ListOutput = `
{
	"images": [
		{
			"OS-DCF:diskConfig": "MANUAL",
			"OS-EXT-IMG-SIZE:size": 1.017415075e+09,
			"created": "2014-10-01T15:49:02Z",
			"id": "30aa010e-080e-4d4b-a7f9-09fc55b07d69",
			"links": [
				{
					"href": "https://iad.servers.api.rackspacecloud.com/v2/111222/images/30aa010e-080e-4d4b-a7f9-09fc55b07d69",
					"rel": "self"
				},
				{
					"href": "https://iad.servers.api.rackspacecloud.com/111222/images/30aa010e-080e-4d4b-a7f9-09fc55b07d69",
					"rel": "bookmark"
				},
				{
					"href": "https://iad.servers.api.rackspacecloud.com/111222/images/30aa010e-080e-4d4b-a7f9-09fc55b07d69",
					"rel": "alternate",
					"type": "application/vnd.openstack.image"
				}
			],
			"metadata": {
				"auto_disk_config": "disabled",
				"cache_in_nova": "True",
				"com.rackspace__1__build_core": "1",
				"com.rackspace__1__build_managed": "1",
				"com.rackspace__1__build_rackconnect": "1",
				"com.rackspace__1__options": "0",
				"com.rackspace__1__platform_target": "PublicCloud",
				"com.rackspace__1__release_build_date": "2014-10-01_15-46-08",
				"com.rackspace__1__release_id": "100",
				"com.rackspace__1__release_version": "10",
				"com.rackspace__1__source": "kickstart",
				"com.rackspace__1__visible_core": "1",
				"com.rackspace__1__visible_managed": "0",
				"com.rackspace__1__visible_rackconnect": "0",
				"image_type": "base",
				"org.openstack__1__architecture": "x64",
				"org.openstack__1__os_distro": "org.archlinux",
				"org.openstack__1__os_version": "2014.8",
				"os_distro": "arch",
				"os_type": "linux",
				"vm_mode": "hvm"
			},
			"minDisk": 20,
			"minRam": 512,
			"name": "Arch 2014.10 (PVHVM)",
			"progress": 100,
			"status": "ACTIVE",
			"updated": "2014-10-01T19:37:58Z"
		},
		{
			"OS-DCF:diskConfig": "AUTO",
			"OS-EXT-IMG-SIZE:size": 1.060306463e+09,
			"created": "2014-10-01T12:58:11Z",
			"id": "e19a734c-c7e6-443a-830c-242209c4d65d",
			"links": [
				{
					"href": "https://iad.servers.api.rackspacecloud.com/v2/111222/images/e19a734c-c7e6-443a-830c-242209c4d65d",
					"rel": "self"
				},
				{
					"href": "https://iad.servers.api.rackspacecloud.com/111222/images/e19a734c-c7e6-443a-830c-242209c4d65d",
					"rel": "bookmark"
				},
				{
					"href": "https://iad.servers.api.rackspacecloud.com/111222/images/e19a734c-c7e6-443a-830c-242209c4d65d",
					"rel": "alternate",
					"type": "application/vnd.openstack.image"
				}
			],
			"metadata": {
				"auto_disk_config": "True",
				"cache_in_nova": "True",
				"com.rackspace__1__build_core": "1",
				"com.rackspace__1__build_managed": "1",
				"com.rackspace__1__build_rackconnect": "1",
				"com.rackspace__1__options": "0",
				"com.rackspace__1__platform_target": "PublicCloud",
				"com.rackspace__1__release_build_date": "2014-10-01_12-31-03",
				"com.rackspace__1__release_id": "1007",
				"com.rackspace__1__release_version": "6",
				"com.rackspace__1__source": "kickstart",
				"com.rackspace__1__visible_core": "1",
				"com.rackspace__1__visible_managed": "1",
				"com.rackspace__1__visible_rackconnect": "1",
				"image_type": "base",
				"org.openstack__1__architecture": "x64",
				"org.openstack__1__os_distro": "com.ubuntu",
				"org.openstack__1__os_version": "14.04",
				"os_distro": "ubuntu",
				"os_type": "linux",
				"vm_mode": "xen"
			},
			"minDisk": 20,
			"minRam": 512,
			"name": "Ubuntu 14.04 LTS (Trusty Tahr)",
			"progress": 100,
			"status": "ACTIVE",
			"updated": "2014-10-01T15:51:44Z"
		}
	]
}
`

// GetOutput is an example response from an /images request.
const GetOutput = `
{
	"image": {
		"OS-DCF:diskConfig": "AUTO",
		"OS-EXT-IMG-SIZE:size": 1060306463,
		"created": "2014-10-01T12:58:11Z",
		"id": "e19a734c-c7e6-443a-830c-242209c4d65d",
		"links": [
			{
				"href": "https://iad.servers.api.rackspacecloud.com/v2/111222/images/e19a734c-c7e6-443a-830c-242209c4d65d",
				"rel": "self"
			},
			{
				"href": "https://iad.servers.api.rackspacecloud.com/111222/images/e19a734c-c7e6-443a-830c-242209c4d65d",
				"rel": "bookmark"
			},
			{
				"href": "https://iad.servers.api.rackspacecloud.com/111222/images/e19a734c-c7e6-443a-830c-242209c4d65d",
				"rel": "alternate",
				"type": "application/vnd.openstack.image"
			}
		],
		"metadata": {
			"auto_disk_config": "True",
			"cache_in_nova": "True",
			"com.rackspace__1__build_core": "1",
			"com.rackspace__1__build_managed": "1",
			"com.rackspace__1__build_rackconnect": "1",
			"com.rackspace__1__options": "0",
			"com.rackspace__1__platform_target": "PublicCloud",
			"com.rackspace__1__release_build_date": "2014-10-01_12-31-03",
			"com.rackspace__1__release_id": "1007",
			"com.rackspace__1__release_version": "6",
			"com.rackspace__1__source": "kickstart",
			"com.rackspace__1__visible_core": "1",
			"com.rackspace__1__visible_managed": "1",
			"com.rackspace__1__visible_rackconnect": "1",
			"image_type": "base",
			"org.openstack__1__architecture": "x64",
			"org.openstack__1__os_distro": "com.ubuntu",
			"org.openstack__1__os_version": "14.04",
			"os_distro": "ubuntu",
			"os_type": "linux",
			"vm_mode": "xen"
		},
		"minDisk": 20,
		"minRam": 512,
		"name": "Ubuntu 14.04 LTS (Trusty Tahr)",
		"progress": 100,
		"status": "ACTIVE",
		"updated": "2014-10-01T15:51:44Z"
	}
}
`

// ArchImage is the first Image structure that should be parsed from ListOutput.
var ArchImage = os.Image{
	ID:       "30aa010e-080e-4d4b-a7f9-09fc55b07d69",
	Name:     "Arch 2014.10 (PVHVM)",
	Created:  "2014-10-01T15:49:02Z",
	Updated:  "2014-10-01T19:37:58Z",
	MinDisk:  20,
	MinRAM:   512,
	Progress: 100,
	Status:   "ACTIVE",
}

// UbuntuImage is the second Image structure that should be parsed from ListOutput and
// the only image that should be extracted from GetOutput.
var UbuntuImage = os.Image{
	ID:       "e19a734c-c7e6-443a-830c-242209c4d65d",
	Name:     "Ubuntu 14.04 LTS (Trusty Tahr)",
	Created:  "2014-10-01T12:58:11Z",
	Updated:  "2014-10-01T15:51:44Z",
	MinDisk:  20,
	MinRAM:   512,
	Progress: 100,
	Status:   "ACTIVE",
}

// ExpectedImageSlice is the collection of images that should be parsed from ListOutput,
// in order.
var ExpectedImageSlice = []os.Image{ArchImage, UbuntuImage}
