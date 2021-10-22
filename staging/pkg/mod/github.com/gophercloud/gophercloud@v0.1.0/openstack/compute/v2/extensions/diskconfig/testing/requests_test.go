package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/diskconfig"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestCreateOpts(t *testing.T) {
	base := servers.CreateOpts{
		Name:      "createdserver",
		ImageRef:  "asdfasdfasdf",
		FlavorRef: "performance1-1",
	}

	ext := diskconfig.CreateOptsExt{
		CreateOptsBuilder: base,
		DiskConfig:        diskconfig.Manual,
	}

	expected := `
		{
			"server": {
				"name": "createdserver",
				"imageRef": "asdfasdfasdf",
				"flavorRef": "performance1-1",
				"OS-DCF:diskConfig": "MANUAL"
			}
		}
	`
	actual, err := ext.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, expected, actual)
}

func TestRebuildOpts(t *testing.T) {
	base := servers.RebuildOpts{
		Name:      "rebuiltserver",
		AdminPass: "swordfish",
		ImageID:   "asdfasdfasdf",
	}

	ext := diskconfig.RebuildOptsExt{
		RebuildOptsBuilder: base,
		DiskConfig:         diskconfig.Auto,
	}

	actual, err := ext.ToServerRebuildMap()
	th.AssertNoErr(t, err)

	expected := `
		{
			"rebuild": {
				"name": "rebuiltserver",
				"imageRef": "asdfasdfasdf",
				"adminPass": "swordfish",
				"OS-DCF:diskConfig": "AUTO"
			}
		}
	`
	th.CheckJSONEquals(t, expected, actual)
}

func TestResizeOpts(t *testing.T) {
	base := servers.ResizeOpts{
		FlavorRef: "performance1-8",
	}

	ext := diskconfig.ResizeOptsExt{
		ResizeOptsBuilder: base,
		DiskConfig:        diskconfig.Auto,
	}

	actual, err := ext.ToServerResizeMap()
	th.AssertNoErr(t, err)

	expected := `
		{
			"resize": {
				"flavorRef": "performance1-8",
				"OS-DCF:diskConfig": "AUTO"
			}
		}
	`
	th.CheckJSONEquals(t, expected, actual)
}
