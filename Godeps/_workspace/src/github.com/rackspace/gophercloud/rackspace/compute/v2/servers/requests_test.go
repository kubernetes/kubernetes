package servers

import (
	"testing"

	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/diskconfig"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestCreateOpts(t *testing.T) {
	opts := CreateOpts{
		Name:       "createdserver",
		ImageRef:   "image-id",
		FlavorRef:  "flavor-id",
		KeyPair:    "mykey",
		DiskConfig: diskconfig.Manual,
	}

	expected := `
	{
		"server": {
			"name": "createdserver",
			"imageRef": "image-id",
			"flavorRef": "flavor-id",
			"flavorName": "",
			"imageName": "",
			"key_name": "mykey",
			"OS-DCF:diskConfig": "MANUAL"
		}
	}
	`
	actual, err := opts.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, expected, actual)
}

func TestRebuildOpts(t *testing.T) {
	opts := RebuildOpts{
		Name:       "rebuiltserver",
		AdminPass:  "swordfish",
		ImageID:    "asdfasdfasdf",
		DiskConfig: diskconfig.Auto,
	}

	actual, err := opts.ToServerRebuildMap()
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
