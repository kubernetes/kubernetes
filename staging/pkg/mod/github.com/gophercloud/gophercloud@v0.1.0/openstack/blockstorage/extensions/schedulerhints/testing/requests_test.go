package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/blockstorage/extensions/schedulerhints"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v3/volumes"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestCreateOpts(t *testing.T) {

	base := volumes.CreateOpts{
		Size: 10,
		Name: "testvolume",
	}
	schedulerHints := schedulerhints.SchedulerHints{
		DifferentHost: []string{
			"a0cf03a5-d921-4877-bb5c-86d26cf818e1",
			"8c19174f-4220-44f0-824a-cd1eeef10287",
		},
		SameHost: []string{
			"a0cf03a5-d921-4877-bb5c-86d26cf818e1",
			"8c19174f-4220-44f0-824a-cd1eeef10287",
		},
		LocalToInstance:      "0ffb2c1b-d621-4fc1-9ae4-88d99c088ff6",
		AdditionalProperties: map[string]interface{}{"mark": "a0cf03a5-d921-4877-bb5c-86d26cf818e1"},
	}

	ext := schedulerhints.CreateOptsExt{
		VolumeCreateOptsBuilder: base,
		SchedulerHints:          schedulerHints,
	}

	expected := `
		{
			"volume": {
				"size": 10,
				"name": "testvolume"
			},
			"OS-SCH-HNT:scheduler_hints": {
				"different_host": [
					"a0cf03a5-d921-4877-bb5c-86d26cf818e1",
					"8c19174f-4220-44f0-824a-cd1eeef10287"
				],
				"same_host": [
					"a0cf03a5-d921-4877-bb5c-86d26cf818e1",
					"8c19174f-4220-44f0-824a-cd1eeef10287"
				],
				"local_to_instance": "0ffb2c1b-d621-4fc1-9ae4-88d99c088ff6",
				"mark": "a0cf03a5-d921-4877-bb5c-86d26cf818e1"
			}
		}
	`
	actual, err := ext.ToVolumeCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, expected, actual)
}
