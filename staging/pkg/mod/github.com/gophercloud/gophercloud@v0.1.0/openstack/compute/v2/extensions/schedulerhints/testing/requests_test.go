package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/schedulerhints"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestCreateOpts(t *testing.T) {
	base := servers.CreateOpts{
		Name:      "createdserver",
		ImageRef:  "asdfasdfasdf",
		FlavorRef: "performance1-1",
	}

	schedulerHints := schedulerhints.SchedulerHints{
		Group: "101aed42-22d9-4a3e-9ba1-21103b0d1aba",
		DifferentHost: []string{
			"a0cf03a5-d921-4877-bb5c-86d26cf818e1",
			"8c19174f-4220-44f0-824a-cd1eeef10287",
		},
		SameHost: []string{
			"a0cf03a5-d921-4877-bb5c-86d26cf818e1",
			"8c19174f-4220-44f0-824a-cd1eeef10287",
		},
		Query:                []interface{}{">=", "$free_ram_mb", "1024"},
		TargetCell:           "foobar",
		BuildNearHostIP:      "192.168.1.1/24",
		AdditionalProperties: map[string]interface{}{"reservation": "a0cf03a5-d921-4877-bb5c-86d26cf818e1"},
	}

	ext := schedulerhints.CreateOptsExt{
		CreateOptsBuilder: base,
		SchedulerHints:    schedulerHints,
	}

	expected := `
		{
			"server": {
				"name": "createdserver",
				"imageRef": "asdfasdfasdf",
				"flavorRef": "performance1-1"
			},
			"os:scheduler_hints": {
				"group": "101aed42-22d9-4a3e-9ba1-21103b0d1aba",
				"different_host": [
					"a0cf03a5-d921-4877-bb5c-86d26cf818e1",
					"8c19174f-4220-44f0-824a-cd1eeef10287"
				],
				"same_host": [
					"a0cf03a5-d921-4877-bb5c-86d26cf818e1",
					"8c19174f-4220-44f0-824a-cd1eeef10287"
				],
				"query": [
					">=", "$free_ram_mb", "1024"
				],
				"target_cell": "foobar",
				"build_near_host_ip": "192.168.1.1",
				"cidr": "/24",
				"reservation": "a0cf03a5-d921-4877-bb5c-86d26cf818e1"
			}
		}
	`
	actual, err := ext.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, expected, actual)
}

func TestCreateOptsWithComplexQuery(t *testing.T) {
	base := servers.CreateOpts{
		Name:      "createdserver",
		ImageRef:  "asdfasdfasdf",
		FlavorRef: "performance1-1",
	}

	schedulerHints := schedulerhints.SchedulerHints{
		Group: "101aed42-22d9-4a3e-9ba1-21103b0d1aba",
		DifferentHost: []string{
			"a0cf03a5-d921-4877-bb5c-86d26cf818e1",
			"8c19174f-4220-44f0-824a-cd1eeef10287",
		},
		SameHost: []string{
			"a0cf03a5-d921-4877-bb5c-86d26cf818e1",
			"8c19174f-4220-44f0-824a-cd1eeef10287",
		},
		Query:                []interface{}{"and", []string{">=", "$free_ram_mb", "1024"}, []string{">=", "$free_disk_mb", "204800"}},
		TargetCell:           "foobar",
		BuildNearHostIP:      "192.168.1.1/24",
		AdditionalProperties: map[string]interface{}{"reservation": "a0cf03a5-d921-4877-bb5c-86d26cf818e1"},
	}

	ext := schedulerhints.CreateOptsExt{
		CreateOptsBuilder: base,
		SchedulerHints:    schedulerHints,
	}

	expected := `
		{
			"server": {
				"name": "createdserver",
				"imageRef": "asdfasdfasdf",
				"flavorRef": "performance1-1"
			},
			"os:scheduler_hints": {
				"group": "101aed42-22d9-4a3e-9ba1-21103b0d1aba",
				"different_host": [
					"a0cf03a5-d921-4877-bb5c-86d26cf818e1",
					"8c19174f-4220-44f0-824a-cd1eeef10287"
				],
				"same_host": [
					"a0cf03a5-d921-4877-bb5c-86d26cf818e1",
					"8c19174f-4220-44f0-824a-cd1eeef10287"
				],
				"query": [
					"and",
					[">=", "$free_ram_mb", "1024"],
					[">=", "$free_disk_mb", "204800"]
				],
				"target_cell": "foobar",
				"build_near_host_ip": "192.168.1.1",
				"cidr": "/24",
				"reservation": "a0cf03a5-d921-4877-bb5c-86d26cf818e1"
			}
		}
	`
	actual, err := ext.ToServerCreateMap()
	th.AssertNoErr(t, err)
	th.CheckJSONEquals(t, expected, actual)
}
