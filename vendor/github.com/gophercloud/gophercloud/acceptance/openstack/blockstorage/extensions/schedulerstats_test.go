// +build acceptance blockstorage

package extensions

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/extensions/schedulerstats"
)

func TestSchedulerStatsList(t *testing.T) {
	blockClient, err := clients.NewBlockStorageV2Client()
	if err != nil {
		t.Fatalf("Unable to create a blockstorage client: %v", err)
	}

	listOpts := schedulerstats.ListOpts{
		Detail: true,
	}

	allPages, err := schedulerstats.List(blockClient, listOpts).AllPages()
	if err != nil {
		t.Fatalf("Unable to query schedulerstats: %v", err)
	}

	allStats, err := schedulerstats.ExtractStoragePools(allPages)
	if err != nil {
		t.Fatalf("Unable to extract pools: %v", err)
	}

	for _, stat := range allStats {
		tools.PrintResource(t, stat)
	}
}
