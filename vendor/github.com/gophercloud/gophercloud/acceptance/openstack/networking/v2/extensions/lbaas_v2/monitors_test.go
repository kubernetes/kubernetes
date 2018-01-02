// +build acceptance networking lbaas_v2 monitors

package lbaas_v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors"
)

func TestMonitorsList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	allPages, err := monitors.List(client, nil).AllPages()
	if err != nil {
		t.Fatalf("Unable to list monitors: %v", err)
	}

	allMonitors, err := monitors.ExtractMonitors(allPages)
	if err != nil {
		t.Fatalf("Unable to extract monitors: %v", err)
	}

	for _, monitor := range allMonitors {
		tools.PrintResource(t, monitor)
	}
}
