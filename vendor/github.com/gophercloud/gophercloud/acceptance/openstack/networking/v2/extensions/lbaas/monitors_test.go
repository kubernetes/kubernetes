// +build acceptance networking lbaas monitors

package lbaas

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas/monitors"
)

func TestMonitorsList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	allPages, err := monitors.List(client, monitors.ListOpts{}).AllPages()
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

func TestMonitorsCRUD(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	monitor, err := CreateMonitor(t, client)
	if err != nil {
		t.Fatalf("Unable to create monitor: %v", err)
	}
	defer DeleteMonitor(t, client, monitor.ID)

	tools.PrintResource(t, monitor)

	updateOpts := monitors.UpdateOpts{
		Delay: 999,
	}

	_, err = monitors.Update(client, monitor.ID, updateOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update monitor: %v")
	}

	newMonitor, err := monitors.Get(client, monitor.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get monitor: %v")
	}

	tools.PrintResource(t, newMonitor)
}
