// +build acceptance networking extensions

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/openstack"
	"github.com/gophercloud/gophercloud/openstack/common/extensions"
)

func TestExtensionsList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	allPages, err := extensions.List(client).AllPages()
	if err != nil {
		t.Fatalf("Unable to list extensions: %v", err)
	}

	allExtensions, err := extensions.ExtractExtensions(allPages)
	if err != nil {
		t.Fatalf("Unable to extract extensions: %v", err)
	}

	for _, extension := range allExtensions {
		openstack.PrintExtension(t, &extension)
	}
}

func TestExtensionGet(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	extension, err := extensions.Get(client, "router").Extract()
	if err != nil {
		t.Fatalf("Unable to get extension port-security: %v", err)
	}

	openstack.PrintExtension(t, extension)
}
