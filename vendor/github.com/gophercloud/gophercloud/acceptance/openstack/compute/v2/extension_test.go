// +build acceptance compute extensions

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/common/extensions"
)

func TestExtensionsList(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
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
		tools.PrintResource(t, extension)
	}
}

func TestExtensionGet(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	extension, err := extensions.Get(client, "os-admin-actions").Extract()
	if err != nil {
		t.Fatalf("Unable to get extension os-admin-actions: %v", err)
	}

	tools.PrintResource(t, extension)
}
