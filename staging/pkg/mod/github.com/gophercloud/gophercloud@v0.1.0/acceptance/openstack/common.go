// Package openstack contains common functions that can be used
// across all OpenStack components for acceptance testing.
package openstack

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/common/extensions"
)

// PrintExtension prints an extension and all of its attributes.
func PrintExtension(t *testing.T, extension *extensions.Extension) {
	t.Logf("Name: %s", extension.Name)
	t.Logf("Namespace: %s", extension.Namespace)
	t.Logf("Alias: %s", extension.Alias)
	t.Logf("Description: %s", extension.Description)
	t.Logf("Updated: %s", extension.Updated)
	t.Logf("Links: %v", extension.Links)
}
