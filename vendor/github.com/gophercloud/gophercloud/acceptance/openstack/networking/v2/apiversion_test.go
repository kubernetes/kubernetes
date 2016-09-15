// +build acceptance networking

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/apiversions"
)

func TestAPIVersionsList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	allPages, err := apiversions.ListVersions(client).AllPages()
	if err != nil {
		t.Fatalf("Unable to list api versions: %v", err)
	}

	allAPIVersions, err := apiversions.ExtractAPIVersions(allPages)
	if err != nil {
		t.Fatalf("Unable to extract api versions: %v", err)
	}

	for _, apiVersion := range allAPIVersions {
		PrintAPIVersion(t, &apiVersion)
	}
}

func TestAPIResourcesList(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	if err != nil {
		t.Fatalf("Unable to create a network client: %v", err)
	}

	allPages, err := apiversions.ListVersionResources(client, "v2.0").AllPages()
	if err != nil {
		t.Fatalf("Unable to list api version reosources: %v", err)
	}

	allVersionResources, err := apiversions.ExtractVersionResources(allPages)
	if err != nil {
		t.Fatalf("Unable to extract version resources: %v", err)
	}

	for _, versionResource := range allVersionResources {
		PrintVersionResource(t, &versionResource)
	}
}
