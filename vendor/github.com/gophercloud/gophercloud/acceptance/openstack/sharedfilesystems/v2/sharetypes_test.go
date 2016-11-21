package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/sharetypes"
)

func TestShareTypeCreateDestroy(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create shared file system client: %v", err)
	}

	shareType, err := CreateShareType(t, client)
	if err != nil {
		t.Fatalf("Unable to create share type: %v", err)
	}

	PrintShareType(t, shareType)

	defer DeleteShareType(t, client, shareType)
}

func TestShareTypeList(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create a shared file system client: %v", err)
	}

	allPages, err := sharetypes.List(client, sharetypes.ListOpts{}).AllPages()
	if err != nil {
		t.Fatalf("Unable to retrieve share types: %v", err)
	}

	allShareTypes, err := sharetypes.ExtractShareTypes(allPages)
	if err != nil {
		t.Fatalf("Unable to extract share types: %v", err)
	}

	for _, shareType := range allShareTypes {
		PrintShareType(t, &shareType)
	}
}
