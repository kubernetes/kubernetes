/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package internal

import (
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

// VAPI REST Paths
const (
	SessionPath                    = "/com/vmware/cis/session"
	CategoryPath                   = "/com/vmware/cis/tagging/category"
	TagPath                        = "/com/vmware/cis/tagging/tag"
	AssociationPath                = "/com/vmware/cis/tagging/tag-association"
	LibraryPath                    = "/com/vmware/content/library"
	LibraryItemFileData            = "/com/vmware/cis/data"
	LibraryItemPath                = "/com/vmware/content/library/item"
	LibraryItemFilePath            = "/com/vmware/content/library/item/file"
	LibraryItemUpdateSession       = "/com/vmware/content/library/item/update-session"
	LibraryItemUpdateSessionFile   = "/com/vmware/content/library/item/updatesession/file"
	LibraryItemDownloadSession     = "/com/vmware/content/library/item/download-session"
	LibraryItemDownloadSessionFile = "/com/vmware/content/library/item/downloadsession/file"
	LocalLibraryPath               = "/com/vmware/content/local-library"
	SubscribedLibraryPath          = "/com/vmware/content/subscribed-library"
	SubscribedLibraryItem          = "/com/vmware/content/library/subscribed-item"
	Subscriptions                  = "/com/vmware/content/library/subscriptions"
	VCenterOVFLibraryItem          = "/com/vmware/vcenter/ovf/library-item"
	VCenterVMTXLibraryItem         = "/vcenter/vm-template/library-items"
	VCenterVM                      = "/vcenter/vm"
	SessionCookieName              = "vmware-api-session-id"
	UseHeaderAuthn                 = "vmware-use-header-authn"
)

// AssociatedObject is the same structure as types.ManagedObjectReference,
// just with a different field name (ID instead of Value).
// In the API we use mo.Reference, this type is only used for wire transfer.
type AssociatedObject struct {
	Type  string `json:"type"`
	Value string `json:"id"`
}

// Reference implements mo.Reference
func (o AssociatedObject) Reference() types.ManagedObjectReference {
	return types.ManagedObjectReference(o)
}

// Association for tag-association requests.
type Association struct {
	ObjectID *AssociatedObject `json:"object_id,omitempty"`
}

// NewAssociation returns an Association, converting ref to an AssociatedObject.
func NewAssociation(ref mo.Reference) Association {
	obj := AssociatedObject(ref.Reference())
	return Association{
		ObjectID: &obj,
	}
}

type SubscriptionDestination struct {
	ID string `json:"subscription"`
}

type SubscriptionDestinationSpec struct {
	Subscriptions []SubscriptionDestination `json:"subscriptions,omitempty"`
}

type SubscriptionItemDestinationSpec struct {
	Force         bool                      `json:"force_sync_content"`
	Subscriptions []SubscriptionDestination `json:"subscriptions,omitempty"`
}
