/*
Copyright (c) 2019 VMware, Inc. All Rights Reserved.

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

package vcenter

import (
	"context"
	"crypto/sha1"
	"fmt"
	"log"
	"net/http"
	"path"

	"github.com/vmware/govmomi/vapi/internal"
	"github.com/vmware/govmomi/vapi/library"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

// vcenter vm template
// The vcenter.vm_template API provides structures and services that will let its client manage VMTX template in Content Library.
// http://vmware.github.io/vsphere-automation-sdk-rest/6.7.1/index.html#SVC_com.vmware.vcenter.vm_template.library_items

// Template create spec
type Template struct {
	Description          string                `json:"description,omitempty"`
	DiskStorage          *DiskStorage          `json:"disk_storage,omitempty"`
	DiskStorageOverrides []DiskStorageOverride `json:"disk_storage_overrides,omitempty"`
	Library              string                `json:"library,omitempty"`
	Name                 string                `json:"name,omitempty"`
	Placement            *Placement            `json:"placement,omitempty"`
	SourceVM             string                `json:"source_vm,omitempty"`
	VMHomeStorage        *DiskStorage          `json:"vm_home_storage,omitempty"`
}

// TemplateInfo for a VM template contained in an existing library item
type TemplateInfo struct {
	GuestOS string `json:"guest_OS,omitempty"`
	// TODO...
}

// Placement information used to place the virtual machine template
type Placement = library.Placement

// StoragePolicy for DiskStorage
type StoragePolicy struct {
	Policy string `json:"policy,omitempty"`
	Type   string `json:"type"`
}

// DiskStorage defines the storage specification for VM files
type DiskStorage struct {
	Datastore     string         `json:"datastore,omitempty"`
	StoragePolicy *StoragePolicy `json:"storage_policy,omitempty"`
}

// DiskStorageOverride storage specification for individual disks in the virtual machine template
type DiskStorageOverride struct {
	Key   string      `json:"key"`
	Value DiskStorage `json:"value"`
}

// GuestCustomization spec to apply to the deployed VM
type GuestCustomization struct {
	Name string `json:"name,omitempty"`
}

// HardwareCustomization spec which specifies updates to the deployed VM
type HardwareCustomization struct {
	// TODO
}

// DeployTemplate specification of how a library VM template clone should be deployed.
type DeployTemplate struct {
	Description           string                 `json:"description,omitempty"`
	DiskStorage           *DiskStorage           `json:"disk_storage,omitempty"`
	DiskStorageOverrides  []DiskStorageOverride  `json:"disk_storage_overrides,omitempty"`
	GuestCustomization    *GuestCustomization    `json:"guest_customization,omitempty"`
	HardwareCustomization *HardwareCustomization `json:"hardware_customization,omitempty"`
	Name                  string                 `json:"name,omitempty"`
	Placement             *Placement             `json:"placement,omitempty"`
	PoweredOn             bool                   `json:"powered_on"`
	VMHomeStorage         *DiskStorage           `json:"vm_home_storage,omitempty"`
}

// CheckOut specification
type CheckOut struct {
	Name      string     `json:"name,omitempty"`
	Placement *Placement `json:"placement,omitempty"`
	PoweredOn bool       `json:"powered_on,omitempty"`
}

// CheckIn specification
type CheckIn struct {
	Message string `json:"message"`
}

// CreateTemplate creates a library VMTX item in content library from an existing VM
func (c *Manager) CreateTemplate(ctx context.Context, vmtx Template) (string, error) {
	url := c.Resource(internal.VCenterVMTXLibraryItem)
	var res string
	spec := struct {
		Template `json:"spec"`
	}{vmtx}
	return res, c.Do(ctx, url.Request(http.MethodPost, spec), &res)
}

// DeployTemplateLibraryItem deploys a VM as a copy of the source VM template contained in the given library item
func (c *Manager) DeployTemplateLibraryItem(ctx context.Context, libraryItemID string, deploy DeployTemplate) (*types.ManagedObjectReference, error) {
	url := c.Resource(path.Join(internal.VCenterVMTXLibraryItem, libraryItemID)).WithParam("action", "deploy")
	var res string
	spec := struct {
		DeployTemplate `json:"spec"`
	}{deploy}
	err := c.Do(ctx, url.Request(http.MethodPost, spec), &res)
	if err != nil {
		return nil, err
	}
	return &types.ManagedObjectReference{Type: "VirtualMachine", Value: res}, nil
}

// CheckOut a library item containing a VM template.
func (c *Manager) CheckOut(ctx context.Context, libraryItemID string, checkout *CheckOut) (*types.ManagedObjectReference, error) {
	url := c.Resource(path.Join(internal.VCenterVMTXLibraryItem, libraryItemID, "check-outs")).WithParam("action", "check-out")
	var res string
	spec := struct {
		*CheckOut `json:"spec"`
	}{checkout}
	err := c.Do(ctx, url.Request(http.MethodPost, spec), &res)
	if err != nil {
		return nil, err
	}
	return &types.ManagedObjectReference{Type: "VirtualMachine", Value: res}, nil
}

// CheckIn a VM into the library item.
func (c *Manager) CheckIn(ctx context.Context, libraryItemID string, vm mo.Reference, checkin *CheckIn) (string, error) {
	p := path.Join(internal.VCenterVMTXLibraryItem, libraryItemID, "check-outs", vm.Reference().Value)
	url := c.Resource(p).WithParam("action", "check-in")
	var res string
	spec := struct {
		*CheckIn `json:"spec"`
	}{checkin}
	return res, c.Do(ctx, url.Request(http.MethodPost, spec), &res)
}

// TemplateLibrary params for synchronizing subscription library OVF items to VM Template items
type TemplateLibrary struct {
	Source      library.Library
	Destination library.Library
	Placement   Target
	Include     func(library.Item, *library.Item) bool
	SyncItem    func(context.Context, library.Item, *Deploy, *Template) error
}

func (c *Manager) includeTemplateLibraryItem(src library.Item, dst *library.Item) bool {
	return dst == nil
}

// SyncTemplateLibraryItem deploys an Library OVF item from which a VM template (vmtx) Library item is created.
// The deployed VM is deleted after being converted to a Library vmtx item.
func (c *Manager) SyncTemplateLibraryItem(ctx context.Context, item library.Item, deploy *Deploy, spec *Template) error {
	destroy := false
	if spec.SourceVM == "" {
		ref, err := c.DeployLibraryItem(ctx, item.ID, *deploy)
		if err != nil {
			return err
		}

		destroy = true
		spec.SourceVM = ref.Value
	}

	_, err := c.CreateTemplate(ctx, *spec)

	if destroy {
		// Delete source VM regardless of CreateTemplate result
		url := c.Resource("/vcenter/vm/" + spec.SourceVM)
		derr := c.Do(ctx, url.Request(http.MethodDelete), nil)
		if derr != nil {
			if err == nil {
				// Return Delete error if CreateTemplate was successful
				return derr
			}
			// Return CreateTemplate error and just log Delete error
			log.Printf("destroy %s: %s", spec.SourceVM, derr)
		}
	}

	return err
}

func vmtxSourceName(l library.Library, item library.Item) string {
	sum := sha1.Sum([]byte(path.Join(l.Name, item.Name)))
	return fmt.Sprintf("vmtx-src-%x", sum)
}

// SyncTemplateLibrary converts TemplateLibrary.Source OVF items to VM Template items within TemplateLibrary.Destination
// The optional TemplateLibrary.Include func can be used to filter which items are synced.
// By default all items that don't exist in the Destination library are synced.
// The optional TemplateLibrary.SyncItem func can be used to change how the item is synced, by default SyncTemplateLibraryItem is used.
func (c *Manager) SyncTemplateLibrary(ctx context.Context, l TemplateLibrary, items ...library.Item) error {
	m := library.NewManager(c.Client)
	var err error
	if len(items) == 0 {
		items, err = m.GetLibraryItems(ctx, l.Source.ID)
		if err != nil {
			return err
		}
	}

	templates, err := m.GetLibraryItems(ctx, l.Destination.ID)
	if err != nil {
		return err
	}

	existing := make(map[string]*library.Item)
	for i := range templates {
		existing[templates[i].Name] = &templates[i]
	}

	include := l.Include
	if include == nil {
		include = c.includeTemplateLibraryItem
	}

	sync := l.SyncItem
	if sync == nil {
		sync = c.SyncTemplateLibraryItem
	}

	for _, item := range items {
		if item.Type != library.ItemTypeOVF {
			continue
		}

		// Deploy source VM from library ovf item
		deploy := Deploy{
			DeploymentSpec: DeploymentSpec{
				Name:               vmtxSourceName(l.Destination, item),
				DefaultDatastoreID: l.Destination.Storage[0].DatastoreID,
				AcceptAllEULA:      true,
			},
			Target: l.Placement,
		}

		// Create library vmtx item from source VM
		storage := &DiskStorage{
			Datastore: deploy.DeploymentSpec.DefaultDatastoreID,
		}
		spec := Template{
			Name:          item.Name,
			Library:       l.Destination.ID,
			DiskStorage:   storage,
			VMHomeStorage: storage,
			Placement: &Placement{
				Folder:       deploy.Target.FolderID,
				ResourcePool: deploy.Target.ResourcePoolID,
			},
		}

		if !l.Include(item, existing[item.Name]) {
			continue
		}

		if err = sync(ctx, item, &deploy, &spec); err != nil {
			return err
		}
	}

	return nil
}
