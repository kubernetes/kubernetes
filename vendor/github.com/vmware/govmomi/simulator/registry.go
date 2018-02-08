/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"fmt"
	"reflect"
	"strings"
	"sync"

	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

// This is a map from a reference type name to a reference value name prefix.
// It's a convention that VirtualCenter follows. The map is not complete, but
// it should cover the most popular objects.
var refValueMap = map[string]string{
	"DistributedVirtualPortgroup":    "dvportgroup",
	"EnvironmentBrowser":             "envbrowser",
	"HostSystem":                     "host",
	"ResourcePool":                   "resgroup",
	"VirtualMachine":                 "vm",
	"VirtualMachineSnapshot":         "snapshot",
	"VmwareDistributedVirtualSwitch": "dvs",
}

// Map is the default Registry instance.
var Map = NewRegistry()

// RegisterObject interface supports callbacks when objects are added and removed from the Registry
type RegisterObject interface {
	mo.Reference
	PutObject(mo.Reference)
	RemoveObject(types.ManagedObjectReference)
}

// Registry manages a map of mo.Reference objects
type Registry struct {
	m        sync.Mutex
	objects  map[types.ManagedObjectReference]mo.Reference
	handlers map[types.ManagedObjectReference]RegisterObject
	counter  int
}

// NewRegistry creates a new instances of Registry
func NewRegistry() *Registry {
	r := &Registry{
		objects:  make(map[types.ManagedObjectReference]mo.Reference),
		handlers: make(map[types.ManagedObjectReference]RegisterObject),
	}

	return r
}

// typeName returns the type of the given object.
func typeName(item mo.Reference) string {
	return reflect.TypeOf(item).Elem().Name()
}

// valuePrefix returns the value name prefix of a given object
func valuePrefix(typeName string) string {
	if v, ok := refValueMap[typeName]; ok {
		return v
	}

	return strings.ToLower(typeName)
}

// newReference returns a new MOR, where Type defaults to type of the given item
// and Value defaults to a unique id for the given type.
func (r *Registry) newReference(item mo.Reference) types.ManagedObjectReference {
	ref := item.Reference()

	if ref.Type == "" {
		ref.Type = typeName(item)
	}

	if ref.Value == "" {
		r.counter++
		ref.Value = fmt.Sprintf("%s-%d", valuePrefix(ref.Type), r.counter)
	}

	return ref
}

// AddHandler adds a RegisterObject handler to the Registry.
func (r *Registry) AddHandler(h RegisterObject) {
	r.handlers[h.Reference()] = h
}

// NewEntity sets Entity().Self with a new, unique Value.
// Useful for creating object instances from templates.
func (r *Registry) NewEntity(item mo.Entity) mo.Entity {
	e := item.Entity()
	e.Self.Value = ""
	e.Self = r.newReference(item)
	return item
}

// PutEntity sets item.Parent to that of parent.Self before adding item to the Registry.
func (r *Registry) PutEntity(parent mo.Entity, item mo.Entity) mo.Entity {
	e := item.Entity()

	if parent != nil {
		e.Parent = &parent.Entity().Self
	}

	r.Put(item)

	return item
}

// Get returns the object for the given reference.
func (r *Registry) Get(ref types.ManagedObjectReference) mo.Reference {
	r.m.Lock()
	defer r.m.Unlock()

	return r.objects[ref]
}

// Any returns the first instance of entity type specified by kind.
func (r *Registry) Any(kind string) mo.Entity {
	r.m.Lock()
	defer r.m.Unlock()

	for ref, val := range r.objects {
		if ref.Type == kind {
			return val.(mo.Entity)
		}
	}

	return nil
}

// Put adds a new object to Registry, generating a ManagedObjectReference if not already set.
func (r *Registry) Put(item mo.Reference) mo.Reference {
	r.m.Lock()
	defer r.m.Unlock()

	ref := item.Reference()
	if ref.Type == "" || ref.Value == "" {
		ref = r.newReference(item)
		// mo.Reference() returns a value, not a pointer so use reflect to set the Self field
		reflect.ValueOf(item).Elem().FieldByName("Self").Set(reflect.ValueOf(ref))
	}

	if me, ok := item.(mo.Entity); ok {
		me.Entity().ConfigStatus = types.ManagedEntityStatusGreen
		me.Entity().OverallStatus = types.ManagedEntityStatusGreen
		me.Entity().EffectiveRole = []int32{-1} // Admin
	}

	r.objects[ref] = item

	for _, h := range r.handlers {
		h.PutObject(item)
	}

	return item
}

// Remove removes an object from the Registry.
func (r *Registry) Remove(item types.ManagedObjectReference) {
	r.m.Lock()
	defer r.m.Unlock()

	for _, h := range r.handlers {
		h.RemoveObject(item)
	}

	delete(r.objects, item)
	delete(r.handlers, item)
}

// getEntityParent traverses up the inventory and returns the first object of type kind.
// If no object of type kind is found, the method will panic when it reaches the
// inventory root Folder where the Parent field is nil.
func (r *Registry) getEntityParent(item mo.Entity, kind string) mo.Entity {
	for {
		parent := item.Entity().Parent

		item = r.Get(*parent).(mo.Entity)

		if item.Reference().Type == kind {
			return item
		}
	}
}

// getEntityDatacenter returns the Datacenter containing the given item
func (r *Registry) getEntityDatacenter(item mo.Entity) *mo.Datacenter {
	return r.getEntityParent(item, "Datacenter").(*mo.Datacenter)
}

func (r *Registry) getEntityFolder(item mo.Entity, kind string) *Folder {
	dc := Map.getEntityDatacenter(item)

	var ref types.ManagedObjectReference

	switch kind {
	case "datastore":
		ref = dc.DatastoreFolder
	}

	folder := r.Get(ref).(*Folder)

	// If Model was created with Folder option, use that Folder; else use top-level folder
	for _, child := range folder.ChildEntity {
		if child.Type == "Folder" {
			folder = Map.Get(child).(*Folder)
			break
		}
	}

	return folder
}

// getEntityComputeResource returns the ComputeResource parent for the given item.
// A ResourcePool for example may have N Parents of type ResourcePool, but the top
// most Parent pool is always a ComputeResource child.
func (r *Registry) getEntityComputeResource(item mo.Entity) mo.Entity {
	for {
		parent := item.Entity().Parent

		item = r.Get(*parent).(mo.Entity)

		switch item.Reference().Type {
		case "ComputeResource":
			return item
		case "ClusterComputeResource":
			return item
		}
	}
}

// FindByName returns the first mo.Entity of the given refs whose Name field is equal to the given name.
// If there is no match, nil is returned.
// This method is useful for cases where objects are required to have a unique name, such as Datastore with
// a HostStorageSystem or HostSystem within a ClusterComputeResource.
func (r *Registry) FindByName(name string, refs []types.ManagedObjectReference) mo.Entity {
	for _, ref := range refs {
		if e, ok := r.Get(ref).(mo.Entity); ok {
			if name == e.Entity().Name {
				return e
			}
		}
	}

	return nil
}

// FindReference returns the 1st match found in refs, or nil if not found.
func FindReference(refs []types.ManagedObjectReference, match ...types.ManagedObjectReference) *types.ManagedObjectReference {
	for _, ref := range refs {
		for _, m := range match {
			if ref == m {
				return &ref
			}
		}
	}

	return nil
}

// RemoveReference returns a slice with ref removed from refs
func RemoveReference(ref types.ManagedObjectReference, refs []types.ManagedObjectReference) []types.ManagedObjectReference {
	var result []types.ManagedObjectReference

	for i, r := range refs {
		if r == ref {
			result = append(result, refs[i+1:]...)
			break
		}

		result = append(result, r)
	}

	return result
}

// AddReference returns a slice with ref appended if not already in refs.
func AddReference(ref types.ManagedObjectReference, refs []types.ManagedObjectReference) []types.ManagedObjectReference {
	if FindReference(refs, ref) == nil {
		return append(refs, ref)
	}

	return refs
}

func (r *Registry) content() types.ServiceContent {
	return r.Get(methods.ServiceInstance).(*ServiceInstance).Content
}

// IsESX returns true if this Registry maps an ESX model
func (r *Registry) IsESX() bool {
	return r.content().About.ApiType == "HostAgent"
}

// IsVPX returns true if this Registry maps a VPX model
func (r *Registry) IsVPX() bool {
	return !r.IsESX()
}

// SearchIndex returns the SearchIndex singleton
func (r *Registry) SearchIndex() *SearchIndex {
	return r.Get(r.content().SearchIndex.Reference()).(*SearchIndex)
}

// FileManager returns the FileManager singleton
func (r *Registry) FileManager() *FileManager {
	return r.Get(r.content().FileManager.Reference()).(*FileManager)
}

// VirtualDiskManager returns the VirtualDiskManager singleton
func (r *Registry) VirtualDiskManager() *VirtualDiskManager {
	return r.Get(r.content().VirtualDiskManager.Reference()).(*VirtualDiskManager)
}

// ViewManager returns the ViewManager singleton
func (r *Registry) ViewManager() *ViewManager {
	return r.Get(r.content().ViewManager.Reference()).(*ViewManager)
}
