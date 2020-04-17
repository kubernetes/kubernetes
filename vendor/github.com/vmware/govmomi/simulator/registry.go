/*
Copyright (c) 2017-2018 VMware, Inc. All Rights Reserved.

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
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/vmware/govmomi/vim25"
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
	"DistributedVirtualSwitch":       "dvs",
}

// Map is the default Registry instance.
var Map = NewRegistry()

// RegisterObject interface supports callbacks when objects are created, updated and deleted from the Registry
type RegisterObject interface {
	mo.Reference
	PutObject(mo.Reference)
	UpdateObject(mo.Reference, []types.PropertyChange)
	RemoveObject(types.ManagedObjectReference)
}

// Registry manages a map of mo.Reference objects
type Registry struct {
	counter  int64 // Keep first to ensure 64-bit alignment
	m        sync.Mutex
	objects  map[types.ManagedObjectReference]mo.Reference
	handlers map[types.ManagedObjectReference]RegisterObject
	locks    map[types.ManagedObjectReference]sync.Locker

	Namespace string
	Path      string

	tagManager tagManager
}

// tagManager is an interface to simplify internal interaction with the vapi tag manager simulator.
type tagManager interface {
	AttachedObjects(types.VslmTagEntry) ([]types.ManagedObjectReference, types.BaseMethodFault)
	AttachedTags(id types.ManagedObjectReference) ([]types.VslmTagEntry, types.BaseMethodFault)
	AttachTag(types.ManagedObjectReference, types.VslmTagEntry) types.BaseMethodFault
	DetachTag(types.ManagedObjectReference, types.VslmTagEntry) types.BaseMethodFault
}

// NewRegistry creates a new instances of Registry
func NewRegistry() *Registry {
	r := &Registry{
		objects:  make(map[types.ManagedObjectReference]mo.Reference),
		handlers: make(map[types.ManagedObjectReference]RegisterObject),
		locks:    make(map[types.ManagedObjectReference]sync.Locker),

		Namespace: vim25.Namespace,
		Path:      vim25.Path,
	}

	return r
}

func (r *Registry) typeFunc(name string) (reflect.Type, bool) {
	if r.Namespace != "" && r.Namespace != vim25.Namespace {
		if kind, ok := defaultMapType(r.Namespace + ":" + name); ok {
			return kind, ok
		}
	}
	return defaultMapType(name)
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
		n := atomic.AddInt64(&r.counter, 1)
		ref.Value = fmt.Sprintf("%s-%d", valuePrefix(ref.Type), n)
	}

	return ref
}

func (r *Registry) setReference(item mo.Reference, ref types.ManagedObjectReference) {
	// mo.Reference() returns a value, not a pointer so use reflect to set the Self field
	reflect.ValueOf(item).Elem().FieldByName("Self").Set(reflect.ValueOf(ref))
}

// AddHandler adds a RegisterObject handler to the Registry.
func (r *Registry) AddHandler(h RegisterObject) {
	r.m.Lock()
	r.handlers[h.Reference()] = h
	r.m.Unlock()
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

// All returns all entities of type specified by kind.
// If kind is empty - all entities will be returned.
func (r *Registry) All(kind string) []mo.Entity {
	r.m.Lock()
	defer r.m.Unlock()

	var entities []mo.Entity
	for ref, val := range r.objects {
		if kind == "" || ref.Type == kind {
			if e, ok := val.(mo.Entity); ok {
				entities = append(entities, e)
			}
		}
	}

	return entities
}

// applyHandlers calls the given func for each r.handlers
func (r *Registry) applyHandlers(f func(o RegisterObject)) {
	r.m.Lock()
	handlers := make([]RegisterObject, 0, len(r.handlers))
	for _, handler := range r.handlers {
		handlers = append(handlers, handler)
	}
	r.m.Unlock()

	for i := range handlers {
		f(handlers[i])
	}
}

// Put adds a new object to Registry, generating a ManagedObjectReference if not already set.
func (r *Registry) Put(item mo.Reference) mo.Reference {
	r.m.Lock()

	ref := item.Reference()
	if ref.Type == "" || ref.Value == "" {
		ref = r.newReference(item)
		r.setReference(item, ref)
	}

	if me, ok := item.(mo.Entity); ok {
		me.Entity().ConfigStatus = types.ManagedEntityStatusGreen
		me.Entity().OverallStatus = types.ManagedEntityStatusGreen
		me.Entity().EffectiveRole = []int32{-1} // Admin
	}

	r.objects[ref] = item

	r.m.Unlock()

	r.applyHandlers(func(o RegisterObject) {
		o.PutObject(item)
	})

	return item
}

// Remove removes an object from the Registry.
func (r *Registry) Remove(item types.ManagedObjectReference) {
	r.applyHandlers(func(o RegisterObject) {
		o.RemoveObject(item)
	})

	r.m.Lock()
	delete(r.objects, item)
	delete(r.handlers, item)
	delete(r.locks, item)
	r.m.Unlock()
}

// Update dispatches object property changes to RegisterObject handlers,
// such as any PropertyCollector instances with in-progress WaitForUpdates calls.
// The changes are also applied to the given object via mo.ApplyPropertyChange,
// so there is no need to set object fields directly.
func (r *Registry) Update(obj mo.Reference, changes []types.PropertyChange) {
	for i := range changes {
		if changes[i].Op == "" {
			changes[i].Op = types.PropertyChangeOpAssign
		}
		if changes[i].Val != nil {
			rval := reflect.ValueOf(changes[i].Val)
			changes[i].Val = wrapValue(rval, rval.Type())
		}
	}

	val := getManagedObject(obj).Addr().Interface().(mo.Reference)

	mo.ApplyPropertyChange(val, changes)

	r.applyHandlers(func(o RegisterObject) {
		o.UpdateObject(val, changes)
	})
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
func (r *Registry) getEntityDatacenter(item mo.Entity) *Datacenter {
	return r.getEntityParent(item, "Datacenter").(*Datacenter)
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

// AppendReference appends the given refs to field.
func (r *Registry) AppendReference(obj mo.Reference, field *[]types.ManagedObjectReference, ref ...types.ManagedObjectReference) {
	r.WithLock(obj, func() {
		*field = append(*field, ref...)
	})
}

// AddReference appends ref to field if not already in the given field.
func (r *Registry) AddReference(obj mo.Reference, field *[]types.ManagedObjectReference, ref types.ManagedObjectReference) {
	r.WithLock(obj, func() {
		if FindReference(*field, ref) == nil {
			*field = append(*field, ref)
		}
	})
}

// RemoveReference removes ref from the given field.
func RemoveReference(field *[]types.ManagedObjectReference, ref types.ManagedObjectReference) {
	for i, r := range *field {
		if r == ref {
			*field = append((*field)[:i], (*field)[i+1:]...)
			break
		}
	}
}

// RemoveReference removes ref from the given field.
func (r *Registry) RemoveReference(obj mo.Reference, field *[]types.ManagedObjectReference, ref types.ManagedObjectReference) {
	r.WithLock(obj, func() {
		RemoveReference(field, ref)
	})
}

func (r *Registry) removeString(obj mo.Reference, field *[]string, val string) {
	r.WithLock(obj, func() {
		for i, name := range *field {
			if name == val {
				*field = append((*field)[:i], (*field)[i+1:]...)
				break
			}
		}
	})
}

func (r *Registry) content() types.ServiceContent {
	return r.Get(vim25.ServiceInstance).(*ServiceInstance).Content
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

// EventManager returns the EventManager singleton
func (r *Registry) EventManager() *EventManager {
	return r.Get(r.content().EventManager.Reference()).(*EventManager)
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

// UserDirectory returns the UserDirectory singleton
func (r *Registry) UserDirectory() *UserDirectory {
	return r.Get(r.content().UserDirectory.Reference()).(*UserDirectory)
}

// SessionManager returns the SessionManager singleton
func (r *Registry) SessionManager() *SessionManager {
	return r.Get(r.content().SessionManager.Reference()).(*SessionManager)
}

// OptionManager returns the OptionManager singleton
func (r *Registry) OptionManager() *OptionManager {
	return r.Get(r.content().Setting.Reference()).(*OptionManager)
}

// CustomFieldsManager returns CustomFieldsManager singleton
func (r *Registry) CustomFieldsManager() *CustomFieldsManager {
	return r.Get(r.content().CustomFieldsManager.Reference()).(*CustomFieldsManager)
}

func (r *Registry) MarshalJSON() ([]byte, error) {
	r.m.Lock()
	defer r.m.Unlock()

	vars := struct {
		Objects int
		Locks   int
	}{
		len(r.objects),
		len(r.locks),
	}

	return json.Marshal(vars)
}

func (r *Registry) locker(obj mo.Reference) sync.Locker {
	var ref types.ManagedObjectReference

	switch x := obj.(type) {
	case types.ManagedObjectReference:
		ref = x
		obj = r.Get(ref) // to check for sync.Locker
	case *types.ManagedObjectReference:
		ref = *x
		obj = r.Get(ref) // to check for sync.Locker
	default:
		ref = obj.Reference()
	}

	if mu, ok := obj.(sync.Locker); ok {
		return mu
	}

	r.m.Lock()
	mu, ok := r.locks[ref]
	if !ok {
		mu = new(sync.Mutex)
		r.locks[ref] = mu
	}
	r.m.Unlock()

	return mu
}

var enableLocker = os.Getenv("VCSIM_LOCKER") != "false"

// WithLock holds a lock for the given object while then given function is run.
func (r *Registry) WithLock(obj mo.Reference, f func()) {
	if enableLocker {
		mu := r.locker(obj)
		mu.Lock()
		defer mu.Unlock()
	}
	f()
}

// nopLocker can be embedded to opt-out of auto-locking (see Registry.WithLock)
type nopLocker struct{}

func (*nopLocker) Lock()   {}
func (*nopLocker) Unlock() {}
