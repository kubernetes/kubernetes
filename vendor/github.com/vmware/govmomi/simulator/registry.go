/*
Copyright (c) 2017-2021 VMware, Inc. All Rights Reserved.

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
	"log"
	"os"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/vmware/govmomi/simulator/internal"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
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
	"ClusterComputeResource":         "domain-c",
	"Folder":                         "group",
	"StoragePod":                     "group-p",
}

// Map is the default Registry instance.
//
// TODO/WIP: To support the eventual removal of this unsyncronized global
// variable, the Map should be accessed through any Context.Map that is passed
// in to functions that may need it.
var Map = NewRegistry()

// RegisterObject interface supports callbacks when objects are created, updated and deleted from the Registry
type RegisterObject interface {
	mo.Reference
	PutObject(mo.Reference)
	UpdateObject(mo.Reference, []types.PropertyChange)
	RemoveObject(*Context, types.ManagedObjectReference)
}

// Registry manages a map of mo.Reference objects
type Registry struct {
	counter  int64 // Keep first to ensure 64-bit alignment
	m        sync.Mutex
	objects  map[types.ManagedObjectReference]mo.Reference
	handlers map[types.ManagedObjectReference]RegisterObject
	locks    map[types.ManagedObjectReference]*internal.ObjectLock

	Namespace string
	Path      string
	Handler   func(*Context, *Method) (mo.Reference, types.BaseMethodFault)

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
		locks:    make(map[types.ManagedObjectReference]*internal.ObjectLock),

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
	v, ok := refValueMap[typeName]
	if ok {
		if strings.Contains(v, "-") {
			return v
		}
	} else {
		v = strings.ToLower(typeName)
	}

	return v + "-"
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
		ref.Value = fmt.Sprintf("%s%d", valuePrefix(ref.Type), n)
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

// RemoveHandler removes a RegisterObject handler from the Registry.
func (r *Registry) RemoveHandler(h RegisterObject) {
	r.m.Lock()
	delete(r.handlers, h.Reference())
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

// AllReference returns all mo.Reference objects of type specified by kind.
// If kind is empty - all objects will be returned.
func (r *Registry) AllReference(kind string) []mo.Reference {
	r.m.Lock()
	defer r.m.Unlock()

	var objs []mo.Reference
	for ref, val := range r.objects {
		if kind == "" || ref.Type == kind {
			objs = append(objs, val)
		}
	}

	return objs
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

func (r *Registry) reference(item mo.Reference) types.ManagedObjectReference {
	ref := item.Reference()
	if ref.Type == "" || ref.Value == "" {
		ref = r.newReference(item)
		r.setReference(item, ref)
	}
	return ref
}

// Put adds a new object to Registry, generating a ManagedObjectReference if not already set.
func (r *Registry) Put(item mo.Reference) mo.Reference {
	r.m.Lock()

	if me, ok := item.(mo.Entity); ok {
		me.Entity().ConfigStatus = types.ManagedEntityStatusGreen
		me.Entity().OverallStatus = types.ManagedEntityStatusGreen
		me.Entity().EffectiveRole = []int32{-1} // Admin
	}

	r.objects[r.reference(item)] = item

	r.m.Unlock()

	r.applyHandlers(func(o RegisterObject) {
		o.PutObject(item)
	})

	return item
}

// Remove removes an object from the Registry.
func (r *Registry) Remove(ctx *Context, item types.ManagedObjectReference) {
	r.applyHandlers(func(o RegisterObject) {
		o.RemoveObject(ctx, item)
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

func (r *Registry) AtomicUpdate(ctx *Context, obj mo.Reference, changes []types.PropertyChange) {
	r.WithLock(ctx, obj, func() {
		r.Update(obj, changes)
	})
}

// getEntityParent traverses up the inventory and returns the first object of type kind.
// If no object of type kind is found, the method will panic when it reaches the
// inventory root Folder where the Parent field is nil.
func (r *Registry) getEntityParent(item mo.Entity, kind string) mo.Entity {
	var ok bool
	for {
		parent := item.Entity().Parent

		item, ok = r.Get(*parent).(mo.Entity)
		if !ok {
			return nil
		}
		if item.Reference().Type == kind {
			return item
		}
	}
}

// getEntityDatacenter returns the Datacenter containing the given item
func (r *Registry) getEntityDatacenter(item mo.Entity) *Datacenter {
	dc, ok := r.getEntityParent(item, "Datacenter").(*Datacenter)
	if ok {
		return dc
	}
	return nil
}

func (r *Registry) getEntityFolder(item mo.Entity, kind string) *mo.Folder {
	dc := r.getEntityDatacenter(item)

	var ref types.ManagedObjectReference

	switch kind {
	case "datastore":
		ref = dc.DatastoreFolder
	}

	folder, _ := asFolderMO(r.Get(ref))

	// If Model was created with Folder option, use that Folder; else use top-level folder
	for _, child := range folder.ChildEntity {
		if child.Type == "Folder" {
			folder, _ = asFolderMO(r.Get(child))
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

func entityName(e mo.Entity) string {
	name := e.Entity().Name
	if name != "" {
		return name
	}

	obj := getManagedObject(e).Addr().Interface()

	// The types below have their own 'Name' field, so ManagedEntity.Name (me.Name) is empty.
	// See also mo.Ancestors
	switch x := obj.(type) {
	case *mo.Network:
		return x.Name
	case *mo.DistributedVirtualSwitch:
		return x.Name
	case *mo.DistributedVirtualPortgroup:
		return x.Name
	case *mo.OpaqueNetwork:
		return x.Name
	}

	log.Panicf("%T object %s does not have a Name", obj, e.Reference())
	return name
}

// FindByName returns the first mo.Entity of the given refs whose Name field is equal to the given name.
// If there is no match, nil is returned.
// This method is useful for cases where objects are required to have a unique name, such as Datastore with
// a HostStorageSystem or HostSystem within a ClusterComputeResource.
func (r *Registry) FindByName(name string, refs []types.ManagedObjectReference) mo.Entity {
	for _, ref := range refs {
		if e, ok := r.Get(ref).(mo.Entity); ok {
			if name == entityName(e) {
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
func (r *Registry) AppendReference(ctx *Context, obj mo.Reference, field *[]types.ManagedObjectReference, ref ...types.ManagedObjectReference) {
	r.WithLock(ctx, obj, func() {
		*field = append(*field, ref...)
	})
}

// AddReference appends ref to field if not already in the given field.
func (r *Registry) AddReference(ctx *Context, obj mo.Reference, field *[]types.ManagedObjectReference, ref types.ManagedObjectReference) {
	r.WithLock(ctx, obj, func() {
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
func (r *Registry) RemoveReference(ctx *Context, obj mo.Reference, field *[]types.ManagedObjectReference, ref types.ManagedObjectReference) {
	r.WithLock(ctx, obj, func() {
		RemoveReference(field, ref)
	})
}

func (r *Registry) removeString(ctx *Context, obj mo.Reference, field *[]string, val string) {
	r.WithLock(ctx, obj, func() {
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

type VirtualDiskManagerInterface interface {
	mo.Reference
	MO() mo.VirtualDiskManager
	CreateVirtualDiskTask(*Context, *types.CreateVirtualDisk_Task) soap.HasFault
	DeleteVirtualDiskTask(*Context, *types.DeleteVirtualDisk_Task) soap.HasFault
	MoveVirtualDiskTask(*Context, *types.MoveVirtualDisk_Task) soap.HasFault
	CopyVirtualDiskTask(*Context, *types.CopyVirtualDisk_Task) soap.HasFault
	QueryVirtualDiskUuid(*Context, *types.QueryVirtualDiskUuid) soap.HasFault
	SetVirtualDiskUuid(*Context, *types.SetVirtualDiskUuid) soap.HasFault
}

// VirtualDiskManager returns the VirtualDiskManager singleton
func (r *Registry) VirtualDiskManager() VirtualDiskManagerInterface {
	return r.Get(r.content().VirtualDiskManager.Reference()).(VirtualDiskManagerInterface)
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

// TenantManager returns TenantManager singleton
func (r *Registry) TenantManager() *TenantManager {
	return r.Get(r.content().TenantManager.Reference()).(*TenantManager)
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

func (r *Registry) locker(obj mo.Reference) *internal.ObjectLock {
	var ref types.ManagedObjectReference

	switch x := obj.(type) {
	case types.ManagedObjectReference:
		ref = x
		obj = r.Get(ref) // to check for sync.Locker
	case *types.ManagedObjectReference:
		ref = *x
		obj = r.Get(ref) // to check for sync.Locker
	default:
		// Use of obj.Reference() may cause a read race, prefer the mo 'Self' field to avoid this
		self := reflect.ValueOf(obj).Elem().FieldByName("Self")
		if self.IsValid() {
			ref = self.Interface().(types.ManagedObjectReference)
		} else {
			ref = obj.Reference()
		}
	}

	if mu, ok := obj.(sync.Locker); ok {
		// Objects that opt out of default locking are responsible for
		// implementing their own lock sharing, if needed. Returning
		// nil as heldBy means that WithLock will call Lock/Unlock
		// every time.
		return internal.NewObjectLock(mu)
	}

	r.m.Lock()
	mu, ok := r.locks[ref]
	if !ok {
		mu = internal.NewObjectLock(new(sync.Mutex))
		r.locks[ref] = mu
	}
	r.m.Unlock()

	return mu
}

var enableLocker = os.Getenv("VCSIM_LOCKER") != "false"

// WithLock holds a lock for the given object while then given function is run.
func (r *Registry) WithLock(onBehalfOf *Context, obj mo.Reference, f func()) {
	unlock := r.AcquireLock(onBehalfOf, obj)
	f()
	unlock()
}

// AcquireLock acquires the lock for onBehalfOf then returns. The lock MUST be
// released by calling the returned function. WithLock should be preferred
// wherever possible.
func (r *Registry) AcquireLock(onBehalfOf *Context, obj mo.Reference) func() {
	if onBehalfOf == nil {
		panic(fmt.Sprintf("Attempt to lock %v with nil onBehalfOf", obj))
	}

	if !enableLocker {
		return func() {}
	}

	l := r.locker(obj)
	l.Acquire(onBehalfOf)
	return func() {
		l.Release(onBehalfOf)
	}
}

// nopLocker can be embedded to opt-out of auto-locking (see Registry.WithLock)
type nopLocker struct{}

func (*nopLocker) Lock()   {}
func (*nopLocker) Unlock() {}
