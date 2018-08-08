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
	"reflect"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type ViewManager struct {
	mo.ViewManager

	entities map[string]bool
}

var entities = []struct {
	Type      reflect.Type
	Container bool
}{
	{reflect.TypeOf((*mo.ManagedEntity)(nil)).Elem(), true},
	{reflect.TypeOf((*mo.Folder)(nil)).Elem(), true},
	{reflect.TypeOf((*mo.StoragePod)(nil)).Elem(), true},
	{reflect.TypeOf((*mo.Datacenter)(nil)).Elem(), true},
	{reflect.TypeOf((*mo.ComputeResource)(nil)).Elem(), true},
	{reflect.TypeOf((*mo.ClusterComputeResource)(nil)).Elem(), true},
	{reflect.TypeOf((*mo.HostSystem)(nil)).Elem(), true},
	{reflect.TypeOf((*mo.ResourcePool)(nil)).Elem(), true},
	{reflect.TypeOf((*mo.VirtualApp)(nil)).Elem(), true},
	{reflect.TypeOf((*mo.VirtualMachine)(nil)).Elem(), false},
	{reflect.TypeOf((*mo.Datastore)(nil)).Elem(), false},
	{reflect.TypeOf((*mo.Network)(nil)).Elem(), false},
	{reflect.TypeOf((*mo.OpaqueNetwork)(nil)).Elem(), false},
	{reflect.TypeOf((*mo.DistributedVirtualPortgroup)(nil)).Elem(), false},
	{reflect.TypeOf((*mo.DistributedVirtualSwitch)(nil)).Elem(), false},
	{reflect.TypeOf((*mo.VmwareDistributedVirtualSwitch)(nil)).Elem(), false},
}

func NewViewManager(ref types.ManagedObjectReference) object.Reference {
	s := &ViewManager{
		entities: make(map[string]bool),
	}

	s.Self = ref

	for _, e := range entities {
		s.entities[e.Type.Name()] = e.Container
	}

	return s
}

func destroyView(ref types.ManagedObjectReference) soap.HasFault {
	m := Map.ViewManager()

	RemoveReference(&m.ViewList, ref)

	return &methods.DestroyViewBody{
		Res: &types.DestroyViewResponse{},
	}
}

func (m *ViewManager) CreateContainerView(ctx *Context, req *types.CreateContainerView) soap.HasFault {
	body := &methods.CreateContainerViewBody{}

	root := Map.Get(req.Container)
	if root == nil {
		body.Fault_ = Fault("", &types.ManagedObjectNotFound{Obj: req.Container})
		return body
	}

	if m.entities[root.Reference().Type] != true {
		body.Fault_ = Fault("", &types.InvalidArgument{InvalidProperty: "container"})
		return body
	}

	container := &ContainerView{
		mo.ContainerView{
			Container: root.Reference(),
			Recursive: req.Recursive,
			Type:      req.Type,
		},
		make(map[string]bool),
	}

	for _, ctype := range container.Type {
		if _, ok := m.entities[ctype]; !ok {
			body.Fault_ = Fault("", &types.InvalidArgument{InvalidProperty: "type"})
			return body
		}

		container.types[ctype] = true

		for _, e := range entities {
			// Check for embedded types
			if f, ok := e.Type.FieldByName(ctype); ok && f.Anonymous {
				container.types[e.Type.Name()] = true
			}
		}
	}

	ctx.Session.Put(container)

	m.ViewList = append(m.ViewList, container.Reference())

	body.Res = &types.CreateContainerViewResponse{
		Returnval: container.Self,
	}

	seen := make(map[types.ManagedObjectReference]bool)
	container.add(root, seen)

	return body
}

type ContainerView struct {
	mo.ContainerView

	types map[string]bool
}

func (v *ContainerView) DestroyView(ctx *Context, c *types.DestroyView) soap.HasFault {
	ctx.Session.Remove(c.This)
	return destroyView(c.This)
}

func (v *ContainerView) include(o types.ManagedObjectReference) bool {
	if len(v.types) == 0 {
		return true
	}

	return v.types[o.Type]
}

func walk(root mo.Reference, f func(child types.ManagedObjectReference)) {
	var children []types.ManagedObjectReference

	switch e := root.(type) {
	case *Datacenter:
		children = []types.ManagedObjectReference{e.VmFolder, e.HostFolder, e.DatastoreFolder, e.NetworkFolder}
	case *Folder:
		children = e.ChildEntity
	case *mo.ComputeResource:
		children = e.Host
		children = append(children, *e.ResourcePool)
	case *ClusterComputeResource:
		children = e.Host
		children = append(children, *e.ResourcePool)
	case *ResourcePool:
		children = e.ResourcePool.ResourcePool
		children = append(children, e.Vm...)
	case *VirtualApp:
		children = e.ResourcePool.ResourcePool
		children = append(children, e.Vm...)
	case *HostSystem:
		children = e.Vm
	}

	for _, child := range children {
		f(child)
	}
}

func (v *ContainerView) add(root mo.Reference, seen map[types.ManagedObjectReference]bool) {
	walk(root, func(child types.ManagedObjectReference) {
		if v.include(child) {
			if seen[child] == false {
				seen[child] = true
				v.View = append(v.View, child)
			}
		}

		if v.Recursive {
			v.add(Map.Get(child), seen)
		}
	})
}

func (m *ViewManager) CreateListView(ctx *Context, req *types.CreateListView) soap.HasFault {
	body := new(methods.CreateListViewBody)
	list := new(ListView)

	if err := list.add(req.Obj); err != nil {
		body.Fault_ = Fault("", err)
		return body
	}

	ctx.Session.Put(list)

	body.Res = &types.CreateListViewResponse{
		Returnval: list.Self,
	}

	return body
}

type ListView struct {
	mo.ListView
}

func (v *ListView) update() {
	Map.Update(v, []types.PropertyChange{{Name: "view", Val: v.View}})
}

func (v *ListView) add(refs []types.ManagedObjectReference) *types.ManagedObjectNotFound {
	for _, ref := range refs {
		obj := Map.Get(ref)
		if obj == nil {
			return &types.ManagedObjectNotFound{Obj: ref}
		}
		v.View = append(v.View, ref)
	}
	return nil
}

func (v *ListView) DestroyView(ctx *Context, c *types.DestroyView) soap.HasFault {
	ctx.Session.Remove(c.This)
	return destroyView(c.This)
}

func (v *ListView) ModifyListView(req *types.ModifyListView) soap.HasFault {
	body := new(methods.ModifyListViewBody)

	for _, ref := range req.Remove {
		RemoveReference(&v.View, ref)
	}

	if err := v.add(req.Add); err != nil {
		body.Fault_ = Fault("", err)
		return body
	}

	body.Res = new(types.ModifyListViewResponse)

	if len(req.Remove) != 0 || len(req.Add) != 0 {
		v.update()
	}

	return body
}

func (v *ListView) ResetListView(req *types.ResetListView) soap.HasFault {
	body := new(methods.ResetListViewBody)

	v.View = nil

	if err := v.add(req.Obj); err != nil {
		body.Fault_ = Fault("", err)
		return body
	}

	body.Res = new(types.ResetListViewResponse)

	v.update()

	return body
}
