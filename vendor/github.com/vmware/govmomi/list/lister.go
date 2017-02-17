/*
Copyright (c) 2014-2016 VMware, Inc. All Rights Reserved.

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

package list

import (
	"context"
	"fmt"
	"path"
	"reflect"

	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type Element struct {
	Path   string
	Object mo.Reference
}

func ToElement(r mo.Reference, prefix string) Element {
	var name string

	// Comments about types to be expected in folders copied from the
	// documentation of the Folder managed object:
	// http://pubs.vmware.com/vsphere-55/topic/com.vmware.wssdk.apiref.doc/vim.Folder.html
	switch m := r.(type) {
	case mo.Folder:
		name = m.Name
	case mo.StoragePod:
		name = m.Name

	// { "vim.Datacenter" } - Identifies the root folder and its descendant
	// folders. Data center folders can contain child data center folders and
	// Datacenter managed objects. Datacenter objects contain virtual machine,
	// compute resource, network entity, and datastore folders.
	case mo.Datacenter:
		name = m.Name

	// { "vim.Virtualmachine", "vim.VirtualApp" } - Identifies a virtual machine
	// folder. A virtual machine folder may contain child virtual machine
	// folders. It also can contain VirtualMachine managed objects, templates,
	// and VirtualApp managed objects.
	case mo.VirtualMachine:
		name = m.Name
	case mo.VirtualApp:
		name = m.Name

	// { "vim.ComputeResource" } - Identifies a compute resource
	// folder, which contains child compute resource folders and ComputeResource
	// hierarchies.
	case mo.ComputeResource:
		name = m.Name
	case mo.ClusterComputeResource:
		name = m.Name
	case mo.HostSystem:
		name = m.Name
	case mo.ResourcePool:
		name = m.Name

	// { "vim.Network" } - Identifies a network entity folder.
	// Network entity folders on a vCenter Server can contain Network,
	// DistributedVirtualSwitch, and DistributedVirtualPortgroup managed objects.
	// Network entity folders on an ESXi host can contain only Network objects.
	case mo.Network:
		name = m.Name
	case mo.OpaqueNetwork:
		name = m.Name
	case mo.DistributedVirtualSwitch:
		name = m.Name
	case mo.DistributedVirtualPortgroup:
		name = m.Name
	case mo.VmwareDistributedVirtualSwitch:
		name = m.Name

	// { "vim.Datastore" } - Identifies a datastore folder. Datastore folders can
	// contain child datastore folders and Datastore managed objects.
	case mo.Datastore:
		name = m.Name

	default:
		panic("not implemented for type " + reflect.TypeOf(r).String())
	}

	e := Element{
		Path:   path.Join(prefix, name),
		Object: r,
	}

	return e
}

type Lister struct {
	Collector *property.Collector
	Reference types.ManagedObjectReference
	Prefix    string
	All       bool
}

func traversable(ref types.ManagedObjectReference) bool {
	switch ref.Type {
	case "Folder":
	case "Datacenter":
	case "ComputeResource", "ClusterComputeResource":
		// Treat ComputeResource and ClusterComputeResource as one and the same.
		// It doesn't matter from the perspective of the lister.
	case "HostSystem":
	case "VirtualApp":
	case "StoragePod":
	default:
		return false
	}

	return true
}

func (l Lister) retrieveProperties(ctx context.Context, req types.RetrieveProperties, dst *[]interface{}) error {
	res, err := l.Collector.RetrieveProperties(ctx, req)
	if err != nil {
		return err
	}

	// Instead of using mo.LoadRetrievePropertiesResponse, use a custom loop to
	// iterate over the results and ignore entries that have properties that
	// could not be retrieved (a non-empty `missingSet` property). Since the
	// returned objects are enumerated by vSphere in the first place, any object
	// that has a non-empty `missingSet` property is indicative of a race
	// condition in vSphere where the object was enumerated initially, but was
	// removed before its properties could be collected.
	for _, p := range res.Returnval {
		v, err := mo.ObjectContentToType(p)
		if err != nil {
			// Ignore fault if it is ManagedObjectNotFound
			if soap.IsVimFault(err) {
				switch soap.ToVimFault(err).(type) {
				case *types.ManagedObjectNotFound:
					continue
				}
			}

			return err
		}

		*dst = append(*dst, v)
	}

	return nil
}

func (l Lister) List(ctx context.Context) ([]Element, error) {
	switch l.Reference.Type {
	case "Folder", "StoragePod":
		return l.ListFolder(ctx)
	case "Datacenter":
		return l.ListDatacenter(ctx)
	case "ComputeResource", "ClusterComputeResource":
		// Treat ComputeResource and ClusterComputeResource as one and the same.
		// It doesn't matter from the perspective of the lister.
		return l.ListComputeResource(ctx)
	case "ResourcePool":
		return l.ListResourcePool(ctx)
	case "HostSystem":
		return l.ListHostSystem(ctx)
	case "VirtualApp":
		return l.ListVirtualApp(ctx)
	default:
		return nil, fmt.Errorf("cannot traverse type " + l.Reference.Type)
	}
}

func (l Lister) ListFolder(ctx context.Context) ([]Element, error) {
	spec := types.PropertyFilterSpec{
		ObjectSet: []types.ObjectSpec{
			{
				Obj: l.Reference,
				SelectSet: []types.BaseSelectionSpec{
					&types.TraversalSpec{
						Path: "childEntity",
						Skip: types.NewBool(false),
						Type: "Folder",
					},
				},
				Skip: types.NewBool(true),
			},
		},
	}

	// Retrieve all objects that we can deal with
	childTypes := []string{
		"Folder",
		"Datacenter",
		"VirtualApp",
		"VirtualMachine",
		"Network",
		"ComputeResource",
		"ClusterComputeResource",
		"Datastore",
		"DistributedVirtualSwitch",
	}

	for _, t := range childTypes {
		pspec := types.PropertySpec{
			Type: t,
		}

		if l.All {
			pspec.All = types.NewBool(true)
		} else {
			pspec.PathSet = []string{"name"}

			// Additional basic properties.
			switch t {
			case "ComputeResource", "ClusterComputeResource":
				// The ComputeResource and ClusterComputeResource are dereferenced in
				// the ResourcePoolFlag. Make sure they always have their resourcePool
				// field populated.
				pspec.PathSet = append(pspec.PathSet, "resourcePool")
			}
		}

		spec.PropSet = append(spec.PropSet, pspec)
	}

	req := types.RetrieveProperties{
		SpecSet: []types.PropertyFilterSpec{spec},
	}

	var dst []interface{}

	err := l.retrieveProperties(ctx, req, &dst)
	if err != nil {
		return nil, err
	}

	es := []Element{}
	for _, v := range dst {
		es = append(es, ToElement(v.(mo.Reference), l.Prefix))
	}

	return es, nil
}

func (l Lister) ListDatacenter(ctx context.Context) ([]Element, error) {
	ospec := types.ObjectSpec{
		Obj:  l.Reference,
		Skip: types.NewBool(true),
	}

	// Include every datastore folder in the select set
	fields := []string{
		"vmFolder",
		"hostFolder",
		"datastoreFolder",
		"networkFolder",
	}

	for _, f := range fields {
		tspec := types.TraversalSpec{
			Path: f,
			Skip: types.NewBool(false),
			Type: "Datacenter",
		}

		ospec.SelectSet = append(ospec.SelectSet, &tspec)
	}

	pspec := types.PropertySpec{
		Type: "Folder",
	}

	if l.All {
		pspec.All = types.NewBool(true)
	} else {
		pspec.PathSet = []string{"name"}
	}

	req := types.RetrieveProperties{
		SpecSet: []types.PropertyFilterSpec{
			{
				ObjectSet: []types.ObjectSpec{ospec},
				PropSet:   []types.PropertySpec{pspec},
			},
		},
	}

	var dst []interface{}

	err := l.retrieveProperties(ctx, req, &dst)
	if err != nil {
		return nil, err
	}

	es := []Element{}
	for _, v := range dst {
		es = append(es, ToElement(v.(mo.Reference), l.Prefix))
	}

	return es, nil
}

func (l Lister) ListComputeResource(ctx context.Context) ([]Element, error) {
	ospec := types.ObjectSpec{
		Obj:  l.Reference,
		Skip: types.NewBool(true),
	}

	fields := []string{
		"host",
		"resourcePool",
	}

	for _, f := range fields {
		tspec := types.TraversalSpec{
			Path: f,
			Skip: types.NewBool(false),
			Type: "ComputeResource",
		}

		ospec.SelectSet = append(ospec.SelectSet, &tspec)
	}

	childTypes := []string{
		"HostSystem",
		"ResourcePool",
	}

	var pspecs []types.PropertySpec
	for _, t := range childTypes {
		pspec := types.PropertySpec{
			Type: t,
		}

		if l.All {
			pspec.All = types.NewBool(true)
		} else {
			pspec.PathSet = []string{"name"}
		}

		pspecs = append(pspecs, pspec)
	}

	req := types.RetrieveProperties{
		SpecSet: []types.PropertyFilterSpec{
			{
				ObjectSet: []types.ObjectSpec{ospec},
				PropSet:   pspecs,
			},
		},
	}

	var dst []interface{}

	err := l.retrieveProperties(ctx, req, &dst)
	if err != nil {
		return nil, err
	}

	es := []Element{}
	for _, v := range dst {
		es = append(es, ToElement(v.(mo.Reference), l.Prefix))
	}

	return es, nil
}

func (l Lister) ListResourcePool(ctx context.Context) ([]Element, error) {
	ospec := types.ObjectSpec{
		Obj:  l.Reference,
		Skip: types.NewBool(true),
	}

	fields := []string{
		"resourcePool",
	}

	for _, f := range fields {
		tspec := types.TraversalSpec{
			Path: f,
			Skip: types.NewBool(false),
			Type: "ResourcePool",
		}

		ospec.SelectSet = append(ospec.SelectSet, &tspec)
	}

	childTypes := []string{
		"ResourcePool",
	}

	var pspecs []types.PropertySpec
	for _, t := range childTypes {
		pspec := types.PropertySpec{
			Type: t,
		}

		if l.All {
			pspec.All = types.NewBool(true)
		} else {
			pspec.PathSet = []string{"name"}
		}

		pspecs = append(pspecs, pspec)
	}

	req := types.RetrieveProperties{
		SpecSet: []types.PropertyFilterSpec{
			{
				ObjectSet: []types.ObjectSpec{ospec},
				PropSet:   pspecs,
			},
		},
	}

	var dst []interface{}

	err := l.retrieveProperties(ctx, req, &dst)
	if err != nil {
		return nil, err
	}

	es := []Element{}
	for _, v := range dst {
		es = append(es, ToElement(v.(mo.Reference), l.Prefix))
	}

	return es, nil
}

func (l Lister) ListHostSystem(ctx context.Context) ([]Element, error) {
	ospec := types.ObjectSpec{
		Obj:  l.Reference,
		Skip: types.NewBool(true),
	}

	fields := []string{
		"datastore",
		"network",
		"vm",
	}

	for _, f := range fields {
		tspec := types.TraversalSpec{
			Path: f,
			Skip: types.NewBool(false),
			Type: "HostSystem",
		}

		ospec.SelectSet = append(ospec.SelectSet, &tspec)
	}

	childTypes := []string{
		"Datastore",
		"Network",
		"VirtualMachine",
	}

	var pspecs []types.PropertySpec
	for _, t := range childTypes {
		pspec := types.PropertySpec{
			Type: t,
		}

		if l.All {
			pspec.All = types.NewBool(true)
		} else {
			pspec.PathSet = []string{"name"}
		}

		pspecs = append(pspecs, pspec)
	}

	req := types.RetrieveProperties{
		SpecSet: []types.PropertyFilterSpec{
			{
				ObjectSet: []types.ObjectSpec{ospec},
				PropSet:   pspecs,
			},
		},
	}

	var dst []interface{}

	err := l.retrieveProperties(ctx, req, &dst)
	if err != nil {
		return nil, err
	}

	es := []Element{}
	for _, v := range dst {
		es = append(es, ToElement(v.(mo.Reference), l.Prefix))
	}

	return es, nil
}

func (l Lister) ListVirtualApp(ctx context.Context) ([]Element, error) {
	ospec := types.ObjectSpec{
		Obj:  l.Reference,
		Skip: types.NewBool(true),
	}

	fields := []string{
		"resourcePool",
		"vm",
	}

	for _, f := range fields {
		tspec := types.TraversalSpec{
			Path: f,
			Skip: types.NewBool(false),
			Type: "VirtualApp",
		}

		ospec.SelectSet = append(ospec.SelectSet, &tspec)
	}

	childTypes := []string{
		"ResourcePool",
		"VirtualMachine",
	}

	var pspecs []types.PropertySpec
	for _, t := range childTypes {
		pspec := types.PropertySpec{
			Type: t,
		}

		if l.All {
			pspec.All = types.NewBool(true)
		} else {
			pspec.PathSet = []string{"name"}
		}

		pspecs = append(pspecs, pspec)
	}

	req := types.RetrieveProperties{
		SpecSet: []types.PropertyFilterSpec{
			{
				ObjectSet: []types.ObjectSpec{ospec},
				PropSet:   pspecs,
			},
		},
	}

	var dst []interface{}

	err := l.retrieveProperties(ctx, req, &dst)
	if err != nil {
		return nil, err
	}

	es := []Element{}
	for _, v := range dst {
		es = append(es, ToElement(v.(mo.Reference), l.Prefix))
	}

	return es, nil
}
