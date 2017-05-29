/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package property

import (
	"context"
	"errors"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

// Collector models the PropertyCollector managed object.
//
// For more information, see:
// http://pubs.vmware.com/vsphere-55/index.jsp#com.vmware.wssdk.apiref.doc/vmodl.query.PropertyCollector.html
//
type Collector struct {
	roundTripper soap.RoundTripper
	reference    types.ManagedObjectReference
}

// DefaultCollector returns the session's default property collector.
func DefaultCollector(c *vim25.Client) *Collector {
	p := Collector{
		roundTripper: c,
		reference:    c.ServiceContent.PropertyCollector,
	}

	return &p
}

func (p Collector) Reference() types.ManagedObjectReference {
	return p.reference
}

// Create creates a new session-specific Collector that can be used to
// retrieve property updates independent of any other Collector.
func (p *Collector) Create(ctx context.Context) (*Collector, error) {
	req := types.CreatePropertyCollector{
		This: p.Reference(),
	}

	res, err := methods.CreatePropertyCollector(ctx, p.roundTripper, &req)
	if err != nil {
		return nil, err
	}

	newp := Collector{
		roundTripper: p.roundTripper,
		reference:    res.Returnval,
	}

	return &newp, nil
}

// Destroy destroys this Collector.
func (p *Collector) Destroy(ctx context.Context) error {
	req := types.DestroyPropertyCollector{
		This: p.Reference(),
	}

	_, err := methods.DestroyPropertyCollector(ctx, p.roundTripper, &req)
	if err != nil {
		return err
	}

	p.reference = types.ManagedObjectReference{}
	return nil
}

func (p *Collector) CreateFilter(ctx context.Context, req types.CreateFilter) error {
	req.This = p.Reference()

	_, err := methods.CreateFilter(ctx, p.roundTripper, &req)
	if err != nil {
		return err
	}

	return nil
}

func (p *Collector) WaitForUpdates(ctx context.Context, v string) (*types.UpdateSet, error) {
	req := types.WaitForUpdatesEx{
		This:    p.Reference(),
		Version: v,
	}

	res, err := methods.WaitForUpdatesEx(ctx, p.roundTripper, &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (p *Collector) RetrieveProperties(ctx context.Context, req types.RetrieveProperties) (*types.RetrievePropertiesResponse, error) {
	req.This = p.Reference()
	return methods.RetrieveProperties(ctx, p.roundTripper, &req)
}

// Retrieve loads properties for a slice of managed objects. The dst argument
// must be a pointer to a []interface{}, which is populated with the instances
// of the specified managed objects, with the relevant properties filled in. If
// the properties slice is nil, all properties are loaded.
func (p *Collector) Retrieve(ctx context.Context, objs []types.ManagedObjectReference, ps []string, dst interface{}) error {
	var propSpec *types.PropertySpec
	var objectSet []types.ObjectSpec

	for _, obj := range objs {
		// Ensure that all object reference types are the same
		if propSpec == nil {
			propSpec = &types.PropertySpec{
				Type: obj.Type,
			}

			if ps == nil {
				propSpec.All = types.NewBool(true)
			} else {
				propSpec.PathSet = ps
			}
		} else {
			if obj.Type != propSpec.Type {
				return errors.New("object references must have the same type")
			}
		}

		objectSpec := types.ObjectSpec{
			Obj:  obj,
			Skip: types.NewBool(false),
		}

		objectSet = append(objectSet, objectSpec)
	}

	req := types.RetrieveProperties{
		SpecSet: []types.PropertyFilterSpec{
			{
				ObjectSet: objectSet,
				PropSet:   []types.PropertySpec{*propSpec},
			},
		},
	}

	res, err := p.RetrieveProperties(ctx, req)
	if err != nil {
		return err
	}

	if d, ok := dst.(*[]types.ObjectContent); ok {
		*d = res.Returnval
		return nil
	}

	return mo.LoadRetrievePropertiesResponse(res, dst)
}

// RetrieveWithFilter populates dst as Retrieve does, but only for entities matching the given filter.
func (p *Collector) RetrieveWithFilter(ctx context.Context, objs []types.ManagedObjectReference, ps []string, dst interface{}, filter Filter) error {
	if len(filter) == 0 {
		return p.Retrieve(ctx, objs, ps, dst)
	}

	var content []types.ObjectContent

	err := p.Retrieve(ctx, objs, filter.Keys(), &content)
	if err != nil {
		return err
	}

	objs = filter.MatchObjectContent(content)

	if len(objs) == 0 {
		return nil
	}

	return p.Retrieve(ctx, objs, ps, dst)
}

// RetrieveOne calls Retrieve with a single managed object reference.
func (p *Collector) RetrieveOne(ctx context.Context, obj types.ManagedObjectReference, ps []string, dst interface{}) error {
	var objs = []types.ManagedObjectReference{obj}
	return p.Retrieve(ctx, objs, ps, dst)
}
