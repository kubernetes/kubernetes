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

package view

import (
	"context"

	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type ContainerView struct {
	ManagedObjectView
}

func NewContainerView(c *vim25.Client, ref types.ManagedObjectReference) *ContainerView {
	return &ContainerView{
		ManagedObjectView: *NewManagedObjectView(c, ref),
	}
}

// Retrieve populates dst as property.Collector.Retrieve does, for all entities in the view of types specified by kind.
func (v ContainerView) Retrieve(ctx context.Context, kind []string, ps []string, dst interface{}) error {
	pc := property.DefaultCollector(v.Client())

	ospec := types.ObjectSpec{
		Obj:  v.Reference(),
		Skip: types.NewBool(true),
		SelectSet: []types.BaseSelectionSpec{
			&types.TraversalSpec{
				Type: v.Reference().Type,
				Path: "view",
			},
		},
	}

	var pspec []types.PropertySpec

	if len(kind) == 0 {
		kind = []string{"ManagedEntity"}
	}

	for _, t := range kind {
		spec := types.PropertySpec{
			Type: t,
		}

		if len(ps) == 0 {
			spec.All = types.NewBool(true)
		} else {
			spec.PathSet = ps
		}

		pspec = append(pspec, spec)
	}

	req := types.RetrieveProperties{
		SpecSet: []types.PropertyFilterSpec{
			{
				ObjectSet: []types.ObjectSpec{ospec},
				PropSet:   pspec,
			},
		},
	}

	res, err := pc.RetrieveProperties(ctx, req)
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
func (v ContainerView) RetrieveWithFilter(ctx context.Context, kind []string, ps []string, dst interface{}, filter property.Filter) error {
	if len(filter) == 0 {
		return v.Retrieve(ctx, kind, ps, dst)
	}

	var content []types.ObjectContent

	err := v.Retrieve(ctx, kind, filter.Keys(), &content)
	if err != nil {
		return err
	}

	objs := filter.MatchObjectContent(content)

	pc := property.DefaultCollector(v.Client())

	return pc.Retrieve(ctx, objs, ps, dst)
}

// Find returns object references for entities of type kind, matching the given filter.
func (v ContainerView) Find(ctx context.Context, kind []string, filter property.Filter) ([]types.ManagedObjectReference, error) {
	if len(filter) == 0 {
		// Ensure we have at least 1 filter to avoid retrieving all properties.
		filter = property.Filter{"name": "*"}
	}

	var content []types.ObjectContent

	err := v.Retrieve(ctx, kind, filter.Keys(), &content)
	if err != nil {
		return nil, err
	}

	return filter.MatchObjectContent(content), nil
}
