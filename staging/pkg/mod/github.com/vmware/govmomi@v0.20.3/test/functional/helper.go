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

package functional

import (
	"context"
	"testing"

	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/test"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type Helper struct {
	*testing.T

	c   *vim25.Client
	f   *find.Finder
	fns []func()
}

func NewHelper(t *testing.T) *Helper {
	h := &Helper{
		T: t,

		c:   test.NewAuthenticatedClient(t),
		fns: make([]func(), 0),
	}

	h.f = find.NewFinder(h.c, true)

	return h
}

func (h *Helper) Defer(fn func()) {
	h.fns = append(h.fns, fn)
}

func (h *Helper) Teardown() {
	for _, fn := range h.fns {
		fn()
	}
}

func (h *Helper) RequireVirtualCenter() {
	var expect = "VirtualCenter"
	var actual = h.c.ServiceContent.About.ApiType
	if actual != expect {
		h.Skipf("Requires %s, running against %s", expect, actual)
	}
}

func (h *Helper) Datacenter() *object.Datacenter {
	dc, err := h.f.DefaultDatacenter(context.Background())
	if err != nil {
		h.Fatal(err)
	}

	h.f.SetDatacenter(dc)

	return dc
}

func (h *Helper) DatacenterFolders() *object.DatacenterFolders {
	df, err := h.Datacenter().Folders(context.Background())
	if err != nil {
		h.Fatal(err)
	}

	return df
}

func (h *Helper) ComputeResource() *object.ComputeResource {
	cr, err := h.f.DefaultComputeResource(context.Background())
	if err != nil {
		h.Fatal(err)
	}

	return cr
}

func (h *Helper) LocalDatastores(ctx context.Context, cr *object.ComputeResource) ([]*object.Datastore, error) {
	// List datastores for compute resource
	dss, err := cr.Datastores(ctx)
	if err != nil {
		return nil, err
	}

	// Filter local datastores
	var ldss []*object.Datastore
	for _, ds := range dss {
		var mds mo.Datastore
		err = property.DefaultCollector(h.c).RetrieveOne(ctx, ds.Reference(), nil, &mds)
		if err != nil {
			return nil, err
		}

		switch i := mds.Info.(type) {
		case *types.VmfsDatastoreInfo:
			if i.Vmfs.Local != nil && *i.Vmfs.Local == true {
				break
			}
		default:
			continue
		}

		ds.InventoryPath = mds.Name
		ldss = append(ldss, ds)
	}

	return ldss, nil
}
