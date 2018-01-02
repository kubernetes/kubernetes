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

package flags

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

type DatacenterFlag struct {
	common

	*ClientFlag
	*OutputFlag

	path   string
	dc     *object.Datacenter
	finder *find.Finder
	err    error
}

var datacenterFlagKey = flagKey("datacenter")

func NewDatacenterFlag(ctx context.Context) (*DatacenterFlag, context.Context) {
	if v := ctx.Value(datacenterFlagKey); v != nil {
		return v.(*DatacenterFlag), ctx
	}

	v := &DatacenterFlag{}
	v.ClientFlag, ctx = NewClientFlag(ctx)
	v.OutputFlag, ctx = NewOutputFlag(ctx)
	ctx = context.WithValue(ctx, datacenterFlagKey, v)
	return v, ctx
}

func (flag *DatacenterFlag) Register(ctx context.Context, f *flag.FlagSet) {
	flag.RegisterOnce(func() {
		flag.ClientFlag.Register(ctx, f)
		flag.OutputFlag.Register(ctx, f)

		env := "GOVC_DATACENTER"
		value := os.Getenv(env)
		usage := fmt.Sprintf("Datacenter [%s]", env)
		f.StringVar(&flag.path, "dc", value, usage)
	})
}

func (flag *DatacenterFlag) Process(ctx context.Context) error {
	return flag.ProcessOnce(func() error {
		if err := flag.ClientFlag.Process(ctx); err != nil {
			return err
		}
		if err := flag.OutputFlag.Process(ctx); err != nil {
			return err
		}
		return nil
	})
}

func (flag *DatacenterFlag) Finder() (*find.Finder, error) {
	if flag.finder != nil {
		return flag.finder, nil
	}

	c, err := flag.Client()
	if err != nil {
		return nil, err
	}

	finder := find.NewFinder(c, flag.JSON || flag.Dump)

	// Datacenter is not required (ls command for example).
	// Set for relative func if dc flag is given or
	// if there is a single (default) Datacenter
	ctx := context.TODO()
	if flag.path == "" {
		flag.dc, flag.err = finder.DefaultDatacenter(ctx)
	} else {
		if flag.dc, err = finder.Datacenter(ctx, flag.path); err != nil {
			return nil, err
		}
	}

	finder.SetDatacenter(flag.dc)

	flag.finder = finder

	return flag.finder, nil
}

func (flag *DatacenterFlag) Datacenter() (*object.Datacenter, error) {
	if flag.dc != nil {
		return flag.dc, nil
	}

	_, err := flag.Finder()
	if err != nil {
		return nil, err
	}

	if flag.err != nil {
		// Should only happen if no dc is specified and len(dcs) > 1
		return nil, flag.err
	}

	return flag.dc, err
}

func (flag *DatacenterFlag) DatacenterIfSpecified() (*object.Datacenter, error) {
	if flag.path == "" {
		return nil, nil
	}
	return flag.Datacenter()
}

func (flag *DatacenterFlag) ManagedObjects(ctx context.Context, args []string) ([]types.ManagedObjectReference, error) {
	var refs []types.ManagedObjectReference

	c, err := flag.Client()
	if err != nil {
		return nil, err
	}

	if len(args) == 0 {
		refs = append(refs, c.ServiceContent.RootFolder)
		return refs, nil
	}

	finder, err := flag.Finder()
	if err != nil {
		return nil, err
	}

	for _, arg := range args {
		elements, err := finder.ManagedObjectList(ctx, arg)
		if err != nil {
			return nil, err
		}

		if len(elements) == 0 {
			return nil, fmt.Errorf("object '%s' not found", arg)
		}

		for _, e := range elements {
			refs = append(refs, e.Object.Reference())
		}
	}

	return refs, nil
}
