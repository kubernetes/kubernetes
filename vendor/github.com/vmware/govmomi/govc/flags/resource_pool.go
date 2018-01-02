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

	"github.com/vmware/govmomi/object"
)

type ResourcePoolFlag struct {
	common

	*DatacenterFlag

	name string
	pool *object.ResourcePool
}

var resourcePoolFlagKey = flagKey("resourcePool")

func NewResourcePoolFlag(ctx context.Context) (*ResourcePoolFlag, context.Context) {
	if v := ctx.Value(resourcePoolFlagKey); v != nil {
		return v.(*ResourcePoolFlag), ctx
	}

	v := &ResourcePoolFlag{}
	v.DatacenterFlag, ctx = NewDatacenterFlag(ctx)
	ctx = context.WithValue(ctx, resourcePoolFlagKey, v)
	return v, ctx
}

func (flag *ResourcePoolFlag) Register(ctx context.Context, f *flag.FlagSet) {
	flag.RegisterOnce(func() {
		flag.DatacenterFlag.Register(ctx, f)

		env := "GOVC_RESOURCE_POOL"
		value := os.Getenv(env)
		usage := fmt.Sprintf("Resource pool [%s]", env)
		f.StringVar(&flag.name, "pool", value, usage)
	})
}

func (flag *ResourcePoolFlag) Process(ctx context.Context) error {
	return flag.ProcessOnce(func() error {
		if err := flag.DatacenterFlag.Process(ctx); err != nil {
			return err
		}
		return nil
	})
}

func (flag *ResourcePoolFlag) ResourcePool() (*object.ResourcePool, error) {
	if flag.pool != nil {
		return flag.pool, nil
	}

	finder, err := flag.Finder()
	if err != nil {
		return nil, err
	}

	if flag.pool, err = finder.ResourcePoolOrDefault(context.TODO(), flag.name); err != nil {
		return nil, err
	}

	return flag.pool, nil
}

func (flag *ResourcePoolFlag) ResourcePoolIfSpecified() (*object.ResourcePool, error) {
	if flag.name == "" {
		return nil, nil
	}
	return flag.ResourcePool()
}
