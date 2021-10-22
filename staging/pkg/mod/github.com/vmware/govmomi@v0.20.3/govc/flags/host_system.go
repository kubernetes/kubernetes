/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

type HostSystemFlag struct {
	common

	*ClientFlag
	*DatacenterFlag
	*SearchFlag

	name string
	host *object.HostSystem
	pool *object.ResourcePool
}

var hostSystemFlagKey = flagKey("hostSystem")

func NewHostSystemFlag(ctx context.Context) (*HostSystemFlag, context.Context) {
	if v := ctx.Value(hostSystemFlagKey); v != nil {
		return v.(*HostSystemFlag), ctx
	}

	v := &HostSystemFlag{}
	v.ClientFlag, ctx = NewClientFlag(ctx)
	v.DatacenterFlag, ctx = NewDatacenterFlag(ctx)
	v.SearchFlag, ctx = NewSearchFlag(ctx, SearchHosts)
	ctx = context.WithValue(ctx, hostSystemFlagKey, v)
	return v, ctx
}

func (flag *HostSystemFlag) Register(ctx context.Context, f *flag.FlagSet) {
	flag.RegisterOnce(func() {
		flag.ClientFlag.Register(ctx, f)
		flag.DatacenterFlag.Register(ctx, f)
		flag.SearchFlag.Register(ctx, f)

		env := "GOVC_HOST"
		value := os.Getenv(env)
		usage := fmt.Sprintf("Host system [%s]", env)
		f.StringVar(&flag.name, "host", value, usage)
	})
}

func (flag *HostSystemFlag) Process(ctx context.Context) error {
	return flag.ProcessOnce(func() error {
		if err := flag.ClientFlag.Process(ctx); err != nil {
			return err
		}
		if err := flag.DatacenterFlag.Process(ctx); err != nil {
			return err
		}
		if err := flag.SearchFlag.Process(ctx); err != nil {
			return err
		}
		return nil
	})
}

func (flag *HostSystemFlag) HostSystemIfSpecified() (*object.HostSystem, error) {
	if flag.host != nil {
		return flag.host, nil
	}

	// Use search flags if specified.
	if flag.SearchFlag.IsSet() {
		host, err := flag.SearchFlag.HostSystem()
		if err != nil {
			return nil, err
		}

		flag.host = host
		return flag.host, nil
	}

	// Never look for a default host system.
	// A host system parameter is optional for vm creation. It uses a mandatory
	// resource pool parameter to determine where the vm should be placed.
	if flag.name == "" {
		return nil, nil
	}

	finder, err := flag.Finder()
	if err != nil {
		return nil, err
	}

	flag.host, err = finder.HostSystem(context.TODO(), flag.name)
	return flag.host, err
}

func (flag *HostSystemFlag) HostSystem() (*object.HostSystem, error) {
	host, err := flag.HostSystemIfSpecified()
	if err != nil {
		return nil, err
	}

	if host != nil {
		return host, nil
	}

	finder, err := flag.Finder()
	if err != nil {
		return nil, err
	}

	flag.host, err = finder.DefaultHostSystem(context.TODO())
	return flag.host, err
}

func (flag *HostSystemFlag) HostNetworkSystem() (*object.HostNetworkSystem, error) {
	host, err := flag.HostSystem()
	if err != nil {
		return nil, err
	}

	return host.ConfigManager().NetworkSystem(context.TODO())
}
