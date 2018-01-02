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

package datastore

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type remove struct {
	*flags.HostSystemFlag
	*flags.DatastoreFlag
}

func init() {
	cli.Register("datastore.remove", &remove{})
}

func (cmd *remove) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)
}

func (cmd *remove) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *remove) Usage() string {
	return "HOST..."
}

func (cmd *remove) Description() string {
	return `Remove datastore from HOST.

Examples:
  govc datastore.remove -ds nfsDatastore cluster1
  govc datastore.remove -ds nasDatastore host1 host2 host3`
}

func (cmd *remove) Run(ctx context.Context, f *flag.FlagSet) error {
	ds, err := cmd.Datastore()
	if err != nil {
		return err
	}

	hosts, err := cmd.HostSystems(f.Args())
	if err != nil {
		return err
	}

	for _, host := range hosts {
		hds, err := host.ConfigManager().DatastoreSystem(ctx)
		if err != nil {
			return err
		}

		err = hds.Remove(ctx, ds)
		if err != nil {
			return err
		}
	}

	return nil
}
