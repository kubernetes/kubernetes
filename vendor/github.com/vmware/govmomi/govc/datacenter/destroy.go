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

package datacenter

import (
	"flag"

	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"golang.org/x/net/context"
)

type destroy struct {
	*flags.ClientFlag
}

func init() {
	cli.Register("datacenter.destroy", &destroy{})
}

func (cmd *destroy) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)
}

func (cmd *destroy) Usage() string {
	return "[DATACENTER NAME]..."
}

func (cmd *destroy) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *destroy) Run(ctx context.Context, f *flag.FlagSet) error {
	if len(f.Args()) < 1 {
		return flag.ErrHelp
	}
	datacentersToDestroy := f.Args()

	client, err := cmd.ClientFlag.Client()
	if err != nil {
		return err
	}

	finder := find.NewFinder(client, false)
	for _, datacenterToDestroy := range datacentersToDestroy {
		foundDatacenters, err := finder.DatacenterList(context.TODO(), datacenterToDestroy)
		if err != nil {
			return err
		}
		for _, foundDatacenter := range foundDatacenters {
			task, err := foundDatacenter.Destroy(context.TODO())
			if err != nil {
				return err
			}

			if err := task.Wait(context.TODO()); err != nil {
				return err
			}
		}
	}
	return nil
}
