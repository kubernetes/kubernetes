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
	"github.com/vmware/govmomi/object"
	"golang.org/x/net/context"
)

type create struct {
	*flags.ClientFlag
}

func init() {
	cli.Register("datacenter.create", &create{})
}

func (cmd *create) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)
}

func (cmd *create) Usage() string {
	return "[DATACENTER NAME]..."
}

func (cmd *create) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *create) Run(ctx context.Context, f *flag.FlagSet) error {
	datacenters := f.Args()
	if len(datacenters) < 1 {
		return flag.ErrHelp
	}

	client, err := cmd.ClientFlag.Client()
	if err != nil {
		return err
	}

	finder := find.NewFinder(client, false)
	rootFolder := object.NewRootFolder(client)
	for _, datacenterToCreate := range datacenters {
		_, err := finder.Datacenter(context.TODO(), datacenterToCreate)
		if err == nil {
			// The datacenter was found, no need to create it
			continue
		}

		switch err.(type) {
		case *find.NotFoundError:
			_, err = rootFolder.CreateDatacenter(context.TODO(), datacenterToCreate)
			if err != nil {
				return err
			}
		default:
			return err
		}
	}

	return nil
}
