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

package cluster

import (
	"flag"

	"golang.org/x/net/context"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

type create struct {
	*flags.DatacenterFlag

	parent string

	types.ClusterConfigSpecEx
}

func init() {
	cli.Register("cluster.create", &create{})
}

func (cmd *create) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	f.StringVar(&cmd.parent, "parent", "", "Path to parent folder for the new cluster")
}

func (cmd *create) Usage() string {
	return "CLUSTER"
}

func (cmd *create) Description() string {
	return `Create CLUSTER in datacenter.

The cluster is added to the folder specified by the 'parent' flag. If not given,
this defaults to the hosts folder in the specified or default datacenter.`
}

func (cmd *create) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *create) Run(ctx context.Context, f *flag.FlagSet) error {
	var parent *object.Folder

	if f.NArg() != 1 {
		return flag.ErrHelp
	}

	if cmd.parent == "" {
		dc, err := cmd.Datacenter()
		if err != nil {
			return err
		}

		folders, err := dc.Folders(ctx)
		if err != nil {
			return err
		}

		parent = folders.HostFolder
	} else {
		finder, err := cmd.Finder()
		if err != nil {
			return err
		}

		parent, err = finder.Folder(ctx, cmd.parent)
		if err != nil {
			return err
		}
	}

	_, err := parent.CreateCluster(ctx, f.Arg(0), cmd.ClusterConfigSpecEx)
	if err != nil {
		return err
	}

	return nil
}
