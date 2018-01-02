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
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
)

type add struct {
	*flags.DatacenterFlag
	*flags.HostConnectFlag

	cluster string
	connect bool
	license string
}

func init() {
	cli.Register("cluster.add", &add{})
}

func (cmd *add) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	cmd.HostConnectFlag, ctx = flags.NewHostConnectFlag(ctx)
	cmd.HostConnectFlag.Register(ctx, f)

	f.StringVar(&cmd.cluster, "cluster", "*", "Path to cluster")

	f.StringVar(&cmd.license, "license", "", "Assign license key")

	f.BoolVar(&cmd.connect, "connect", true, "Immediately connect to host")
}

func (cmd *add) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostConnectFlag.Process(ctx); err != nil {
		return err
	}
	if cmd.HostName == "" {
		return flag.ErrHelp
	}
	if cmd.UserName == "" {
		return flag.ErrHelp
	}
	if cmd.Password == "" {
		return flag.ErrHelp
	}
	return nil
}

func (cmd *add) Description() string {
	return `Add HOST to CLUSTER.

The host is added to the cluster specified by the 'cluster' flag.

Examples:
  thumbprint=$(govc about.cert -k -u host.example.com -thumbprint | awk '{print $2}')
  govc cluster.add -cluster ClusterA -hostname host.example.com -username root -password pass -thumbprint $thumbprint
  govc cluster.add -cluster ClusterB -hostname 10.0.6.1 -username root -password pass -noverify`
}

func (cmd *add) Add(ctx context.Context, cluster *object.ClusterComputeResource) error {
	spec := cmd.HostConnectSpec

	var license *string
	if cmd.license != "" {
		license = &cmd.license
	}

	task, err := cluster.AddHost(ctx, cmd.Spec(cluster.Client()), cmd.connect, license, nil)
	if err != nil {
		return err
	}

	logger := cmd.ProgressLogger(fmt.Sprintf("adding %s to cluster %s... ", spec.HostName, cluster.InventoryPath))
	defer logger.Wait()

	_, err = task.WaitForResult(ctx, logger)
	return err
}

func (cmd *add) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 0 {
		return flag.ErrHelp
	}

	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	cluster, err := finder.ClusterComputeResource(ctx, cmd.cluster)
	if err != nil {
		return err
	}

	return cmd.Fault(cmd.Add(ctx, cluster))
}
