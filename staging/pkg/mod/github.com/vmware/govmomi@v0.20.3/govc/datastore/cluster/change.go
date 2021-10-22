/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

func DrsBehaviorUsage() string {
	drsModes := []string{
		string(types.StorageDrsPodConfigInfoBehaviorManual),
		string(types.StorageDrsPodConfigInfoBehaviorAutomated),
	}

	return "Storage DRS behavior: " + strings.Join(drsModes, ", ")
}

type change struct {
	*flags.DatacenterFlag

	types.StorageDrsConfigSpec
}

func init() {
	cli.Register("datastore.cluster.change", &change{})
}

func (cmd *change) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	cmd.PodConfigSpec = new(types.StorageDrsPodConfigSpec)

	f.Var(flags.NewOptionalBool(&cmd.PodConfigSpec.Enabled), "drs-enabled", "Enable Storage DRS")

	f.StringVar((*string)(&cmd.PodConfigSpec.DefaultVmBehavior), "drs-mode", "", DrsBehaviorUsage())
}

func (cmd *change) Usage() string {
	return "CLUSTER..."
}

func (cmd *change) Description() string {
	return `Change configuration of the given datastore clusters.

Examples:
  govc datastore.cluster.change -drs-enabled ClusterA
  govc datastore.cluster.change -drs-enabled=false ClusterB`
}

func (cmd *change) Run(ctx context.Context, f *flag.FlagSet) error {
	client, err := cmd.Client()
	if err != nil {
		return err
	}
	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	m := object.NewStorageResourceManager(client)

	for _, path := range f.Args() {
		clusters, err := finder.DatastoreClusterList(ctx, path)
		if err != nil {
			return err
		}

		for _, cluster := range clusters {
			task, err := m.ConfigureStorageDrsForPod(ctx, cluster, cmd.StorageDrsConfigSpec, true)
			if err != nil {
				return err
			}

			_, err = task.WaitForResult(ctx, nil)
			if err != nil {
				return err
			}
		}
	}

	return nil
}
