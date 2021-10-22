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

package license

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/license"
	"github.com/vmware/govmomi/vim25/types"
)

type assign struct {
	*flags.ClientFlag
	*flags.OutputFlag
	*flags.HostSystemFlag
	*flags.ClusterFlag

	name   string
	remove bool
}

func init() {
	cli.Register("license.assign", &assign{})
}

func (cmd *assign) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	cmd.ClusterFlag, ctx = flags.NewClusterFlag(ctx)
	cmd.ClusterFlag.Register(ctx, f)

	f.StringVar(&cmd.name, "name", "", "Display name")
	f.BoolVar(&cmd.remove, "remove", false, "Remove assignment")
}

func (cmd *assign) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return cmd.ClusterFlag.Process(ctx)
}

func (cmd *assign) Usage() string {
	return "KEY"
}

func (cmd *assign) Description() string {
	return `Assign licenses to HOST or CLUSTER.

Examples:
  govc license.assign $VCSA_LICENSE_KEY
  govc license.assign -host a_host.example.com $ESX_LICENSE_KEY
  govc license.assign -cluster a_cluster $VSAN_LICENSE_KEY`
}

func (cmd *assign) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}

	key := f.Arg(0)

	client, err := cmd.Client()
	if err != nil {
		return err
	}

	m, err := license.NewManager(client).AssignmentManager(ctx)
	if err != nil {
		return err
	}

	host, err := cmd.HostSystemIfSpecified()
	if err != nil {
		return err
	}

	var id string

	if host == nil {
		cluster, cerr := cmd.ClusterIfSpecified()
		if cerr != nil {
			return cerr
		}
		if cluster == nil {
			// Default to vCenter UUID
			id = client.ServiceContent.About.InstanceUuid
		} else {
			id = cluster.Reference().Value
		}
	} else {
		id = host.Reference().Value
	}

	if cmd.remove {
		return m.Remove(ctx, id)
	}

	info, err := m.Update(ctx, id, key, cmd.name)
	if err != nil {
		return err
	}

	return cmd.WriteResult(licenseOutput([]types.LicenseManagerLicenseInfo{*info}))
}
