/*
Copyright (c) 2016-2017 VMware, Inc. All Rights Reserved.

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

package option

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/govc/option"
)

type set struct {
	*option.Set
	*flags.HostSystemFlag
}

func init() {
	cli.Register("host.option.set", &set{})
}

func (cmd *set) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.Set = &option.Set{}
	cmd.Set.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.Set.ClientFlag.Register(ctx, f)

	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)
}

func (cmd *set) Process(ctx context.Context) error {
	if err := cmd.Set.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *set) Description() string {
	return option.SetDescription + `

Examples:
  govc host.option.set Config.HostAgent.plugins.solo.enableMob true
  govc host.option.set Config.HostAgent.log.level verbose`
}

func (cmd *set) Run(ctx context.Context, f *flag.FlagSet) error {
	host, err := cmd.HostSystem()
	if err != nil {
		return err
	}

	m, err := host.ConfigManager().OptionManager(ctx)
	if err != nil {
		return err
	}

	return cmd.Update(ctx, f, m)
}
