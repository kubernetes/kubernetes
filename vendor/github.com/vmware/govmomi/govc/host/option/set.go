/*
Copyright (c) 2016 VMware, Inc. All Rights Reserved.

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
	"fmt"
	"strconv"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type set struct {
	*flags.ClientFlag
	*flags.HostSystemFlag
}

func init() {
	cli.Register("host.option.set", &set{})
}

func (cmd *set) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)
}

func (cmd *set) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *set) Usage() string {
	return "NAME VALUE"
}

func (cmd *set) Description() string {
	return `Set host option NAME to VALUE.

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

	if f.NArg() != 2 {
		return flag.ErrHelp
	}

	name := f.Arg(0)
	opts, err := m.Query(ctx, name)
	if err != nil {
		return err
	}

	if len(opts) != 1 {
		return flag.ErrHelp
	}

	val := f.Arg(1)
	var set types.AnyType

	switch x := opts[0].GetOptionValue().Value.(type) {
	case string:
		set = val
	case bool:
		set, err = strconv.ParseBool(val)
		if err != nil {
			return err
		}
	case int32:
		s, err := strconv.ParseInt(val, 10, 32)
		if err != nil {
			return err
		}
		set = s
	case int64:
		set, err = strconv.ParseInt(val, 10, 64)
		if err != nil {
			return err
		}
	default:
		return fmt.Errorf("type %T conversion not supported", x)
	}

	opts[0].GetOptionValue().Value = set

	return m.Update(ctx, opts)
}
