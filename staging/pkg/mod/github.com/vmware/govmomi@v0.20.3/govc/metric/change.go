/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package metric

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type change struct {
	*PerformanceFlag

	level  int
	device int
}

func init() {
	cli.Register("metric.change", &change{})
}

func (cmd *change) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.PerformanceFlag, ctx = NewPerformanceFlag(ctx)
	cmd.PerformanceFlag.Register(ctx, f)

	f.IntVar(&cmd.level, "level", 0, "Level for the aggregate counter")
	f.IntVar(&cmd.device, "device-level", 0, "Level for the per device counter")
}

func (cmd *change) Usage() string {
	return "NAME..."
}

func (cmd *change) Description() string {
	return `Change counter NAME levels.

Examples:
  govc metric.change -level 1 net.bytesRx.average net.bytesTx.average`
}

func (cmd *change) Process(ctx context.Context) error {
	if err := cmd.PerformanceFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *change) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() == 0 || (cmd.level == 0 && cmd.device == 0) {
		return flag.ErrHelp
	}

	m, err := cmd.Manager(ctx)
	if err != nil {
		return err
	}

	counters, err := m.CounterInfoByName(ctx)
	if err != nil {
		return err
	}

	var mapping []types.PerformanceManagerCounterLevelMapping

	for _, name := range f.Args() {
		counter, ok := counters[name]
		if !ok {
			return cmd.ErrNotFound(name)
		}

		mapping = append(mapping, types.PerformanceManagerCounterLevelMapping{
			CounterId:      counter.Key,
			AggregateLevel: int32(cmd.level),
			PerDeviceLevel: int32(cmd.device),
		})
	}

	_, err = methods.UpdateCounterLevelMapping(ctx, m.Client(), &types.UpdateCounterLevelMapping{
		This:            m.Reference(),
		CounterLevelMap: mapping,
	})

	return err
}
