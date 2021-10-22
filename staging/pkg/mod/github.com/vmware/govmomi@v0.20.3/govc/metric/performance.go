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
	"fmt"

	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/performance"
)

type PerformanceFlag struct {
	*flags.DatacenterFlag
	*flags.OutputFlag

	m *performance.Manager

	interval int
}

func NewPerformanceFlag(ctx context.Context) (*PerformanceFlag, context.Context) {
	f := &PerformanceFlag{}
	f.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	f.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	return f, ctx
}

func (f *PerformanceFlag) Register(ctx context.Context, fs *flag.FlagSet) {
	f.DatacenterFlag.Register(ctx, fs)
	f.OutputFlag.Register(ctx, fs)

	fs.IntVar(&f.interval, "i", 0, "Interval ID")
}

func (f *PerformanceFlag) Process(ctx context.Context) error {
	if err := f.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	if err := f.OutputFlag.Process(ctx); err != nil {
		return err
	}

	return nil
}

func (f *PerformanceFlag) Manager(ctx context.Context) (*performance.Manager, error) {
	if f.m != nil {
		return f.m, nil
	}

	c, err := f.Client()
	if err != nil {
		return nil, err
	}

	f.m = performance.NewManager(c)

	f.m.Sort = true

	return f.m, err
}

func (f *PerformanceFlag) Interval(val int32) int32 {
	interval := int32(f.interval)

	if interval == 0 {
		if val == -1 {
			// realtime not supported
			return 300
		}

		return val
	}

	return interval
}

func (f *PerformanceFlag) ErrNotFound(name string) error {
	return fmt.Errorf("counter %q not found", name)
}
