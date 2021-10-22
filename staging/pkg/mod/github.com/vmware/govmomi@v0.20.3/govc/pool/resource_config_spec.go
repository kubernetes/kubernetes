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

package pool

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

func NewResourceConfigSpecFlag() *ResourceConfigSpecFlag {
	return &ResourceConfigSpecFlag{types.DefaultResourceConfigSpec(), nil}
}

type ResourceConfigSpecFlag struct {
	types.ResourceConfigSpec
	*flags.ResourceAllocationFlag
}

func (s *ResourceConfigSpecFlag) Register(ctx context.Context, f *flag.FlagSet) {
	s.ResourceAllocationFlag = flags.NewResourceAllocationFlag(&s.CpuAllocation, &s.MemoryAllocation)
	s.ResourceAllocationFlag.Register(ctx, f)
}

func (s *ResourceConfigSpecFlag) Process(ctx context.Context) error {
	return nil
}
