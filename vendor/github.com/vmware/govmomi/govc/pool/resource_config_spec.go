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

package pool

import (
	"context"
	"flag"
	"strconv"
	"strings"

	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type sharesInfo types.SharesInfo

func (s *sharesInfo) String() string {
	return string(s.Level)
}

func (s *sharesInfo) Set(val string) error {
	switch val {
	case string(types.SharesLevelNormal), string(types.SharesLevelLow), string(types.SharesLevelHigh):
		s.Level = types.SharesLevel(val)
	default:
		n, err := strconv.Atoi(val)
		if err != nil {
			return err
		}

		s.Level = types.SharesLevelCustom
		s.Shares = int32(n)
	}

	return nil
}

func NewResourceConfigSpecFlag() *ResourceConfigSpecFlag {
	f := new(ResourceConfigSpecFlag)
	f.MemoryAllocation = new(types.ResourceAllocationInfo)
	f.CpuAllocation = new(types.ResourceAllocationInfo)

	f.SetAllocation(func(a types.BaseResourceAllocationInfo) {
		a.GetResourceAllocationInfo().Shares = new(types.SharesInfo)
	})
	return f
}

type ResourceConfigSpecFlag struct {
	types.ResourceConfigSpec
}

func (s *ResourceConfigSpecFlag) Register(ctx context.Context, f *flag.FlagSet) {
	opts := []struct {
		name  string
		units string
		types.BaseResourceAllocationInfo
	}{
		{"CPU", "MHz", s.CpuAllocation},
		{"Memory", "MB", s.MemoryAllocation},
	}

	for _, opt := range opts {
		prefix := strings.ToLower(opt.name)[:3]
		ra := opt.GetResourceAllocationInfo()
		shares := (*sharesInfo)(ra.Shares)

		f.Int64Var(&ra.Limit, prefix+".limit", 0, opt.name+" limit in "+opt.units)
		f.Int64Var(&ra.Reservation, prefix+".reservation", 0, opt.name+" reservation in "+opt.units)
		f.Var(flags.NewOptionalBool(&ra.ExpandableReservation), prefix+".expandable", opt.name+" expandable reservation")
		f.Var(shares, prefix+".shares", opt.name+" shares level or number")
	}
}

func (s *ResourceConfigSpecFlag) Process(ctx context.Context) error {
	return nil
}

func (s *ResourceConfigSpecFlag) SetAllocation(f func(types.BaseResourceAllocationInfo)) {
	for _, a := range []types.BaseResourceAllocationInfo{s.CpuAllocation, s.MemoryAllocation} {
		f(a)
	}
}
