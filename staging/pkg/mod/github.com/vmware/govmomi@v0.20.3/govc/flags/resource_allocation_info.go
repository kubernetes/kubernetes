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

package flags

import (
	"context"
	"flag"
	"strconv"
	"strings"

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

type ResourceAllocationFlag struct {
	cpu, mem              *types.ResourceAllocationInfo
	ExpandableReservation bool
}

func NewResourceAllocationFlag(cpu, mem *types.ResourceAllocationInfo) *ResourceAllocationFlag {
	return &ResourceAllocationFlag{cpu, mem, true}
}

func (r *ResourceAllocationFlag) Register(ctx context.Context, f *flag.FlagSet) {
	opts := []struct {
		name  string
		units string
		*types.ResourceAllocationInfo
	}{
		{"CPU", "MHz", r.cpu},
		{"Memory", "MB", r.mem},
	}

	for _, opt := range opts {
		prefix := strings.ToLower(opt.name)[:3]
		shares := (*sharesInfo)(opt.Shares)

		f.Var(NewOptionalInt64(&opt.Limit), prefix+".limit", opt.name+" limit in "+opt.units)
		f.Var(NewOptionalInt64(&opt.Reservation), prefix+".reservation", opt.name+" reservation in "+opt.units)
		if r.ExpandableReservation {
			f.Var(NewOptionalBool(&opt.ExpandableReservation), prefix+".expandable", opt.name+" expandable reservation")
		}
		f.Var(shares, prefix+".shares", opt.name+" shares level or number")
	}
}

func (s *ResourceAllocationFlag) Process(ctx context.Context) error {
	return nil
}
