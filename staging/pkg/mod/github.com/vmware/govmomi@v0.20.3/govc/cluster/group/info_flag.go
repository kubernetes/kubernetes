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

package group

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type InfoFlag struct {
	*flags.ClusterFlag

	groups []types.BaseClusterGroupInfo

	name string
}

func NewInfoFlag(ctx context.Context) (*InfoFlag, context.Context) {
	f := &InfoFlag{}
	f.ClusterFlag, ctx = flags.NewClusterFlag(ctx)
	return f, ctx
}

func (f *InfoFlag) Register(ctx context.Context, fs *flag.FlagSet) {
	f.ClusterFlag.Register(ctx, fs)

	fs.StringVar(&f.name, "name", "", "Cluster group name")
}

func (f *InfoFlag) Process(ctx context.Context) error {
	return f.ClusterFlag.Process(ctx)
}

func (f *InfoFlag) Groups(ctx context.Context) ([]types.BaseClusterGroupInfo, error) {
	if f.groups != nil {
		return f.groups, nil
	}

	cluster, err := f.Cluster()
	if err != nil {
		return nil, err
	}

	config, err := cluster.Configuration(ctx)
	if err != nil {
		return nil, err
	}

	f.groups = config.Group

	return f.groups, nil
}

type ClusterGroupInfo struct {
	info types.BaseClusterGroupInfo

	refs *[]types.ManagedObjectReference

	kind string
}

func newGroupInfo(info types.BaseClusterGroupInfo) *ClusterGroupInfo {
	group := &ClusterGroupInfo{info: info}

	switch info := info.(type) {
	case *types.ClusterHostGroup:
		group.refs = &info.Host
		group.kind = "HostSystem"
	case *types.ClusterVmGroup:
		group.refs = &info.Vm
		group.kind = "VirtualMachine"
	}

	return group
}

func (f *InfoFlag) Group(ctx context.Context) (*ClusterGroupInfo, error) {
	groups, err := f.Groups(ctx)
	if err != nil {
		return nil, err
	}

	for _, group := range groups {
		if group.GetClusterGroupInfo().Name == f.name {
			return newGroupInfo(group), nil
		}
	}

	return nil, fmt.Errorf("group %q not found", f.name)
}

func (f *InfoFlag) Apply(ctx context.Context, update types.ArrayUpdateSpec, info types.BaseClusterGroupInfo) error {
	spec := &types.ClusterConfigSpecEx{
		GroupSpec: []types.ClusterGroupSpec{
			{
				ArrayUpdateSpec: update,
				Info:            info,
			},
		},
	}

	return f.Reconfigure(ctx, spec)
}
