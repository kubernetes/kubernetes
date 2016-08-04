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

package object

import (
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

type ClusterComputeResource struct {
	ComputeResource
}

func NewClusterComputeResource(c *vim25.Client, ref types.ManagedObjectReference) *ClusterComputeResource {
	return &ClusterComputeResource{
		ComputeResource: *NewComputeResource(c, ref),
	}
}

func (c ClusterComputeResource) ReconfigureCluster(ctx context.Context, spec types.ClusterConfigSpec) (*Task, error) {
	req := types.ReconfigureCluster_Task{
		This:   c.Reference(),
		Spec:   spec,
		Modify: true,
	}

	res, err := methods.ReconfigureCluster_Task(ctx, c.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(c.c, res.Returnval), nil
}

func (c ClusterComputeResource) AddHost(ctx context.Context, spec types.HostConnectSpec, asConnected bool, license *string, resourcePool *types.ManagedObjectReference) (*Task, error) {
	req := types.AddHost_Task{
		This:        c.Reference(),
		Spec:        spec,
		AsConnected: asConnected,
	}

	if license != nil {
		req.License = *license
	}

	if resourcePool != nil {
		req.ResourcePool = resourcePool
	}

	res, err := methods.AddHost_Task(ctx, c.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(c.c, res.Returnval), nil
}

func (c ClusterComputeResource) Destroy(ctx context.Context) (*Task, error) {
	req := types.Destroy_Task{
		This: c.Reference(),
	}

	res, err := methods.Destroy_Task(ctx, c.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(c.c, res.Returnval), nil
}
