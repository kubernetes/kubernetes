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

package simulator

import (
	"sync/atomic"

	"github.com/google/uuid"
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type ClusterComputeResource struct {
	mo.ClusterComputeResource

	ruleKey int32
}

func (c *ClusterComputeResource) RenameTask(req *types.Rename_Task) soap.HasFault {
	return RenameTask(c, req)
}

type addHost struct {
	*ClusterComputeResource

	req *types.AddHost_Task
}

func (add *addHost) Run(task *Task) (types.AnyType, types.BaseMethodFault) {
	spec := add.req.Spec

	if spec.HostName == "" {
		return nil, &types.NoHost{}
	}

	host := NewHostSystem(esx.HostSystem)
	host.Summary.Config.Name = spec.HostName
	host.Name = host.Summary.Config.Name
	if add.req.AsConnected {
		host.Runtime.ConnectionState = types.HostSystemConnectionStateConnected
	} else {
		host.Runtime.ConnectionState = types.HostSystemConnectionStateDisconnected
	}

	cr := add.ClusterComputeResource
	Map.PutEntity(cr, Map.NewEntity(host))
	host.Summary.Host = &host.Self

	cr.Host = append(cr.Host, host.Reference())
	addComputeResource(cr.Summary.GetComputeResourceSummary(), host)

	return host.Reference(), nil
}

func (c *ClusterComputeResource) AddHostTask(add *types.AddHost_Task) soap.HasFault {
	return &methods.AddHost_TaskBody{
		Res: &types.AddHost_TaskResponse{
			Returnval: NewTask(&addHost{c, add}).Run(),
		},
	}
}

func (c *ClusterComputeResource) updateRules(cfg *types.ClusterConfigInfoEx, cspec *types.ClusterConfigSpecEx) types.BaseMethodFault {
	for _, spec := range cspec.RulesSpec {
		var i int
		exists := false

		match := func(info types.BaseClusterRuleInfo) bool {
			return info.GetClusterRuleInfo().Name == spec.Info.GetClusterRuleInfo().Name
		}

		if spec.Operation == types.ArrayUpdateOperationRemove {
			match = func(rule types.BaseClusterRuleInfo) bool {
				return rule.GetClusterRuleInfo().Key == spec.ArrayUpdateSpec.RemoveKey.(int32)
			}
		}

		for i = range cfg.Rule {
			if match(cfg.Rule[i].GetClusterRuleInfo()) {
				exists = true
				break
			}
		}

		switch spec.Operation {
		case types.ArrayUpdateOperationAdd:
			if exists {
				return new(types.InvalidArgument)
			}
			info := spec.Info.GetClusterRuleInfo()
			info.Key = atomic.AddInt32(&c.ruleKey, 1)
			info.RuleUuid = uuid.New().String()
			cfg.Rule = append(cfg.Rule, spec.Info)
		case types.ArrayUpdateOperationEdit:
			if !exists {
				return new(types.InvalidArgument)
			}
			cfg.Rule[i] = spec.Info
		case types.ArrayUpdateOperationRemove:
			if !exists {
				return new(types.InvalidArgument)
			}
			cfg.Rule = append(cfg.Rule[:i], cfg.Rule[i+1:]...)
		}
	}

	return nil
}

func (c *ClusterComputeResource) updateGroups(cfg *types.ClusterConfigInfoEx, cspec *types.ClusterConfigSpecEx) types.BaseMethodFault {
	for _, spec := range cspec.GroupSpec {
		var i int
		exists := false

		match := func(info types.BaseClusterGroupInfo) bool {
			return info.GetClusterGroupInfo().Name == spec.Info.GetClusterGroupInfo().Name
		}

		if spec.Operation == types.ArrayUpdateOperationRemove {
			match = func(info types.BaseClusterGroupInfo) bool {
				return info.GetClusterGroupInfo().Name == spec.ArrayUpdateSpec.RemoveKey.(string)
			}
		}

		for i = range cfg.Group {
			if match(cfg.Group[i].GetClusterGroupInfo()) {
				exists = true
				break
			}
		}

		switch spec.Operation {
		case types.ArrayUpdateOperationAdd:
			if exists {
				return new(types.InvalidArgument)
			}
			cfg.Group = append(cfg.Group, spec.Info)
		case types.ArrayUpdateOperationEdit:
			if !exists {
				return new(types.InvalidArgument)
			}
			cfg.Group[i] = spec.Info
		case types.ArrayUpdateOperationRemove:
			if !exists {
				return new(types.InvalidArgument)
			}
			cfg.Group = append(cfg.Group[:i], cfg.Group[i+1:]...)
		}
	}

	return nil
}

func (c *ClusterComputeResource) updateOverridesDAS(cfg *types.ClusterConfigInfoEx, cspec *types.ClusterConfigSpecEx) types.BaseMethodFault {
	for _, spec := range cspec.DasVmConfigSpec {
		var i int
		var key types.ManagedObjectReference
		exists := false

		if spec.Operation == types.ArrayUpdateOperationRemove {
			key = spec.RemoveKey.(types.ManagedObjectReference)
		} else {
			key = spec.Info.Key
		}

		for i = range cfg.DasVmConfig {
			if cfg.DasVmConfig[i].Key == key {
				exists = true
				break
			}
		}

		switch spec.Operation {
		case types.ArrayUpdateOperationAdd:
			if exists {
				return new(types.InvalidArgument)
			}
			cfg.DasVmConfig = append(cfg.DasVmConfig, *spec.Info)
		case types.ArrayUpdateOperationEdit:
			if !exists {
				return new(types.InvalidArgument)
			}
			src := spec.Info.DasSettings
			if src == nil {
				return new(types.InvalidArgument)
			}
			dst := cfg.DasVmConfig[i].DasSettings
			if src.RestartPriority != "" {
				dst.RestartPriority = src.RestartPriority
			}
			if src.RestartPriorityTimeout != 0 {
				dst.RestartPriorityTimeout = src.RestartPriorityTimeout
			}
		case types.ArrayUpdateOperationRemove:
			if !exists {
				return new(types.InvalidArgument)
			}
			cfg.DasVmConfig = append(cfg.DasVmConfig[:i], cfg.DasVmConfig[i+1:]...)
		}
	}

	return nil
}

func (c *ClusterComputeResource) updateOverridesDRS(cfg *types.ClusterConfigInfoEx, cspec *types.ClusterConfigSpecEx) types.BaseMethodFault {
	for _, spec := range cspec.DrsVmConfigSpec {
		var i int
		var key types.ManagedObjectReference
		exists := false

		if spec.Operation == types.ArrayUpdateOperationRemove {
			key = spec.RemoveKey.(types.ManagedObjectReference)
		} else {
			key = spec.Info.Key
		}

		for i = range cfg.DrsVmConfig {
			if cfg.DrsVmConfig[i].Key == key {
				exists = true
				break
			}
		}

		switch spec.Operation {
		case types.ArrayUpdateOperationAdd:
			if exists {
				return new(types.InvalidArgument)
			}
			cfg.DrsVmConfig = append(cfg.DrsVmConfig, *spec.Info)
		case types.ArrayUpdateOperationEdit:
			if !exists {
				return new(types.InvalidArgument)
			}
			if spec.Info.Enabled != nil {
				cfg.DrsVmConfig[i].Enabled = spec.Info.Enabled
			}
			if spec.Info.Behavior != "" {
				cfg.DrsVmConfig[i].Behavior = spec.Info.Behavior
			}
		case types.ArrayUpdateOperationRemove:
			if !exists {
				return new(types.InvalidArgument)
			}
			cfg.DrsVmConfig = append(cfg.DrsVmConfig[:i], cfg.DrsVmConfig[i+1:]...)
		}
	}

	return nil
}

func (c *ClusterComputeResource) ReconfigureComputeResourceTask(req *types.ReconfigureComputeResource_Task) soap.HasFault {
	task := CreateTask(c, "reconfigureCluster", func(*Task) (types.AnyType, types.BaseMethodFault) {
		spec, ok := req.Spec.(*types.ClusterConfigSpecEx)
		if !ok {
			return nil, new(types.InvalidArgument)
		}

		updates := []func(*types.ClusterConfigInfoEx, *types.ClusterConfigSpecEx) types.BaseMethodFault{
			c.updateRules,
			c.updateGroups,
			c.updateOverridesDAS,
			c.updateOverridesDRS,
		}

		for _, update := range updates {
			if err := update(c.ConfigurationEx.(*types.ClusterConfigInfoEx), spec); err != nil {
				return nil, err
			}
		}

		return nil, nil
	})

	return &methods.ReconfigureComputeResource_TaskBody{
		Res: &types.ReconfigureComputeResource_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func CreateClusterComputeResource(f *Folder, name string, spec types.ClusterConfigSpecEx) (*ClusterComputeResource, types.BaseMethodFault) {
	if e := Map.FindByName(name, f.ChildEntity); e != nil {
		return nil, &types.DuplicateName{
			Name:   e.Entity().Name,
			Object: e.Reference(),
		}
	}

	cluster := &ClusterComputeResource{}
	cluster.EnvironmentBrowser = newEnvironmentBrowser()
	cluster.Name = name
	cluster.Summary = &types.ClusterComputeResourceSummary{
		UsageSummary: new(types.ClusterUsageSummary),
	}

	config := &types.ClusterConfigInfoEx{}
	cluster.ConfigurationEx = config

	config.VmSwapPlacement = string(types.VirtualMachineConfigInfoSwapPlacementTypeVmDirectory)
	config.DrsConfig.Enabled = types.NewBool(true)

	pool := NewResourcePool()
	Map.PutEntity(cluster, Map.NewEntity(pool))
	cluster.ResourcePool = &pool.Self

	f.putChild(cluster)
	pool.Owner = cluster.Self

	return cluster, nil
}
