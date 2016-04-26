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
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

type HostConfigManager struct {
	Common
}

func NewHostConfigManager(c *vim25.Client, ref types.ManagedObjectReference) *HostConfigManager {
	return &HostConfigManager{
		Common: NewCommon(c, ref),
	}
}

func (m HostConfigManager) DatastoreSystem(ctx context.Context) (*HostDatastoreSystem, error) {
	var h mo.HostSystem

	err := m.Properties(ctx, m.Reference(), []string{"configManager.datastoreSystem"}, &h)
	if err != nil {
		return nil, err
	}

	return NewHostDatastoreSystem(m.c, *h.ConfigManager.DatastoreSystem), nil
}

func (m HostConfigManager) NetworkSystem(ctx context.Context) (*HostNetworkSystem, error) {
	var h mo.HostSystem

	err := m.Properties(ctx, m.Reference(), []string{"configManager.networkSystem"}, &h)
	if err != nil {
		return nil, err
	}

	return NewHostNetworkSystem(m.c, *h.ConfigManager.NetworkSystem), nil
}

func (m HostConfigManager) FirewallSystem(ctx context.Context) (*HostFirewallSystem, error) {
	var h mo.HostSystem

	err := m.Properties(ctx, m.Reference(), []string{"configManager.firewallSystem"}, &h)
	if err != nil {
		return nil, err
	}

	return NewHostFirewallSystem(m.c, *h.ConfigManager.FirewallSystem), nil
}

func (m HostConfigManager) StorageSystem(ctx context.Context) (*HostStorageSystem, error) {
	var h mo.HostSystem

	err := m.Properties(ctx, m.Reference(), []string{"configManager.storageSystem"}, &h)
	if err != nil {
		return nil, err
	}

	return NewHostStorageSystem(m.c, *h.ConfigManager.StorageSystem), nil
}

func (m HostConfigManager) VirtualNicManager(ctx context.Context) (*HostVirtualNicManager, error) {
	var h mo.HostSystem

	err := m.Properties(ctx, m.Reference(), []string{"configManager.virtualNicManager"}, &h)
	if err != nil {
		return nil, err
	}

	return NewHostVirtualNicManager(m.c, *h.ConfigManager.VirtualNicManager, m.Reference()), nil
}

func (m HostConfigManager) VsanSystem(ctx context.Context) (*HostVsanSystem, error) {
	var h mo.HostSystem

	err := m.Properties(ctx, m.Reference(), []string{"configManager.vsanSystem"}, &h)
	if err != nil {
		return nil, err
	}

	return NewHostVsanSystem(m.c, *h.ConfigManager.VsanSystem), nil
}
