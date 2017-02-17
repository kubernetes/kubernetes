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
	"context"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
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

	// Added in 5.5
	if h.ConfigManager.VsanSystem == nil {
		return nil, ErrNotSupported
	}

	return NewHostVsanSystem(m.c, *h.ConfigManager.VsanSystem), nil
}

func (m HostConfigManager) AccountManager(ctx context.Context) (*HostAccountManager, error) {
	var h mo.HostSystem

	err := m.Properties(ctx, m.Reference(), []string{"configManager.accountManager"}, &h)
	if err != nil {
		return nil, err
	}

	ref := h.ConfigManager.AccountManager // Added in 6.0
	if ref == nil {
		// Versions < 5.5 can use the ServiceContent ref,
		// but we can only use it when connected directly to ESX.
		c := m.Client()
		if !c.IsVC() {
			ref = c.ServiceContent.AccountManager
		}

		if ref == nil {
			return nil, ErrNotSupported
		}
	}

	return NewHostAccountManager(m.c, *ref), nil
}

func (m HostConfigManager) OptionManager(ctx context.Context) (*OptionManager, error) {
	var h mo.HostSystem

	err := m.Properties(ctx, m.Reference(), []string{"configManager.advancedOption"}, &h)
	if err != nil {
		return nil, err
	}

	return NewOptionManager(m.c, *h.ConfigManager.AdvancedOption), nil
}

func (m HostConfigManager) ServiceSystem(ctx context.Context) (*HostServiceSystem, error) {
	var h mo.HostSystem

	err := m.Properties(ctx, m.Reference(), []string{"configManager.serviceSystem"}, &h)
	if err != nil {
		return nil, err
	}

	return NewHostServiceSystem(m.c, *h.ConfigManager.ServiceSystem), nil
}

func (m HostConfigManager) CertificateManager(ctx context.Context) (*HostCertificateManager, error) {
	var h mo.HostSystem

	err := m.Properties(ctx, m.Reference(), []string{"configManager.certificateManager"}, &h)
	if err != nil {
		return nil, err
	}

	// Added in 6.0
	if h.ConfigManager.CertificateManager == nil {
		return nil, ErrNotSupported
	}

	return NewHostCertificateManager(m.c, *h.ConfigManager.CertificateManager, m.Reference()), nil
}

func (m HostConfigManager) DateTimeSystem(ctx context.Context) (*HostDateTimeSystem, error) {
	var h mo.HostSystem

	err := m.Properties(ctx, m.Reference(), []string{"configManager.dateTimeSystem"}, &h)
	if err != nil {
		return nil, err
	}

	return NewHostDateTimeSystem(m.c, *h.ConfigManager.DateTimeSystem), nil
}
