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
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type HostVirtualNicManager struct {
	Common
	Host *HostSystem
}

func NewHostVirtualNicManager(c *vim25.Client, ref types.ManagedObjectReference, host types.ManagedObjectReference) *HostVirtualNicManager {
	return &HostVirtualNicManager{
		Common: NewCommon(c, ref),
		Host:   NewHostSystem(c, host),
	}
}

func (m HostVirtualNicManager) Info(ctx context.Context) (*types.HostVirtualNicManagerInfo, error) {
	var vnm mo.HostVirtualNicManager

	err := m.Properties(ctx, m.Reference(), []string{"info"}, &vnm)
	if err != nil {
		return nil, err
	}

	return &vnm.Info, nil
}

func (m HostVirtualNicManager) DeselectVnic(ctx context.Context, nicType string, device string) error {
	if nicType == string(types.HostVirtualNicManagerNicTypeVsan) {
		// Avoid fault.NotSupported:
		// "Error deselecting device '$device': VSAN interfaces must be deselected using vim.host.VsanSystem"
		s, err := m.Host.ConfigManager().VsanSystem(ctx)
		if err != nil {
			return err
		}

		return s.updateVnic(ctx, device, false)
	}

	req := types.DeselectVnicForNicType{
		This:    m.Reference(),
		NicType: nicType,
		Device:  device,
	}

	_, err := methods.DeselectVnicForNicType(ctx, m.Client(), &req)
	return err
}

func (m HostVirtualNicManager) SelectVnic(ctx context.Context, nicType string, device string) error {
	if nicType == string(types.HostVirtualNicManagerNicTypeVsan) {
		// Avoid fault.NotSupported:
		// "Error selecting device '$device': VSAN interfaces must be selected using vim.host.VsanSystem"
		s, err := m.Host.ConfigManager().VsanSystem(ctx)
		if err != nil {
			return err
		}

		return s.updateVnic(ctx, device, true)
	}

	req := types.SelectVnicForNicType{
		This:    m.Reference(),
		NicType: nicType,
		Device:  device,
	}

	_, err := methods.SelectVnicForNicType(ctx, m.Client(), &req)
	return err
}
