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
	"github.com/vmware/govmomi/vim25/types"
)

type HostNetworkSystem struct {
	Common
}

func NewHostNetworkSystem(c *vim25.Client, ref types.ManagedObjectReference) *HostNetworkSystem {
	return &HostNetworkSystem{
		Common: NewCommon(c, ref),
	}
}

// AddPortGroup wraps methods.AddPortGroup
func (o HostNetworkSystem) AddPortGroup(ctx context.Context, portgrp types.HostPortGroupSpec) error {
	req := types.AddPortGroup{
		This:    o.Reference(),
		Portgrp: portgrp,
	}

	_, err := methods.AddPortGroup(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// AddServiceConsoleVirtualNic wraps methods.AddServiceConsoleVirtualNic
func (o HostNetworkSystem) AddServiceConsoleVirtualNic(ctx context.Context, portgroup string, nic types.HostVirtualNicSpec) (string, error) {
	req := types.AddServiceConsoleVirtualNic{
		This:      o.Reference(),
		Portgroup: portgroup,
		Nic:       nic,
	}

	res, err := methods.AddServiceConsoleVirtualNic(ctx, o.c, &req)
	if err != nil {
		return "", err
	}

	return res.Returnval, nil
}

// AddVirtualNic wraps methods.AddVirtualNic
func (o HostNetworkSystem) AddVirtualNic(ctx context.Context, portgroup string, nic types.HostVirtualNicSpec) (string, error) {
	req := types.AddVirtualNic{
		This:      o.Reference(),
		Portgroup: portgroup,
		Nic:       nic,
	}

	res, err := methods.AddVirtualNic(ctx, o.c, &req)
	if err != nil {
		return "", err
	}

	return res.Returnval, nil
}

// AddVirtualSwitch wraps methods.AddVirtualSwitch
func (o HostNetworkSystem) AddVirtualSwitch(ctx context.Context, vswitchName string, spec *types.HostVirtualSwitchSpec) error {
	req := types.AddVirtualSwitch{
		This:        o.Reference(),
		VswitchName: vswitchName,
		Spec:        spec,
	}

	_, err := methods.AddVirtualSwitch(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// QueryNetworkHint wraps methods.QueryNetworkHint
func (o HostNetworkSystem) QueryNetworkHint(ctx context.Context, device []string) error {
	req := types.QueryNetworkHint{
		This:   o.Reference(),
		Device: device,
	}

	_, err := methods.QueryNetworkHint(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// RefreshNetworkSystem wraps methods.RefreshNetworkSystem
func (o HostNetworkSystem) RefreshNetworkSystem(ctx context.Context) error {
	req := types.RefreshNetworkSystem{
		This: o.Reference(),
	}

	_, err := methods.RefreshNetworkSystem(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// RemovePortGroup wraps methods.RemovePortGroup
func (o HostNetworkSystem) RemovePortGroup(ctx context.Context, pgName string) error {
	req := types.RemovePortGroup{
		This:   o.Reference(),
		PgName: pgName,
	}

	_, err := methods.RemovePortGroup(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// RemoveServiceConsoleVirtualNic wraps methods.RemoveServiceConsoleVirtualNic
func (o HostNetworkSystem) RemoveServiceConsoleVirtualNic(ctx context.Context, device string) error {
	req := types.RemoveServiceConsoleVirtualNic{
		This:   o.Reference(),
		Device: device,
	}

	_, err := methods.RemoveServiceConsoleVirtualNic(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// RemoveVirtualNic wraps methods.RemoveVirtualNic
func (o HostNetworkSystem) RemoveVirtualNic(ctx context.Context, device string) error {
	req := types.RemoveVirtualNic{
		This:   o.Reference(),
		Device: device,
	}

	_, err := methods.RemoveVirtualNic(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// RemoveVirtualSwitch wraps methods.RemoveVirtualSwitch
func (o HostNetworkSystem) RemoveVirtualSwitch(ctx context.Context, vswitchName string) error {
	req := types.RemoveVirtualSwitch{
		This:        o.Reference(),
		VswitchName: vswitchName,
	}

	_, err := methods.RemoveVirtualSwitch(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// RestartServiceConsoleVirtualNic wraps methods.RestartServiceConsoleVirtualNic
func (o HostNetworkSystem) RestartServiceConsoleVirtualNic(ctx context.Context, device string) error {
	req := types.RestartServiceConsoleVirtualNic{
		This:   o.Reference(),
		Device: device,
	}

	_, err := methods.RestartServiceConsoleVirtualNic(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// UpdateConsoleIpRouteConfig wraps methods.UpdateConsoleIpRouteConfig
func (o HostNetworkSystem) UpdateConsoleIpRouteConfig(ctx context.Context, config types.BaseHostIpRouteConfig) error {
	req := types.UpdateConsoleIpRouteConfig{
		This:   o.Reference(),
		Config: config,
	}

	_, err := methods.UpdateConsoleIpRouteConfig(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// UpdateDnsConfig wraps methods.UpdateDnsConfig
func (o HostNetworkSystem) UpdateDnsConfig(ctx context.Context, config types.BaseHostDnsConfig) error {
	req := types.UpdateDnsConfig{
		This:   o.Reference(),
		Config: config,
	}

	_, err := methods.UpdateDnsConfig(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// UpdateIpRouteConfig wraps methods.UpdateIpRouteConfig
func (o HostNetworkSystem) UpdateIpRouteConfig(ctx context.Context, config types.BaseHostIpRouteConfig) error {
	req := types.UpdateIpRouteConfig{
		This:   o.Reference(),
		Config: config,
	}

	_, err := methods.UpdateIpRouteConfig(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// UpdateIpRouteTableConfig wraps methods.UpdateIpRouteTableConfig
func (o HostNetworkSystem) UpdateIpRouteTableConfig(ctx context.Context, config types.HostIpRouteTableConfig) error {
	req := types.UpdateIpRouteTableConfig{
		This:   o.Reference(),
		Config: config,
	}

	_, err := methods.UpdateIpRouteTableConfig(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// UpdateNetworkConfig wraps methods.UpdateNetworkConfig
func (o HostNetworkSystem) UpdateNetworkConfig(ctx context.Context, config types.HostNetworkConfig, changeMode string) (*types.HostNetworkConfigResult, error) {
	req := types.UpdateNetworkConfig{
		This:       o.Reference(),
		Config:     config,
		ChangeMode: changeMode,
	}

	res, err := methods.UpdateNetworkConfig(ctx, o.c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

// UpdatePhysicalNicLinkSpeed wraps methods.UpdatePhysicalNicLinkSpeed
func (o HostNetworkSystem) UpdatePhysicalNicLinkSpeed(ctx context.Context, device string, linkSpeed *types.PhysicalNicLinkInfo) error {
	req := types.UpdatePhysicalNicLinkSpeed{
		This:      o.Reference(),
		Device:    device,
		LinkSpeed: linkSpeed,
	}

	_, err := methods.UpdatePhysicalNicLinkSpeed(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// UpdatePortGroup wraps methods.UpdatePortGroup
func (o HostNetworkSystem) UpdatePortGroup(ctx context.Context, pgName string, portgrp types.HostPortGroupSpec) error {
	req := types.UpdatePortGroup{
		This:    o.Reference(),
		PgName:  pgName,
		Portgrp: portgrp,
	}

	_, err := methods.UpdatePortGroup(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// UpdateServiceConsoleVirtualNic wraps methods.UpdateServiceConsoleVirtualNic
func (o HostNetworkSystem) UpdateServiceConsoleVirtualNic(ctx context.Context, device string, nic types.HostVirtualNicSpec) error {
	req := types.UpdateServiceConsoleVirtualNic{
		This:   o.Reference(),
		Device: device,
		Nic:    nic,
	}

	_, err := methods.UpdateServiceConsoleVirtualNic(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// UpdateVirtualNic wraps methods.UpdateVirtualNic
func (o HostNetworkSystem) UpdateVirtualNic(ctx context.Context, device string, nic types.HostVirtualNicSpec) error {
	req := types.UpdateVirtualNic{
		This:   o.Reference(),
		Device: device,
		Nic:    nic,
	}

	_, err := methods.UpdateVirtualNic(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// UpdateVirtualSwitch wraps methods.UpdateVirtualSwitch
func (o HostNetworkSystem) UpdateVirtualSwitch(ctx context.Context, vswitchName string, spec types.HostVirtualSwitchSpec) error {
	req := types.UpdateVirtualSwitch{
		This:        o.Reference(),
		VswitchName: vswitchName,
		Spec:        spec,
	}

	_, err := methods.UpdateVirtualSwitch(ctx, o.c, &req)
	if err != nil {
		return err
	}

	return nil
}
