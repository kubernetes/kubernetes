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
	"errors"
	"fmt"

	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

const (
	PropRuntimePowerState = "summary.runtime.powerState"
)

type VirtualMachine struct {
	Common

	InventoryPath string
}

func (v VirtualMachine) String() string {
	if v.InventoryPath == "" {
		return v.Common.String()
	}
	return fmt.Sprintf("%v @ %v", v.Common, v.InventoryPath)
}

func NewVirtualMachine(c *vim25.Client, ref types.ManagedObjectReference) *VirtualMachine {
	return &VirtualMachine{
		Common: NewCommon(c, ref),
	}
}

func (v VirtualMachine) Name(ctx context.Context) (string, error) {
	var o mo.VirtualMachine

	err := v.Properties(ctx, v.Reference(), []string{"name"}, &o)
	if err != nil {
		return "", err
	}

	return o.Name, nil
}

func (v VirtualMachine) PowerState(ctx context.Context) (types.VirtualMachinePowerState, error) {
	var o mo.VirtualMachine

	err := v.Properties(ctx, v.Reference(), []string{PropRuntimePowerState}, &o)
	if err != nil {
		return "", err
	}

	return o.Summary.Runtime.PowerState, nil
}

func (v VirtualMachine) PowerOn(ctx context.Context) (*Task, error) {
	req := types.PowerOnVM_Task{
		This: v.Reference(),
	}

	res, err := methods.PowerOnVM_Task(ctx, v.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(v.c, res.Returnval), nil
}

func (v VirtualMachine) PowerOff(ctx context.Context) (*Task, error) {
	req := types.PowerOffVM_Task{
		This: v.Reference(),
	}

	res, err := methods.PowerOffVM_Task(ctx, v.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(v.c, res.Returnval), nil
}

func (v VirtualMachine) Reset(ctx context.Context) (*Task, error) {
	req := types.ResetVM_Task{
		This: v.Reference(),
	}

	res, err := methods.ResetVM_Task(ctx, v.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(v.c, res.Returnval), nil
}

func (v VirtualMachine) Suspend(ctx context.Context) (*Task, error) {
	req := types.SuspendVM_Task{
		This: v.Reference(),
	}

	res, err := methods.SuspendVM_Task(ctx, v.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(v.c, res.Returnval), nil
}

func (v VirtualMachine) ShutdownGuest(ctx context.Context) error {
	req := types.ShutdownGuest{
		This: v.Reference(),
	}

	_, err := methods.ShutdownGuest(ctx, v.c, &req)
	return err
}

func (v VirtualMachine) RebootGuest(ctx context.Context) error {
	req := types.RebootGuest{
		This: v.Reference(),
	}

	_, err := methods.RebootGuest(ctx, v.c, &req)
	return err
}

func (v VirtualMachine) Destroy(ctx context.Context) (*Task, error) {
	req := types.Destroy_Task{
		This: v.Reference(),
	}

	res, err := methods.Destroy_Task(ctx, v.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(v.c, res.Returnval), nil
}

func (v VirtualMachine) Clone(ctx context.Context, folder *Folder, name string, config types.VirtualMachineCloneSpec) (*Task, error) {
	req := types.CloneVM_Task{
		This:   v.Reference(),
		Folder: folder.Reference(),
		Name:   name,
		Spec:   config,
	}

	res, err := methods.CloneVM_Task(ctx, v.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(v.c, res.Returnval), nil
}

func (v VirtualMachine) Customize(ctx context.Context, spec types.CustomizationSpec) (*Task, error) {
	req := types.CustomizeVM_Task{
		This: v.Reference(),
		Spec: spec,
	}

	res, err := methods.CustomizeVM_Task(ctx, v.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(v.c, res.Returnval), nil
}

func (v VirtualMachine) Relocate(ctx context.Context, config types.VirtualMachineRelocateSpec, priority types.VirtualMachineMovePriority) (*Task, error) {
	req := types.RelocateVM_Task{
		This:     v.Reference(),
		Spec:     config,
		Priority: priority,
	}

	res, err := methods.RelocateVM_Task(ctx, v.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(v.c, res.Returnval), nil
}

func (v VirtualMachine) Reconfigure(ctx context.Context, config types.VirtualMachineConfigSpec) (*Task, error) {
	req := types.ReconfigVM_Task{
		This: v.Reference(),
		Spec: config,
	}

	res, err := methods.ReconfigVM_Task(ctx, v.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(v.c, res.Returnval), nil
}

func (v VirtualMachine) WaitForIP(ctx context.Context) (string, error) {
	var ip string

	p := property.DefaultCollector(v.c)
	err := property.Wait(ctx, p, v.Reference(), []string{"guest.ipAddress"}, func(pc []types.PropertyChange) bool {
		for _, c := range pc {
			if c.Name != "guest.ipAddress" {
				continue
			}
			if c.Op != types.PropertyChangeOpAssign {
				continue
			}
			if c.Val == nil {
				continue
			}

			ip = c.Val.(string)
			return true
		}

		return false
	})

	if err != nil {
		return "", err
	}

	return ip, nil
}

// Device returns the VirtualMachine's config.hardware.device property.
func (v VirtualMachine) Device(ctx context.Context) (VirtualDeviceList, error) {
	var o mo.VirtualMachine

	err := v.Properties(ctx, v.Reference(), []string{"config.hardware.device"}, &o)
	if err != nil {
		return nil, err
	}

	return VirtualDeviceList(o.Config.Hardware.Device), nil
}

func (v VirtualMachine) HostSystem(ctx context.Context) (*HostSystem, error) {
	var o mo.VirtualMachine

	err := v.Properties(ctx, v.Reference(), []string{"summary"}, &o)
	if err != nil {
		return nil, err
	}

	host := o.Summary.Runtime.Host
	if host == nil {
		return nil, errors.New("VM doesn't have a HostSystem")
	}

	return NewHostSystem(v.c, *host), nil
}

func (v VirtualMachine) ResourcePool(ctx context.Context) (*ResourcePool, error) {
	var o mo.VirtualMachine

	err := v.Properties(ctx, v.Reference(), []string{"resourcePool"}, &o)
	if err != nil {
		return nil, err
	}

	rp := o.ResourcePool
	if rp == nil {
		return nil, errors.New("VM doesn't have a resourcePool")
	}

	return NewResourcePool(v.c, *rp), nil
}

func (v VirtualMachine) configureDevice(ctx context.Context, op types.VirtualDeviceConfigSpecOperation, fop types.VirtualDeviceConfigSpecFileOperation, devices ...types.BaseVirtualDevice) error {
	spec := types.VirtualMachineConfigSpec{}

	for _, device := range devices {
		config := &types.VirtualDeviceConfigSpec{
			Device:    device,
			Operation: op,
		}

		if disk, ok := device.(*types.VirtualDisk); ok {
			config.FileOperation = fop

			// Special case to attach an existing disk
			if op == types.VirtualDeviceConfigSpecOperationAdd && disk.CapacityInKB == 0 {
				childDisk := false
				if b, ok := disk.Backing.(*types.VirtualDiskFlatVer2BackingInfo); ok {
					childDisk = b.Parent != nil
				}

				if !childDisk {
					config.FileOperation = "" // existing disk
				}
			}
		}

		spec.DeviceChange = append(spec.DeviceChange, config)
	}

	task, err := v.Reconfigure(ctx, spec)
	if err != nil {
		return err
	}

	return task.Wait(ctx)
}

// AddDevice adds the given devices to the VirtualMachine
func (v VirtualMachine) AddDevice(ctx context.Context, device ...types.BaseVirtualDevice) error {
	return v.configureDevice(ctx, types.VirtualDeviceConfigSpecOperationAdd, types.VirtualDeviceConfigSpecFileOperationCreate, device...)
}

// EditDevice edits the given (existing) devices on the VirtualMachine
func (v VirtualMachine) EditDevice(ctx context.Context, device ...types.BaseVirtualDevice) error {
	return v.configureDevice(ctx, types.VirtualDeviceConfigSpecOperationEdit, types.VirtualDeviceConfigSpecFileOperationReplace, device...)
}

// RemoveDevice removes the given devices on the VirtualMachine
func (v VirtualMachine) RemoveDevice(ctx context.Context, device ...types.BaseVirtualDevice) error {
	return v.configureDevice(ctx, types.VirtualDeviceConfigSpecOperationRemove, types.VirtualDeviceConfigSpecFileOperationDestroy, device...)
}

// BootOptions returns the VirtualMachine's config.bootOptions property.
func (v VirtualMachine) BootOptions(ctx context.Context) (*types.VirtualMachineBootOptions, error) {
	var o mo.VirtualMachine

	err := v.Properties(ctx, v.Reference(), []string{"config.bootOptions"}, &o)
	if err != nil {
		return nil, err
	}

	return o.Config.BootOptions, nil
}

// SetBootOptions reconfigures the VirtualMachine with the given options.
func (v VirtualMachine) SetBootOptions(ctx context.Context, options *types.VirtualMachineBootOptions) error {
	spec := types.VirtualMachineConfigSpec{}

	spec.BootOptions = options

	task, err := v.Reconfigure(ctx, spec)
	if err != nil {
		return err
	}

	return task.Wait(ctx)
}

// Answer answers a pending question.
func (v VirtualMachine) Answer(ctx context.Context, id, answer string) error {
	req := types.AnswerVM{
		This:         v.Reference(),
		QuestionId:   id,
		AnswerChoice: answer,
	}

	_, err := methods.AnswerVM(ctx, v.c, &req)
	if err != nil {
		return err
	}

	return nil
}

// CreateSnapshot creates a new snapshot of a virtual machine.
func (v VirtualMachine) CreateSnapshot(ctx context.Context, name string, description string, memory bool, quiesce bool) (*Task, error) {
	req := types.CreateSnapshot_Task{
		This:        v.Reference(),
		Name:        name,
		Description: description,
		Memory:      memory,
		Quiesce:     quiesce,
	}

	res, err := methods.CreateSnapshot_Task(ctx, v.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(v.c, res.Returnval), nil
}

// IsToolsRunning returns true if VMware Tools is currently running in the guest OS, and false otherwise.
func (v VirtualMachine) IsToolsRunning(ctx context.Context) (bool, error) {
	var o mo.VirtualMachine

	err := v.Properties(ctx, v.Reference(), []string{"guest.toolsRunningStatus"}, &o)
	if err != nil {
		return false, err
	}

	return o.Guest.ToolsRunningStatus == string(types.VirtualMachineToolsRunningStatusGuestToolsRunning), nil
}

// Wait for the VirtualMachine to change to the desired power state.
func (v VirtualMachine) WaitForPowerState(ctx context.Context, state types.VirtualMachinePowerState) error {
	p := property.DefaultCollector(v.c)
	err := property.Wait(ctx, p, v.Reference(), []string{PropRuntimePowerState}, func(pc []types.PropertyChange) bool {
		for _, c := range pc {
			if c.Name != PropRuntimePowerState {
				continue
			}
			if c.Val == nil {
				continue
			}

			ps := c.Val.(types.VirtualMachinePowerState)
			if ps == state {
				return true
			}
		}
		return false
	})

	return err
}

func (v VirtualMachine) MarkAsTemplate(ctx context.Context) error {
	req := types.MarkAsTemplate{
		This: v.Reference(),
	}

	_, err := methods.MarkAsTemplate(ctx, v.c, &req)
	if err != nil {
		return err
	}

	return nil
}

func (v VirtualMachine) MarkAsVirtualMachine(ctx context.Context, pool ResourcePool, host *HostSystem) error {
	req := types.MarkAsVirtualMachine{
		This: v.Reference(),
		Pool: pool.Reference(),
	}

	if host != nil {
		ref := host.Reference()
		req.Host = &ref
	}

	_, err := methods.MarkAsVirtualMachine(ctx, v.c, &req)
	if err != nil {
		return err
	}

	return nil
}
