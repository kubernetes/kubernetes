/*
Copyright (c) 2015-2022 VMware, Inc. All Rights Reserved.

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

package types

import (
	"net/url"
	"reflect"
	"strings"
	"time"
)

func NewBool(v bool) *bool {
	return &v
}

func NewInt32(v int32) *int32 {
	return &v
}

func NewInt64(v int64) *int64 {
	return &v
}

func NewTime(v time.Time) *time.Time {
	return &v
}

func NewReference(r ManagedObjectReference) *ManagedObjectReference {
	return &r
}

func (r ManagedObjectReference) Reference() ManagedObjectReference {
	return r
}

func (r ManagedObjectReference) String() string {
	return strings.Join([]string{r.Type, r.Value}, ":")
}

func (r *ManagedObjectReference) FromString(o string) bool {
	s := strings.SplitN(o, ":", 2)

	if len(s) != 2 {
		return false
	}

	r.Type = s[0]
	r.Value = s[1]

	return true
}

// Encode ManagedObjectReference for use with URL and File paths
func (r ManagedObjectReference) Encode() string {
	return strings.Join([]string{r.Type, url.QueryEscape(r.Value)}, "-")
}

func (c *PerfCounterInfo) Name() string {
	return c.GroupInfo.GetElementDescription().Key + "." + c.NameInfo.GetElementDescription().Key + "." + string(c.RollupType)
}

func defaultResourceAllocationInfo() ResourceAllocationInfo {
	return ResourceAllocationInfo{
		Reservation:           NewInt64(0),
		ExpandableReservation: NewBool(true),
		Limit:                 NewInt64(-1),
		Shares: &SharesInfo{
			Level: SharesLevelNormal,
		},
	}
}

// DefaultResourceConfigSpec returns a ResourceConfigSpec populated with the same default field values as vCenter.
// Note that the wsdl marks these fields as optional, but they are required to be set when creating a resource pool.
// They are only optional when updating a resource pool.
func DefaultResourceConfigSpec() ResourceConfigSpec {
	return ResourceConfigSpec{
		CpuAllocation:    defaultResourceAllocationInfo(),
		MemoryAllocation: defaultResourceAllocationInfo(),
	}
}

// ToConfigSpec returns a VirtualMachineConfigSpec based on the
// VirtualMachineConfigInfo.
func (ci VirtualMachineConfigInfo) ToConfigSpec() VirtualMachineConfigSpec {
	cs := VirtualMachineConfigSpec{
		ChangeVersion:                ci.ChangeVersion,
		Name:                         ci.Name,
		Version:                      ci.Version,
		CreateDate:                   ci.CreateDate,
		Uuid:                         ci.Uuid,
		InstanceUuid:                 ci.InstanceUuid,
		NpivNodeWorldWideName:        ci.NpivNodeWorldWideName,
		NpivPortWorldWideName:        ci.NpivPortWorldWideName,
		NpivWorldWideNameType:        ci.NpivWorldWideNameType,
		NpivDesiredNodeWwns:          ci.NpivDesiredNodeWwns,
		NpivDesiredPortWwns:          ci.NpivDesiredPortWwns,
		NpivTemporaryDisabled:        ci.NpivTemporaryDisabled,
		NpivOnNonRdmDisks:            ci.NpivOnNonRdmDisks,
		LocationId:                   ci.LocationId,
		GuestId:                      ci.GuestId,
		AlternateGuestName:           ci.AlternateGuestName,
		Annotation:                   ci.Annotation,
		Files:                        &ci.Files,
		Tools:                        ci.Tools,
		Flags:                        &ci.Flags,
		ConsolePreferences:           ci.ConsolePreferences,
		PowerOpInfo:                  &ci.DefaultPowerOps,
		NumCPUs:                      ci.Hardware.NumCPU,
		VcpuConfig:                   ci.VcpuConfig,
		NumCoresPerSocket:            ci.Hardware.NumCoresPerSocket,
		MemoryMB:                     int64(ci.Hardware.MemoryMB),
		MemoryHotAddEnabled:          ci.MemoryHotAddEnabled,
		CpuHotAddEnabled:             ci.CpuHotAddEnabled,
		CpuHotRemoveEnabled:          ci.CpuHotRemoveEnabled,
		VirtualICH7MPresent:          ci.Hardware.VirtualICH7MPresent,
		VirtualSMCPresent:            ci.Hardware.VirtualSMCPresent,
		DeviceChange:                 make([]BaseVirtualDeviceConfigSpec, len(ci.Hardware.Device)),
		CpuAllocation:                ci.CpuAllocation,
		MemoryAllocation:             ci.MemoryAllocation,
		LatencySensitivity:           ci.LatencySensitivity,
		CpuAffinity:                  ci.CpuAffinity,
		MemoryAffinity:               ci.MemoryAffinity,
		NetworkShaper:                ci.NetworkShaper,
		CpuFeatureMask:               make([]VirtualMachineCpuIdInfoSpec, len(ci.CpuFeatureMask)),
		ExtraConfig:                  ci.ExtraConfig,
		SwapPlacement:                ci.SwapPlacement,
		BootOptions:                  ci.BootOptions,
		FtInfo:                       ci.FtInfo,
		RepConfig:                    ci.RepConfig,
		VAssertsEnabled:              ci.VAssertsEnabled,
		ChangeTrackingEnabled:        ci.ChangeTrackingEnabled,
		Firmware:                     ci.Firmware,
		MaxMksConnections:            ci.MaxMksConnections,
		GuestAutoLockEnabled:         ci.GuestAutoLockEnabled,
		ManagedBy:                    ci.ManagedBy,
		MemoryReservationLockedToMax: ci.MemoryReservationLockedToMax,
		NestedHVEnabled:              ci.NestedHVEnabled,
		VPMCEnabled:                  ci.VPMCEnabled,
		MessageBusTunnelEnabled:      ci.MessageBusTunnelEnabled,
		MigrateEncryption:            ci.MigrateEncryption,
		FtEncryptionMode:             ci.FtEncryptionMode,
		SevEnabled:                   ci.SevEnabled,
		PmemFailoverEnabled:          ci.PmemFailoverEnabled,
		Pmem:                         ci.Pmem,
		NpivWorldWideNameOp:          ci.NpivWorldWideNameType,
		RebootPowerOff:               ci.RebootPowerOff,
		ScheduledHardwareUpgradeInfo: ci.ScheduledHardwareUpgradeInfo,
		SgxInfo:                      ci.SgxInfo,
		GuestMonitoringModeInfo:      ci.GuestMonitoringModeInfo,
		VmxStatsCollectionEnabled:    ci.VmxStatsCollectionEnabled,
		VmOpNotificationToAppEnabled: ci.VmOpNotificationToAppEnabled,
		VmOpNotificationTimeout:      ci.VmOpNotificationTimeout,
		DeviceSwap:                   ci.DeviceSwap,
		SimultaneousThreads:          ci.Hardware.SimultaneousThreads,
		DeviceGroups:                 ci.DeviceGroups,
		MotherboardLayout:            ci.Hardware.MotherboardLayout,
	}

	// Unassign the Files field if all of its fields are empty.
	if ci.Files.FtMetadataDirectory == "" && ci.Files.LogDirectory == "" &&
		ci.Files.SnapshotDirectory == "" && ci.Files.SuspendDirectory == "" &&
		ci.Files.VmPathName == "" {
		cs.Files = nil
	}

	// Unassign the Flags field if all of its fields are empty.
	if ci.Flags.CbrcCacheEnabled == nil &&
		ci.Flags.DisableAcceleration == nil &&
		ci.Flags.DiskUuidEnabled == nil &&
		ci.Flags.EnableLogging == nil &&
		ci.Flags.FaultToleranceType == "" &&
		ci.Flags.HtSharing == "" &&
		ci.Flags.MonitorType == "" &&
		ci.Flags.RecordReplayEnabled == nil &&
		ci.Flags.RunWithDebugInfo == nil &&
		ci.Flags.SnapshotDisabled == nil &&
		ci.Flags.SnapshotLocked == nil &&
		ci.Flags.SnapshotPowerOffBehavior == "" &&
		ci.Flags.UseToe == nil &&
		ci.Flags.VbsEnabled == nil &&
		ci.Flags.VirtualExecUsage == "" &&
		ci.Flags.VirtualMmuUsage == "" &&
		ci.Flags.VvtdEnabled == nil {
		cs.Flags = nil
	}

	// Unassign the PowerOps field if all of its fields are empty.
	if ci.DefaultPowerOps.DefaultPowerOffType == "" &&
		ci.DefaultPowerOps.DefaultResetType == "" &&
		ci.DefaultPowerOps.DefaultSuspendType == "" &&
		ci.DefaultPowerOps.PowerOffType == "" &&
		ci.DefaultPowerOps.ResetType == "" &&
		ci.DefaultPowerOps.StandbyAction == "" &&
		ci.DefaultPowerOps.SuspendType == "" {
		cs.PowerOpInfo = nil
	}

	for i := 0; i < len(cs.CpuFeatureMask); i++ {
		cs.CpuFeatureMask[i] = VirtualMachineCpuIdInfoSpec{
			ArrayUpdateSpec: ArrayUpdateSpec{
				Operation: ArrayUpdateOperationAdd,
			},
			Info: &HostCpuIdInfo{
				// TODO: Does DynamicData need to be copied?
				//       It is an empty struct...
				Level:  ci.CpuFeatureMask[i].Level,
				Vendor: ci.CpuFeatureMask[i].Vendor,
				Eax:    ci.CpuFeatureMask[i].Eax,
				Ebx:    ci.CpuFeatureMask[i].Ebx,
				Ecx:    ci.CpuFeatureMask[i].Ecx,
				Edx:    ci.CpuFeatureMask[i].Edx,
			},
		}
	}

	for i := 0; i < len(cs.DeviceChange); i++ {
		cs.DeviceChange[i] = &VirtualDeviceConfigSpec{
			// TODO: Does DynamicData need to be copied?
			//       It is an empty struct...
			Operation:     VirtualDeviceConfigSpecOperationAdd,
			FileOperation: VirtualDeviceConfigSpecFileOperationCreate,
			Device:        ci.Hardware.Device[i],
			// TODO: It is unclear how the profiles associated with the VM or
			//       its hardware can be reintroduced/persisted in the
			//       ConfigSpec.
			Profile: nil,
			// The backing will come from the device.
			Backing: nil,
			// TODO: Investigate futher.
			FilterSpec: nil,
		}
	}

	if ni := ci.NumaInfo; ni != nil {
		cs.VirtualNuma = &VirtualMachineVirtualNuma{
			CoresPerNumaNode:       ni.CoresPerNumaNode,
			ExposeVnumaOnCpuHotadd: ni.VnumaOnCpuHotaddExposed,
		}
	}

	if civa, ok := ci.VAppConfig.(*VmConfigInfo); ok {
		var csva VmConfigSpec

		csva.Eula = civa.Eula
		csva.InstallBootRequired = &civa.InstallBootRequired
		csva.InstallBootStopDelay = civa.InstallBootStopDelay

		ipAssignment := civa.IpAssignment
		csva.IpAssignment = &ipAssignment

		csva.OvfEnvironmentTransport = civa.OvfEnvironmentTransport
		for i := range civa.OvfSection {
			s := civa.OvfSection[i]
			csva.OvfSection = append(
				csva.OvfSection,
				VAppOvfSectionSpec{
					ArrayUpdateSpec: ArrayUpdateSpec{
						Operation: ArrayUpdateOperationAdd,
					},
					Info: &s,
				},
			)
		}

		for i := range civa.Product {
			p := civa.Product[i]
			csva.Product = append(
				csva.Product,
				VAppProductSpec{
					ArrayUpdateSpec: ArrayUpdateSpec{
						Operation: ArrayUpdateOperationAdd,
					},
					Info: &p,
				},
			)
		}

		for i := range civa.Property {
			p := civa.Property[i]
			csva.Property = append(
				csva.Property,
				VAppPropertySpec{
					ArrayUpdateSpec: ArrayUpdateSpec{
						Operation: ArrayUpdateOperationAdd,
					},
					Info: &p,
				},
			)
		}

		cs.VAppConfig = &csva
	}

	return cs
}

func init() {
	// Known 6.5 issue where this event type is sent even though it is internal.
	// This workaround allows us to unmarshal and avoid NPEs.
	t["HostSubSpecificationUpdateEvent"] = reflect.TypeOf((*HostEvent)(nil)).Elem()
}
