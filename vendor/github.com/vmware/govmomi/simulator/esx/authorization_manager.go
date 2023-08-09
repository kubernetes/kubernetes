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

package esx

import "github.com/vmware/govmomi/vim25/types"

// RoleList is the default template for the AuthorizationManager roleList property.
// Capture method:
//   govc object.collect -s -dump AuthorizationManager:ha-authmgr roleList
var RoleList = []types.AuthorizationRole{
	{
		RoleId: -6,
		System: true,
		Name:   "NoCryptoAdmin",
		Info: &types.Description{
			Label:   "No cryptography administrator",
			Summary: "Full access without Cryptographic operations privileges",
		},
		Privilege: nil,
	},
	{
		RoleId: -5,
		System: true,
		Name:   "NoAccess",
		Info: &types.Description{
			Label:   "No access",
			Summary: "Used for restricting granted access",
		},
		Privilege: nil,
	},
	{
		RoleId: -4,
		System: true,
		Name:   "Anonymous",
		Info: &types.Description{
			Label:   "Anonymous",
			Summary: "Not logged-in user (cannot be granted)",
		},
		Privilege: []string{"System.Anonymous"},
	},
	{
		RoleId: -3,
		System: true,
		Name:   "View",
		Info: &types.Description{
			Label:   "View",
			Summary: "Visibility access (cannot be granted)",
		},
		Privilege: []string{"System.Anonymous", "System.View"},
	},
	{
		RoleId: -2,
		System: true,
		Name:   "ReadOnly",
		Info: &types.Description{
			Label:   "Read-only",
			Summary: "See details of objects, but not make changes",
		},
		Privilege: []string{"System.Anonymous", "System.Read", "System.View"},
	},
	{
		RoleId: -1,
		System: true,
		Name:   "Admin",
		Info: &types.Description{
			Label:   "Administrator",
			Summary: "Full access rights",
		},
		Privilege: []string{"Alarm.Acknowledge", "Alarm.Create", "Alarm.Delete", "Alarm.DisableActions", "Alarm.Edit", "Alarm.SetStatus", "Authorization.ModifyPermissions", "Authorization.ModifyRoles", "Authorization.ReassignRolePermissions", "Certificate.Manage", "Cryptographer.Access", "Cryptographer.AddDisk", "Cryptographer.Clone", "Cryptographer.Decrypt", "Cryptographer.Encrypt", "Cryptographer.EncryptNew", "Cryptographer.ManageEncryptionPolicy", "Cryptographer.ManageKeyServers", "Cryptographer.ManageKeys", "Cryptographer.Migrate", "Cryptographer.Recrypt", "Cryptographer.RegisterHost", "Cryptographer.RegisterVM", "DVPortgroup.Create", "DVPortgroup.Delete", "DVPortgroup.Modify", "DVPortgroup.PolicyOp", "DVPortgroup.ScopeOp", "DVSwitch.Create", "DVSwitch.Delete", "DVSwitch.HostOp", "DVSwitch.Modify", "DVSwitch.Move", "DVSwitch.PolicyOp", "DVSwitch.PortConfig", "DVSwitch.PortSetting", "DVSwitch.ResourceManagement", "DVSwitch.Vspan", "Datacenter.Create", "Datacenter.Delete", "Datacenter.IpPoolConfig", "Datacenter.IpPoolQueryAllocations", "Datacenter.IpPoolReleaseIp", "Datacenter.Move", "Datacenter.Reconfigure", "Datacenter.Rename", "Datastore.AllocateSpace", "Datastore.Browse", "Datastore.Config", "Datastore.Delete", "Datastore.DeleteFile", "Datastore.FileManagement", "Datastore.Move", "Datastore.Rename", "Datastore.UpdateVirtualMachineFiles", "Datastore.UpdateVirtualMachineMetadata", "EAM.Config", "EAM.Modify", "EAM.View", "Extension.Register", "Extension.Unregister", "Extension.Update", "ExternalStatsProvider.Register", "ExternalStatsProvider.Unregister", "ExternalStatsProvider.Update", "Folder.Create", "Folder.Delete", "Folder.Move", "Folder.Rename", "Global.CancelTask", "Global.CapacityPlanning", "Global.Diagnostics", "Global.DisableMethods", "Global.EnableMethods", "Global.GlobalTag", "Global.Health", "Global.Licenses", "Global.LogEvent", "Global.ManageCustomFields", "Global.Proxy", "Global.ScriptAction", "Global.ServiceManagers", "Global.SetCustomField", "Global.Settings", "Global.SystemTag", "Global.VCServer", "HealthUpdateProvider.Register", "HealthUpdateProvider.Unregister", "HealthUpdateProvider.Update", "Host.Cim.CimInteraction", "Host.Config.AdvancedConfig", "Host.Config.AuthenticationStore", "Host.Config.AutoStart", "Host.Config.Connection", "Host.Config.DateTime", "Host.Config.Firmware", "Host.Config.HyperThreading", "Host.Config.Image", "Host.Config.Maintenance", "Host.Config.Memory", "Host.Config.NetService", "Host.Config.Network", "Host.Config.Patch", "Host.Config.PciPassthru", "Host.Config.Power", "Host.Config.Quarantine", "Host.Config.Resources", "Host.Config.Settings", "Host.Config.Snmp", "Host.Config.Storage", "Host.Config.SystemManagement", "Host.Hbr.HbrManagement", "Host.Inventory.AddHostToCluster", "Host.Inventory.AddStandaloneHost", "Host.Inventory.CreateCluster", "Host.Inventory.DeleteCluster", "Host.Inventory.EditCluster", "Host.Inventory.MoveCluster", "Host.Inventory.MoveHost", "Host.Inventory.RemoveHostFromCluster", "Host.Inventory.RenameCluster", "Host.Local.CreateVM", "Host.Local.DeleteVM", "Host.Local.InstallAgent", "Host.Local.ManageUserGroups", "Host.Local.ReconfigVM", "Network.Assign", "Network.Config", "Network.Delete", "Network.Move", "Performance.ModifyIntervals", "Profile.Clear", "Profile.Create", "Profile.Delete", "Profile.Edit", "Profile.Export", "Profile.View", "Resource.ApplyRecommendation", "Resource.AssignVAppToPool", "Resource.AssignVMToPool", "Resource.ColdMigrate", "Resource.CreatePool", "Resource.DeletePool", "Resource.EditPool", "Resource.HotMigrate", "Resource.MovePool", "Resource.QueryVMotion", "Resource.RenamePool", "ScheduledTask.Create", "ScheduledTask.Delete", "ScheduledTask.Edit", "ScheduledTask.Run", "Sessions.GlobalMessage", "Sessions.ImpersonateUser", "Sessions.TerminateSession", "Sessions.ValidateSession", "StoragePod.Config", "System.Anonymous", "System.Read", "System.View", "Task.Create", "Task.Update", "VApp.ApplicationConfig", "VApp.AssignResourcePool", "VApp.AssignVApp", "VApp.AssignVM", "VApp.Clone", "VApp.Create", "VApp.Delete", "VApp.Export", "VApp.ExtractOvfEnvironment", "VApp.Import", "VApp.InstanceConfig", "VApp.ManagedByConfig", "VApp.Move", "VApp.PowerOff", "VApp.PowerOn", "VApp.Rename", "VApp.ResourceConfig", "VApp.Suspend", "VApp.Unregister", "VRMPolicy.Query", "VRMPolicy.Update", "VirtualMachine.Config.AddExistingDisk", "VirtualMachine.Config.AddNewDisk", "VirtualMachine.Config.AddRemoveDevice", "VirtualMachine.Config.AdvancedConfig", "VirtualMachine.Config.Annotation", "VirtualMachine.Config.CPUCount", "VirtualMachine.Config.ChangeTracking", "VirtualMachine.Config.DiskExtend", "VirtualMachine.Config.DiskLease", "VirtualMachine.Config.EditDevice", "VirtualMachine.Config.HostUSBDevice", "VirtualMachine.Config.ManagedBy", "VirtualMachine.Config.Memory", "VirtualMachine.Config.MksControl", "VirtualMachine.Config.QueryFTCompatibility", "VirtualMachine.Config.QueryUnownedFiles", "VirtualMachine.Config.RawDevice", "VirtualMachine.Config.ReloadFromPath", "VirtualMachine.Config.RemoveDisk", "VirtualMachine.Config.Rename", "VirtualMachine.Config.ResetGuestInfo", "VirtualMachine.Config.Resource", "VirtualMachine.Config.Settings", "VirtualMachine.Config.SwapPlacement", "VirtualMachine.Config.ToggleForkParent", "VirtualMachine.Config.Unlock", "VirtualMachine.Config.UpgradeVirtualHardware", "VirtualMachine.GuestOperations.Execute", "VirtualMachine.GuestOperations.Modify", "VirtualMachine.GuestOperations.ModifyAliases", "VirtualMachine.GuestOperations.Query", "VirtualMachine.GuestOperations.QueryAliases", "VirtualMachine.Hbr.ConfigureReplication", "VirtualMachine.Hbr.MonitorReplication", "VirtualMachine.Hbr.ReplicaManagement", "VirtualMachine.Interact.AnswerQuestion", "VirtualMachine.Interact.Backup", "VirtualMachine.Interact.ConsoleInteract", "VirtualMachine.Interact.CreateScreenshot", "VirtualMachine.Interact.CreateSecondary", "VirtualMachine.Interact.DefragmentAllDisks", "VirtualMachine.Interact.DeviceConnection", "VirtualMachine.Interact.DisableSecondary", "VirtualMachine.Interact.DnD", "VirtualMachine.Interact.EnableSecondary", "VirtualMachine.Interact.GuestControl", "VirtualMachine.Interact.MakePrimary", "VirtualMachine.Interact.Pause", "VirtualMachine.Interact.PowerOff", "VirtualMachine.Interact.PowerOn", "VirtualMachine.Interact.PutUsbScanCodes", "VirtualMachine.Interact.Record", "VirtualMachine.Interact.Replay", "VirtualMachine.Interact.Reset", "VirtualMachine.Interact.SESparseMaintenance", "VirtualMachine.Interact.SetCDMedia", "VirtualMachine.Interact.SetFloppyMedia", "VirtualMachine.Interact.Suspend", "VirtualMachine.Interact.TerminateFaultTolerantVM", "VirtualMachine.Interact.ToolsInstall", "VirtualMachine.Interact.TurnOffFaultTolerance", "VirtualMachine.Inventory.Create", "VirtualMachine.Inventory.CreateFromExisting", "VirtualMachine.Inventory.Delete", "VirtualMachine.Inventory.Move", "VirtualMachine.Inventory.Register", "VirtualMachine.Inventory.Unregister", "VirtualMachine.Namespace.Event", "VirtualMachine.Namespace.EventNotify", "VirtualMachine.Namespace.Management", "VirtualMachine.Namespace.ModifyContent", "VirtualMachine.Namespace.Query", "VirtualMachine.Namespace.ReadContent", "VirtualMachine.Provisioning.Clone", "VirtualMachine.Provisioning.CloneTemplate", "VirtualMachine.Provisioning.CreateTemplateFromVM", "VirtualMachine.Provisioning.Customize", "VirtualMachine.Provisioning.DeployTemplate", "VirtualMachine.Provisioning.DiskRandomAccess", "VirtualMachine.Provisioning.DiskRandomRead", "VirtualMachine.Provisioning.FileRandomAccess", "VirtualMachine.Provisioning.GetVmFiles", "VirtualMachine.Provisioning.MarkAsTemplate", "VirtualMachine.Provisioning.MarkAsVM", "VirtualMachine.Provisioning.ModifyCustSpecs", "VirtualMachine.Provisioning.PromoteDisks", "VirtualMachine.Provisioning.PutVmFiles", "VirtualMachine.Provisioning.ReadCustSpecs", "VirtualMachine.State.CreateSnapshot", "VirtualMachine.State.RemoveSnapshot", "VirtualMachine.State.RenameSnapshot", "VirtualMachine.State.RevertToSnapshot"},
	},
}
