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

// ServiceContent is the default template for the ServiceInstance content property.
// Capture method:
//   govc object.collect -s -dump - content
var ServiceContent = types.ServiceContent{
	RootFolder:        types.ManagedObjectReference{Type: "Folder", Value: "ha-folder-root"},
	PropertyCollector: types.ManagedObjectReference{Type: "PropertyCollector", Value: "ha-property-collector"},
	ViewManager:       &types.ManagedObjectReference{Type: "ViewManager", Value: "ViewManager"},
	About: types.AboutInfo{
		Name:                  "VMware ESXi",
		FullName:              "VMware ESXi 6.5.0 build-5969303",
		Vendor:                "VMware, Inc.",
		Version:               "6.5.0",
		Build:                 "5969303",
		LocaleVersion:         "INTL",
		LocaleBuild:           "000",
		OsType:                "vmnix-x86",
		ProductLineId:         "embeddedEsx",
		ApiType:               "HostAgent",
		ApiVersion:            "6.5",
		InstanceUuid:          "",
		LicenseProductName:    "VMware ESX Server",
		LicenseProductVersion: "6.0",
	},
	Setting:                     &types.ManagedObjectReference{Type: "OptionManager", Value: "HostAgentSettings"},
	UserDirectory:               &types.ManagedObjectReference{Type: "UserDirectory", Value: "ha-user-directory"},
	SessionManager:              &types.ManagedObjectReference{Type: "SessionManager", Value: "ha-sessionmgr"},
	AuthorizationManager:        &types.ManagedObjectReference{Type: "AuthorizationManager", Value: "ha-authmgr"},
	ServiceManager:              &types.ManagedObjectReference{Type: "ServiceManager", Value: "ha-servicemanager"},
	PerfManager:                 &types.ManagedObjectReference{Type: "PerformanceManager", Value: "ha-perfmgr"},
	ScheduledTaskManager:        (*types.ManagedObjectReference)(nil),
	AlarmManager:                (*types.ManagedObjectReference)(nil),
	EventManager:                &types.ManagedObjectReference{Type: "EventManager", Value: "ha-eventmgr"},
	TaskManager:                 &types.ManagedObjectReference{Type: "TaskManager", Value: "ha-taskmgr"},
	ExtensionManager:            (*types.ManagedObjectReference)(nil),
	CustomizationSpecManager:    (*types.ManagedObjectReference)(nil),
	CustomFieldsManager:         (*types.ManagedObjectReference)(nil),
	AccountManager:              &types.ManagedObjectReference{Type: "HostLocalAccountManager", Value: "ha-localacctmgr"},
	DiagnosticManager:           &types.ManagedObjectReference{Type: "DiagnosticManager", Value: "ha-diagnosticmgr"},
	LicenseManager:              &types.ManagedObjectReference{Type: "LicenseManager", Value: "ha-license-manager"},
	SearchIndex:                 &types.ManagedObjectReference{Type: "SearchIndex", Value: "ha-searchindex"},
	FileManager:                 &types.ManagedObjectReference{Type: "FileManager", Value: "ha-nfc-file-manager"},
	DatastoreNamespaceManager:   &types.ManagedObjectReference{Type: "DatastoreNamespaceManager", Value: "ha-datastore-namespace-manager"},
	VirtualDiskManager:          &types.ManagedObjectReference{Type: "VirtualDiskManager", Value: "ha-vdiskmanager"},
	VirtualizationManager:       (*types.ManagedObjectReference)(nil),
	SnmpSystem:                  (*types.ManagedObjectReference)(nil),
	VmProvisioningChecker:       (*types.ManagedObjectReference)(nil),
	VmCompatibilityChecker:      (*types.ManagedObjectReference)(nil),
	OvfManager:                  &types.ManagedObjectReference{Type: "OvfManager", Value: "ha-ovf-manager"},
	IpPoolManager:               (*types.ManagedObjectReference)(nil),
	DvSwitchManager:             &types.ManagedObjectReference{Type: "DistributedVirtualSwitchManager", Value: "ha-dvsmanager"},
	HostProfileManager:          (*types.ManagedObjectReference)(nil),
	ClusterProfileManager:       (*types.ManagedObjectReference)(nil),
	ComplianceManager:           (*types.ManagedObjectReference)(nil),
	LocalizationManager:         &types.ManagedObjectReference{Type: "LocalizationManager", Value: "ha-l10n-manager"},
	StorageResourceManager:      &types.ManagedObjectReference{Type: "StorageResourceManager", Value: "ha-storage-resource-manager"},
	GuestOperationsManager:      &types.ManagedObjectReference{Type: "GuestOperationsManager", Value: "ha-guest-operations-manager"},
	OverheadMemoryManager:       (*types.ManagedObjectReference)(nil),
	CertificateManager:          (*types.ManagedObjectReference)(nil),
	IoFilterManager:             (*types.ManagedObjectReference)(nil),
	VStorageObjectManager:       &types.ManagedObjectReference{Type: "HostVStorageObjectManager", Value: "ha-vstorage-object-manager"},
	HostSpecManager:             (*types.ManagedObjectReference)(nil),
	CryptoManager:               &types.ManagedObjectReference{Type: "CryptoManager", Value: "ha-crypto-manager"},
	HealthUpdateManager:         (*types.ManagedObjectReference)(nil),
	FailoverClusterConfigurator: (*types.ManagedObjectReference)(nil),
	FailoverClusterManager:      (*types.ManagedObjectReference)(nil),
}
