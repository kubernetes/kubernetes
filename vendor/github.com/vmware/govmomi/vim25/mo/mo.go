/*
Copyright (c) 2014-2016 VMware, Inc. All Rights Reserved.

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

package mo

import (
	"reflect"
	"time"

	"github.com/vmware/govmomi/vim25/types"
)

type Alarm struct {
	ExtensibleManagedObject

	Info types.AlarmInfo `mo:"info"`
}

func init() {
	t["Alarm"] = reflect.TypeOf((*Alarm)(nil)).Elem()
}

type AlarmManager struct {
	Self types.ManagedObjectReference

	DefaultExpression []types.BaseAlarmExpression `mo:"defaultExpression"`
	Description       types.AlarmDescription      `mo:"description"`
}

func (m AlarmManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["AlarmManager"] = reflect.TypeOf((*AlarmManager)(nil)).Elem()
}

type AuthorizationManager struct {
	Self types.ManagedObjectReference

	PrivilegeList []types.AuthorizationPrivilege `mo:"privilegeList"`
	RoleList      []types.AuthorizationRole      `mo:"roleList"`
	Description   types.AuthorizationDescription `mo:"description"`
}

func (m AuthorizationManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["AuthorizationManager"] = reflect.TypeOf((*AuthorizationManager)(nil)).Elem()
}

type CertificateManager struct {
	Self types.ManagedObjectReference
}

func (m CertificateManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["CertificateManager"] = reflect.TypeOf((*CertificateManager)(nil)).Elem()
}

type ClusterComputeResource struct {
	ComputeResource

	Configuration     types.ClusterConfigInfo          `mo:"configuration"`
	Recommendation    []types.ClusterRecommendation    `mo:"recommendation"`
	DrsRecommendation []types.ClusterDrsRecommendation `mo:"drsRecommendation"`
	MigrationHistory  []types.ClusterDrsMigration      `mo:"migrationHistory"`
	ActionHistory     []types.ClusterActionHistory     `mo:"actionHistory"`
	DrsFault          []types.ClusterDrsFaults         `mo:"drsFault"`
}

func init() {
	t["ClusterComputeResource"] = reflect.TypeOf((*ClusterComputeResource)(nil)).Elem()
}

type ClusterEVCManager struct {
	ExtensibleManagedObject

	ManagedCluster types.ManagedObjectReference    `mo:"managedCluster"`
	EvcState       types.ClusterEVCManagerEVCState `mo:"evcState"`
}

func init() {
	t["ClusterEVCManager"] = reflect.TypeOf((*ClusterEVCManager)(nil)).Elem()
}

type ClusterProfile struct {
	Profile
}

func init() {
	t["ClusterProfile"] = reflect.TypeOf((*ClusterProfile)(nil)).Elem()
}

type ClusterProfileManager struct {
	ProfileManager
}

func init() {
	t["ClusterProfileManager"] = reflect.TypeOf((*ClusterProfileManager)(nil)).Elem()
}

type ComputeResource struct {
	ManagedEntity

	ResourcePool       *types.ManagedObjectReference       `mo:"resourcePool"`
	Host               []types.ManagedObjectReference      `mo:"host"`
	Datastore          []types.ManagedObjectReference      `mo:"datastore"`
	Network            []types.ManagedObjectReference      `mo:"network"`
	Summary            types.BaseComputeResourceSummary    `mo:"summary"`
	EnvironmentBrowser *types.ManagedObjectReference       `mo:"environmentBrowser"`
	ConfigurationEx    types.BaseComputeResourceConfigInfo `mo:"configurationEx"`
}

func (m *ComputeResource) Entity() *ManagedEntity {
	return &m.ManagedEntity
}

func init() {
	t["ComputeResource"] = reflect.TypeOf((*ComputeResource)(nil)).Elem()
}

type ContainerView struct {
	ManagedObjectView

	Container types.ManagedObjectReference `mo:"container"`
	Type      []string                     `mo:"type"`
	Recursive bool                         `mo:"recursive"`
}

func init() {
	t["ContainerView"] = reflect.TypeOf((*ContainerView)(nil)).Elem()
}

type CustomFieldsManager struct {
	Self types.ManagedObjectReference

	Field []types.CustomFieldDef `mo:"field"`
}

func (m CustomFieldsManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["CustomFieldsManager"] = reflect.TypeOf((*CustomFieldsManager)(nil)).Elem()
}

type CustomizationSpecManager struct {
	Self types.ManagedObjectReference

	Info          []types.CustomizationSpecInfo `mo:"info"`
	EncryptionKey []byte                        `mo:"encryptionKey"`
}

func (m CustomizationSpecManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["CustomizationSpecManager"] = reflect.TypeOf((*CustomizationSpecManager)(nil)).Elem()
}

type Datacenter struct {
	ManagedEntity

	VmFolder        types.ManagedObjectReference   `mo:"vmFolder"`
	HostFolder      types.ManagedObjectReference   `mo:"hostFolder"`
	DatastoreFolder types.ManagedObjectReference   `mo:"datastoreFolder"`
	NetworkFolder   types.ManagedObjectReference   `mo:"networkFolder"`
	Datastore       []types.ManagedObjectReference `mo:"datastore"`
	Network         []types.ManagedObjectReference `mo:"network"`
	Configuration   types.DatacenterConfigInfo     `mo:"configuration"`
}

func (m *Datacenter) Entity() *ManagedEntity {
	return &m.ManagedEntity
}

func init() {
	t["Datacenter"] = reflect.TypeOf((*Datacenter)(nil)).Elem()
}

type Datastore struct {
	ManagedEntity

	Info              types.BaseDatastoreInfo        `mo:"info"`
	Summary           types.DatastoreSummary         `mo:"summary"`
	Host              []types.DatastoreHostMount     `mo:"host"`
	Vm                []types.ManagedObjectReference `mo:"vm"`
	Browser           types.ManagedObjectReference   `mo:"browser"`
	Capability        types.DatastoreCapability      `mo:"capability"`
	IormConfiguration *types.StorageIORMInfo         `mo:"iormConfiguration"`
}

func (m *Datastore) Entity() *ManagedEntity {
	return &m.ManagedEntity
}

func init() {
	t["Datastore"] = reflect.TypeOf((*Datastore)(nil)).Elem()
}

type DatastoreNamespaceManager struct {
	Self types.ManagedObjectReference
}

func (m DatastoreNamespaceManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["DatastoreNamespaceManager"] = reflect.TypeOf((*DatastoreNamespaceManager)(nil)).Elem()
}

type DiagnosticManager struct {
	Self types.ManagedObjectReference
}

func (m DiagnosticManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["DiagnosticManager"] = reflect.TypeOf((*DiagnosticManager)(nil)).Elem()
}

type DistributedVirtualPortgroup struct {
	Network

	Key      string                      `mo:"key"`
	Config   types.DVPortgroupConfigInfo `mo:"config"`
	PortKeys []string                    `mo:"portKeys"`
}

func init() {
	t["DistributedVirtualPortgroup"] = reflect.TypeOf((*DistributedVirtualPortgroup)(nil)).Elem()
}

type DistributedVirtualSwitch struct {
	ManagedEntity

	Uuid                string                         `mo:"uuid"`
	Capability          types.DVSCapability            `mo:"capability"`
	Summary             types.DVSSummary               `mo:"summary"`
	Config              types.BaseDVSConfigInfo        `mo:"config"`
	NetworkResourcePool []types.DVSNetworkResourcePool `mo:"networkResourcePool"`
	Portgroup           []types.ManagedObjectReference `mo:"portgroup"`
	Runtime             *types.DVSRuntimeInfo          `mo:"runtime"`
}

func (m *DistributedVirtualSwitch) Entity() *ManagedEntity {
	return &m.ManagedEntity
}

func init() {
	t["DistributedVirtualSwitch"] = reflect.TypeOf((*DistributedVirtualSwitch)(nil)).Elem()
}

type DistributedVirtualSwitchManager struct {
	Self types.ManagedObjectReference
}

func (m DistributedVirtualSwitchManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["DistributedVirtualSwitchManager"] = reflect.TypeOf((*DistributedVirtualSwitchManager)(nil)).Elem()
}

type EnvironmentBrowser struct {
	Self types.ManagedObjectReference

	DatastoreBrowser *types.ManagedObjectReference `mo:"datastoreBrowser"`
}

func (m EnvironmentBrowser) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["EnvironmentBrowser"] = reflect.TypeOf((*EnvironmentBrowser)(nil)).Elem()
}

type EventHistoryCollector struct {
	HistoryCollector

	LatestPage []types.BaseEvent `mo:"latestPage"`
}

func init() {
	t["EventHistoryCollector"] = reflect.TypeOf((*EventHistoryCollector)(nil)).Elem()
}

type EventManager struct {
	Self types.ManagedObjectReference

	Description  types.EventDescription `mo:"description"`
	LatestEvent  types.BaseEvent        `mo:"latestEvent"`
	MaxCollector int32                  `mo:"maxCollector"`
}

func (m EventManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["EventManager"] = reflect.TypeOf((*EventManager)(nil)).Elem()
}

type ExtensibleManagedObject struct {
	Self types.ManagedObjectReference

	Value          []types.BaseCustomFieldValue `mo:"value"`
	AvailableField []types.CustomFieldDef       `mo:"availableField"`
}

func (m ExtensibleManagedObject) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["ExtensibleManagedObject"] = reflect.TypeOf((*ExtensibleManagedObject)(nil)).Elem()
}

type ExtensionManager struct {
	Self types.ManagedObjectReference

	ExtensionList []types.Extension `mo:"extensionList"`
}

func (m ExtensionManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["ExtensionManager"] = reflect.TypeOf((*ExtensionManager)(nil)).Elem()
}

type FileManager struct {
	Self types.ManagedObjectReference
}

func (m FileManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["FileManager"] = reflect.TypeOf((*FileManager)(nil)).Elem()
}

type Folder struct {
	ManagedEntity

	ChildType   []string                       `mo:"childType"`
	ChildEntity []types.ManagedObjectReference `mo:"childEntity"`
}

func (m *Folder) Entity() *ManagedEntity {
	return &m.ManagedEntity
}

func init() {
	t["Folder"] = reflect.TypeOf((*Folder)(nil)).Elem()
}

type GuestAliasManager struct {
	Self types.ManagedObjectReference
}

func (m GuestAliasManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["GuestAliasManager"] = reflect.TypeOf((*GuestAliasManager)(nil)).Elem()
}

type GuestAuthManager struct {
	Self types.ManagedObjectReference
}

func (m GuestAuthManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["GuestAuthManager"] = reflect.TypeOf((*GuestAuthManager)(nil)).Elem()
}

type GuestFileManager struct {
	Self types.ManagedObjectReference
}

func (m GuestFileManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["GuestFileManager"] = reflect.TypeOf((*GuestFileManager)(nil)).Elem()
}

type GuestOperationsManager struct {
	Self types.ManagedObjectReference

	AuthManager                 *types.ManagedObjectReference `mo:"authManager"`
	FileManager                 *types.ManagedObjectReference `mo:"fileManager"`
	ProcessManager              *types.ManagedObjectReference `mo:"processManager"`
	GuestWindowsRegistryManager *types.ManagedObjectReference `mo:"guestWindowsRegistryManager"`
	AliasManager                *types.ManagedObjectReference `mo:"aliasManager"`
}

func (m GuestOperationsManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["GuestOperationsManager"] = reflect.TypeOf((*GuestOperationsManager)(nil)).Elem()
}

type GuestProcessManager struct {
	Self types.ManagedObjectReference
}

func (m GuestProcessManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["GuestProcessManager"] = reflect.TypeOf((*GuestProcessManager)(nil)).Elem()
}

type GuestWindowsRegistryManager struct {
	Self types.ManagedObjectReference
}

func (m GuestWindowsRegistryManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["GuestWindowsRegistryManager"] = reflect.TypeOf((*GuestWindowsRegistryManager)(nil)).Elem()
}

type HistoryCollector struct {
	Self types.ManagedObjectReference

	Filter types.AnyType `mo:"filter"`
}

func (m HistoryCollector) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HistoryCollector"] = reflect.TypeOf((*HistoryCollector)(nil)).Elem()
}

type HostAccessManager struct {
	Self types.ManagedObjectReference

	LockdownMode types.HostLockdownMode `mo:"lockdownMode"`
}

func (m HostAccessManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostAccessManager"] = reflect.TypeOf((*HostAccessManager)(nil)).Elem()
}

type HostActiveDirectoryAuthentication struct {
	HostDirectoryStore
}

func init() {
	t["HostActiveDirectoryAuthentication"] = reflect.TypeOf((*HostActiveDirectoryAuthentication)(nil)).Elem()
}

type HostAuthenticationManager struct {
	Self types.ManagedObjectReference

	Info           types.HostAuthenticationManagerInfo `mo:"info"`
	SupportedStore []types.ManagedObjectReference      `mo:"supportedStore"`
}

func (m HostAuthenticationManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostAuthenticationManager"] = reflect.TypeOf((*HostAuthenticationManager)(nil)).Elem()
}

type HostAuthenticationStore struct {
	Self types.ManagedObjectReference

	Info types.BaseHostAuthenticationStoreInfo `mo:"info"`
}

func (m HostAuthenticationStore) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostAuthenticationStore"] = reflect.TypeOf((*HostAuthenticationStore)(nil)).Elem()
}

type HostAutoStartManager struct {
	Self types.ManagedObjectReference

	Config types.HostAutoStartManagerConfig `mo:"config"`
}

func (m HostAutoStartManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostAutoStartManager"] = reflect.TypeOf((*HostAutoStartManager)(nil)).Elem()
}

type HostBootDeviceSystem struct {
	Self types.ManagedObjectReference
}

func (m HostBootDeviceSystem) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostBootDeviceSystem"] = reflect.TypeOf((*HostBootDeviceSystem)(nil)).Elem()
}

type HostCacheConfigurationManager struct {
	Self types.ManagedObjectReference

	CacheConfigurationInfo []types.HostCacheConfigurationInfo `mo:"cacheConfigurationInfo"`
}

func (m HostCacheConfigurationManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostCacheConfigurationManager"] = reflect.TypeOf((*HostCacheConfigurationManager)(nil)).Elem()
}

type HostCertificateManager struct {
	Self types.ManagedObjectReference

	CertificateInfo types.HostCertificateManagerCertificateInfo `mo:"certificateInfo"`
}

func (m HostCertificateManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostCertificateManager"] = reflect.TypeOf((*HostCertificateManager)(nil)).Elem()
}

type HostCpuSchedulerSystem struct {
	ExtensibleManagedObject

	HyperthreadInfo *types.HostHyperThreadScheduleInfo `mo:"hyperthreadInfo"`
}

func init() {
	t["HostCpuSchedulerSystem"] = reflect.TypeOf((*HostCpuSchedulerSystem)(nil)).Elem()
}

type HostDatastoreBrowser struct {
	Self types.ManagedObjectReference

	Datastore     []types.ManagedObjectReference `mo:"datastore"`
	SupportedType []types.BaseFileQuery          `mo:"supportedType"`
}

func (m HostDatastoreBrowser) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostDatastoreBrowser"] = reflect.TypeOf((*HostDatastoreBrowser)(nil)).Elem()
}

type HostDatastoreSystem struct {
	Self types.ManagedObjectReference

	Datastore    []types.ManagedObjectReference        `mo:"datastore"`
	Capabilities types.HostDatastoreSystemCapabilities `mo:"capabilities"`
}

func (m HostDatastoreSystem) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostDatastoreSystem"] = reflect.TypeOf((*HostDatastoreSystem)(nil)).Elem()
}

type HostDateTimeSystem struct {
	Self types.ManagedObjectReference

	DateTimeInfo types.HostDateTimeInfo `mo:"dateTimeInfo"`
}

func (m HostDateTimeSystem) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostDateTimeSystem"] = reflect.TypeOf((*HostDateTimeSystem)(nil)).Elem()
}

type HostDiagnosticSystem struct {
	Self types.ManagedObjectReference

	ActivePartition *types.HostDiagnosticPartition `mo:"activePartition"`
}

func (m HostDiagnosticSystem) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostDiagnosticSystem"] = reflect.TypeOf((*HostDiagnosticSystem)(nil)).Elem()
}

type HostDirectoryStore struct {
	HostAuthenticationStore
}

func init() {
	t["HostDirectoryStore"] = reflect.TypeOf((*HostDirectoryStore)(nil)).Elem()
}

type HostEsxAgentHostManager struct {
	Self types.ManagedObjectReference

	ConfigInfo types.HostEsxAgentHostManagerConfigInfo `mo:"configInfo"`
}

func (m HostEsxAgentHostManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostEsxAgentHostManager"] = reflect.TypeOf((*HostEsxAgentHostManager)(nil)).Elem()
}

type HostFirewallSystem struct {
	ExtensibleManagedObject

	FirewallInfo *types.HostFirewallInfo `mo:"firewallInfo"`
}

func init() {
	t["HostFirewallSystem"] = reflect.TypeOf((*HostFirewallSystem)(nil)).Elem()
}

type HostFirmwareSystem struct {
	Self types.ManagedObjectReference
}

func (m HostFirmwareSystem) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostFirmwareSystem"] = reflect.TypeOf((*HostFirmwareSystem)(nil)).Elem()
}

type HostGraphicsManager struct {
	ExtensibleManagedObject

	GraphicsInfo           []types.HostGraphicsInfo `mo:"graphicsInfo"`
	SharedPassthruGpuTypes []string                 `mo:"sharedPassthruGpuTypes"`
}

func init() {
	t["HostGraphicsManager"] = reflect.TypeOf((*HostGraphicsManager)(nil)).Elem()
}

type HostHealthStatusSystem struct {
	Self types.ManagedObjectReference

	Runtime types.HealthSystemRuntime `mo:"runtime"`
}

func (m HostHealthStatusSystem) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostHealthStatusSystem"] = reflect.TypeOf((*HostHealthStatusSystem)(nil)).Elem()
}

type HostImageConfigManager struct {
	Self types.ManagedObjectReference
}

func (m HostImageConfigManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostImageConfigManager"] = reflect.TypeOf((*HostImageConfigManager)(nil)).Elem()
}

type HostKernelModuleSystem struct {
	Self types.ManagedObjectReference
}

func (m HostKernelModuleSystem) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostKernelModuleSystem"] = reflect.TypeOf((*HostKernelModuleSystem)(nil)).Elem()
}

type HostLocalAccountManager struct {
	Self types.ManagedObjectReference
}

func (m HostLocalAccountManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostLocalAccountManager"] = reflect.TypeOf((*HostLocalAccountManager)(nil)).Elem()
}

type HostLocalAuthentication struct {
	HostAuthenticationStore
}

func init() {
	t["HostLocalAuthentication"] = reflect.TypeOf((*HostLocalAuthentication)(nil)).Elem()
}

type HostMemorySystem struct {
	ExtensibleManagedObject

	ConsoleReservationInfo        *types.ServiceConsoleReservationInfo       `mo:"consoleReservationInfo"`
	VirtualMachineReservationInfo *types.VirtualMachineMemoryReservationInfo `mo:"virtualMachineReservationInfo"`
}

func init() {
	t["HostMemorySystem"] = reflect.TypeOf((*HostMemorySystem)(nil)).Elem()
}

type HostNetworkSystem struct {
	ExtensibleManagedObject

	Capabilities         *types.HostNetCapabilities        `mo:"capabilities"`
	NetworkInfo          *types.HostNetworkInfo            `mo:"networkInfo"`
	OffloadCapabilities  *types.HostNetOffloadCapabilities `mo:"offloadCapabilities"`
	NetworkConfig        *types.HostNetworkConfig          `mo:"networkConfig"`
	DnsConfig            types.BaseHostDnsConfig           `mo:"dnsConfig"`
	IpRouteConfig        types.BaseHostIpRouteConfig       `mo:"ipRouteConfig"`
	ConsoleIpRouteConfig types.BaseHostIpRouteConfig       `mo:"consoleIpRouteConfig"`
}

func init() {
	t["HostNetworkSystem"] = reflect.TypeOf((*HostNetworkSystem)(nil)).Elem()
}

type HostPatchManager struct {
	Self types.ManagedObjectReference
}

func (m HostPatchManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostPatchManager"] = reflect.TypeOf((*HostPatchManager)(nil)).Elem()
}

type HostPciPassthruSystem struct {
	ExtensibleManagedObject

	PciPassthruInfo []types.BaseHostPciPassthruInfo `mo:"pciPassthruInfo"`
}

func init() {
	t["HostPciPassthruSystem"] = reflect.TypeOf((*HostPciPassthruSystem)(nil)).Elem()
}

type HostPowerSystem struct {
	Self types.ManagedObjectReference

	Capability types.PowerSystemCapability `mo:"capability"`
	Info       types.PowerSystemInfo       `mo:"info"`
}

func (m HostPowerSystem) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostPowerSystem"] = reflect.TypeOf((*HostPowerSystem)(nil)).Elem()
}

type HostProfile struct {
	Profile

	ReferenceHost *types.ManagedObjectReference `mo:"referenceHost"`
}

func init() {
	t["HostProfile"] = reflect.TypeOf((*HostProfile)(nil)).Elem()
}

type HostProfileManager struct {
	ProfileManager
}

func init() {
	t["HostProfileManager"] = reflect.TypeOf((*HostProfileManager)(nil)).Elem()
}

type HostServiceSystem struct {
	ExtensibleManagedObject

	ServiceInfo types.HostServiceInfo `mo:"serviceInfo"`
}

func init() {
	t["HostServiceSystem"] = reflect.TypeOf((*HostServiceSystem)(nil)).Elem()
}

type HostSnmpSystem struct {
	Self types.ManagedObjectReference

	Configuration types.HostSnmpConfigSpec        `mo:"configuration"`
	Limits        types.HostSnmpSystemAgentLimits `mo:"limits"`
}

func (m HostSnmpSystem) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostSnmpSystem"] = reflect.TypeOf((*HostSnmpSystem)(nil)).Elem()
}

type HostStorageSystem struct {
	ExtensibleManagedObject

	StorageDeviceInfo    *types.HostStorageDeviceInfo   `mo:"storageDeviceInfo"`
	FileSystemVolumeInfo types.HostFileSystemVolumeInfo `mo:"fileSystemVolumeInfo"`
	SystemFile           []string                       `mo:"systemFile"`
	MultipathStateInfo   *types.HostMultipathStateInfo  `mo:"multipathStateInfo"`
}

func init() {
	t["HostStorageSystem"] = reflect.TypeOf((*HostStorageSystem)(nil)).Elem()
}

type HostSystem struct {
	ManagedEntity

	Runtime            types.HostRuntimeInfo            `mo:"runtime"`
	Summary            types.HostListSummary            `mo:"summary"`
	Hardware           *types.HostHardwareInfo          `mo:"hardware"`
	Capability         *types.HostCapability            `mo:"capability"`
	LicensableResource types.HostLicensableResourceInfo `mo:"licensableResource"`
	ConfigManager      types.HostConfigManager          `mo:"configManager"`
	Config             *types.HostConfigInfo            `mo:"config"`
	Vm                 []types.ManagedObjectReference   `mo:"vm"`
	Datastore          []types.ManagedObjectReference   `mo:"datastore"`
	Network            []types.ManagedObjectReference   `mo:"network"`
	DatastoreBrowser   types.ManagedObjectReference     `mo:"datastoreBrowser"`
	SystemResources    *types.HostSystemResourceInfo    `mo:"systemResources"`
}

func (m *HostSystem) Entity() *ManagedEntity {
	return &m.ManagedEntity
}

func init() {
	t["HostSystem"] = reflect.TypeOf((*HostSystem)(nil)).Elem()
}

type HostVFlashManager struct {
	Self types.ManagedObjectReference

	VFlashConfigInfo *types.HostVFlashManagerVFlashConfigInfo `mo:"vFlashConfigInfo"`
}

func (m HostVFlashManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostVFlashManager"] = reflect.TypeOf((*HostVFlashManager)(nil)).Elem()
}

type HostVMotionSystem struct {
	ExtensibleManagedObject

	NetConfig *types.HostVMotionNetConfig `mo:"netConfig"`
	IpConfig  *types.HostIpConfig         `mo:"ipConfig"`
}

func init() {
	t["HostVMotionSystem"] = reflect.TypeOf((*HostVMotionSystem)(nil)).Elem()
}

type HostVirtualNicManager struct {
	ExtensibleManagedObject

	Info types.HostVirtualNicManagerInfo `mo:"info"`
}

func init() {
	t["HostVirtualNicManager"] = reflect.TypeOf((*HostVirtualNicManager)(nil)).Elem()
}

type HostVsanInternalSystem struct {
	Self types.ManagedObjectReference
}

func (m HostVsanInternalSystem) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostVsanInternalSystem"] = reflect.TypeOf((*HostVsanInternalSystem)(nil)).Elem()
}

type HostVsanSystem struct {
	Self types.ManagedObjectReference

	Config types.VsanHostConfigInfo `mo:"config"`
}

func (m HostVsanSystem) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HostVsanSystem"] = reflect.TypeOf((*HostVsanSystem)(nil)).Elem()
}

type HttpNfcLease struct {
	Self types.ManagedObjectReference

	InitializeProgress int32                       `mo:"initializeProgress"`
	Info               *types.HttpNfcLeaseInfo     `mo:"info"`
	State              types.HttpNfcLeaseState     `mo:"state"`
	Error              *types.LocalizedMethodFault `mo:"error"`
}

func (m HttpNfcLease) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["HttpNfcLease"] = reflect.TypeOf((*HttpNfcLease)(nil)).Elem()
}

type InventoryView struct {
	ManagedObjectView
}

func init() {
	t["InventoryView"] = reflect.TypeOf((*InventoryView)(nil)).Elem()
}

type IoFilterManager struct {
	Self types.ManagedObjectReference
}

func (m IoFilterManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["IoFilterManager"] = reflect.TypeOf((*IoFilterManager)(nil)).Elem()
}

type IpPoolManager struct {
	Self types.ManagedObjectReference
}

func (m IpPoolManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["IpPoolManager"] = reflect.TypeOf((*IpPoolManager)(nil)).Elem()
}

type IscsiManager struct {
	Self types.ManagedObjectReference
}

func (m IscsiManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["IscsiManager"] = reflect.TypeOf((*IscsiManager)(nil)).Elem()
}

type LicenseAssignmentManager struct {
	Self types.ManagedObjectReference
}

func (m LicenseAssignmentManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["LicenseAssignmentManager"] = reflect.TypeOf((*LicenseAssignmentManager)(nil)).Elem()
}

type LicenseManager struct {
	Self types.ManagedObjectReference

	Source                   types.BaseLicenseSource            `mo:"source"`
	SourceAvailable          bool                               `mo:"sourceAvailable"`
	Diagnostics              *types.LicenseDiagnostics          `mo:"diagnostics"`
	FeatureInfo              []types.LicenseFeatureInfo         `mo:"featureInfo"`
	LicensedEdition          string                             `mo:"licensedEdition"`
	Licenses                 []types.LicenseManagerLicenseInfo  `mo:"licenses"`
	LicenseAssignmentManager *types.ManagedObjectReference      `mo:"licenseAssignmentManager"`
	Evaluation               types.LicenseManagerEvaluationInfo `mo:"evaluation"`
}

func (m LicenseManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["LicenseManager"] = reflect.TypeOf((*LicenseManager)(nil)).Elem()
}

type ListView struct {
	ManagedObjectView
}

func init() {
	t["ListView"] = reflect.TypeOf((*ListView)(nil)).Elem()
}

type LocalizationManager struct {
	Self types.ManagedObjectReference

	Catalog []types.LocalizationManagerMessageCatalog `mo:"catalog"`
}

func (m LocalizationManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["LocalizationManager"] = reflect.TypeOf((*LocalizationManager)(nil)).Elem()
}

type ManagedEntity struct {
	ExtensibleManagedObject

	Parent              *types.ManagedObjectReference  `mo:"parent"`
	CustomValue         []types.BaseCustomFieldValue   `mo:"customValue"`
	OverallStatus       types.ManagedEntityStatus      `mo:"overallStatus"`
	ConfigStatus        types.ManagedEntityStatus      `mo:"configStatus"`
	ConfigIssue         []types.BaseEvent              `mo:"configIssue"`
	EffectiveRole       []int32                        `mo:"effectiveRole"`
	Permission          []types.Permission             `mo:"permission"`
	Name                string                         `mo:"name"`
	DisabledMethod      []string                       `mo:"disabledMethod"`
	RecentTask          []types.ManagedObjectReference `mo:"recentTask"`
	DeclaredAlarmState  []types.AlarmState             `mo:"declaredAlarmState"`
	TriggeredAlarmState []types.AlarmState             `mo:"triggeredAlarmState"`
	AlarmActionsEnabled *bool                          `mo:"alarmActionsEnabled"`
	Tag                 []types.Tag                    `mo:"tag"`
}

func init() {
	t["ManagedEntity"] = reflect.TypeOf((*ManagedEntity)(nil)).Elem()
}

type ManagedObjectView struct {
	Self types.ManagedObjectReference

	View []types.ManagedObjectReference `mo:"view"`
}

func (m ManagedObjectView) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["ManagedObjectView"] = reflect.TypeOf((*ManagedObjectView)(nil)).Elem()
}

type MessageBusProxy struct {
	Self types.ManagedObjectReference
}

func (m MessageBusProxy) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["MessageBusProxy"] = reflect.TypeOf((*MessageBusProxy)(nil)).Elem()
}

type Network struct {
	ManagedEntity

	Name    string                         `mo:"name"`
	Summary types.BaseNetworkSummary       `mo:"summary"`
	Host    []types.ManagedObjectReference `mo:"host"`
	Vm      []types.ManagedObjectReference `mo:"vm"`
}

func (m *Network) Entity() *ManagedEntity {
	return &m.ManagedEntity
}

func init() {
	t["Network"] = reflect.TypeOf((*Network)(nil)).Elem()
}

type OpaqueNetwork struct {
	Network
}

func init() {
	t["OpaqueNetwork"] = reflect.TypeOf((*OpaqueNetwork)(nil)).Elem()
}

type OptionManager struct {
	Self types.ManagedObjectReference

	SupportedOption []types.OptionDef       `mo:"supportedOption"`
	Setting         []types.BaseOptionValue `mo:"setting"`
}

func (m OptionManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["OptionManager"] = reflect.TypeOf((*OptionManager)(nil)).Elem()
}

type OverheadMemoryManager struct {
	Self types.ManagedObjectReference
}

func (m OverheadMemoryManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["OverheadMemoryManager"] = reflect.TypeOf((*OverheadMemoryManager)(nil)).Elem()
}

type OvfManager struct {
	Self types.ManagedObjectReference

	OvfImportOption []types.OvfOptionInfo `mo:"ovfImportOption"`
	OvfExportOption []types.OvfOptionInfo `mo:"ovfExportOption"`
}

func (m OvfManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["OvfManager"] = reflect.TypeOf((*OvfManager)(nil)).Elem()
}

type PerformanceManager struct {
	Self types.ManagedObjectReference

	Description        types.PerformanceDescription `mo:"description"`
	HistoricalInterval []types.PerfInterval         `mo:"historicalInterval"`
	PerfCounter        []types.PerfCounterInfo      `mo:"perfCounter"`
}

func (m PerformanceManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["PerformanceManager"] = reflect.TypeOf((*PerformanceManager)(nil)).Elem()
}

type Profile struct {
	Self types.ManagedObjectReference

	Config           types.BaseProfileConfigInfo    `mo:"config"`
	Description      *types.ProfileDescription      `mo:"description"`
	Name             string                         `mo:"name"`
	CreatedTime      time.Time                      `mo:"createdTime"`
	ModifiedTime     time.Time                      `mo:"modifiedTime"`
	Entity           []types.ManagedObjectReference `mo:"entity"`
	ComplianceStatus string                         `mo:"complianceStatus"`
}

func (m Profile) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["Profile"] = reflect.TypeOf((*Profile)(nil)).Elem()
}

type ProfileComplianceManager struct {
	Self types.ManagedObjectReference
}

func (m ProfileComplianceManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["ProfileComplianceManager"] = reflect.TypeOf((*ProfileComplianceManager)(nil)).Elem()
}

type ProfileManager struct {
	Self types.ManagedObjectReference

	Profile []types.ManagedObjectReference `mo:"profile"`
}

func (m ProfileManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["ProfileManager"] = reflect.TypeOf((*ProfileManager)(nil)).Elem()
}

type PropertyCollector struct {
	Self types.ManagedObjectReference

	Filter []types.ManagedObjectReference `mo:"filter"`
}

func (m PropertyCollector) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["PropertyCollector"] = reflect.TypeOf((*PropertyCollector)(nil)).Elem()
}

type PropertyFilter struct {
	Self types.ManagedObjectReference

	Spec           types.PropertyFilterSpec `mo:"spec"`
	PartialUpdates bool                     `mo:"partialUpdates"`
}

func (m PropertyFilter) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["PropertyFilter"] = reflect.TypeOf((*PropertyFilter)(nil)).Elem()
}

type ResourcePlanningManager struct {
	Self types.ManagedObjectReference
}

func (m ResourcePlanningManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["ResourcePlanningManager"] = reflect.TypeOf((*ResourcePlanningManager)(nil)).Elem()
}

type ResourcePool struct {
	ManagedEntity

	Summary            types.BaseResourcePoolSummary  `mo:"summary"`
	Runtime            types.ResourcePoolRuntimeInfo  `mo:"runtime"`
	Owner              types.ManagedObjectReference   `mo:"owner"`
	ResourcePool       []types.ManagedObjectReference `mo:"resourcePool"`
	Vm                 []types.ManagedObjectReference `mo:"vm"`
	Config             types.ResourceConfigSpec       `mo:"config"`
	ChildConfiguration []types.ResourceConfigSpec     `mo:"childConfiguration"`
}

func (m *ResourcePool) Entity() *ManagedEntity {
	return &m.ManagedEntity
}

func init() {
	t["ResourcePool"] = reflect.TypeOf((*ResourcePool)(nil)).Elem()
}

type ScheduledTask struct {
	ExtensibleManagedObject

	Info types.ScheduledTaskInfo `mo:"info"`
}

func init() {
	t["ScheduledTask"] = reflect.TypeOf((*ScheduledTask)(nil)).Elem()
}

type ScheduledTaskManager struct {
	Self types.ManagedObjectReference

	ScheduledTask []types.ManagedObjectReference `mo:"scheduledTask"`
	Description   types.ScheduledTaskDescription `mo:"description"`
}

func (m ScheduledTaskManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["ScheduledTaskManager"] = reflect.TypeOf((*ScheduledTaskManager)(nil)).Elem()
}

type SearchIndex struct {
	Self types.ManagedObjectReference
}

func (m SearchIndex) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["SearchIndex"] = reflect.TypeOf((*SearchIndex)(nil)).Elem()
}

type ServiceInstance struct {
	Self types.ManagedObjectReference

	ServerClock time.Time            `mo:"serverClock"`
	Capability  types.Capability     `mo:"capability"`
	Content     types.ServiceContent `mo:"content"`
}

func (m ServiceInstance) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["ServiceInstance"] = reflect.TypeOf((*ServiceInstance)(nil)).Elem()
}

type ServiceManager struct {
	Self types.ManagedObjectReference

	Service []types.ServiceManagerServiceInfo `mo:"service"`
}

func (m ServiceManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["ServiceManager"] = reflect.TypeOf((*ServiceManager)(nil)).Elem()
}

type SessionManager struct {
	Self types.ManagedObjectReference

	SessionList         []types.UserSession `mo:"sessionList"`
	CurrentSession      *types.UserSession  `mo:"currentSession"`
	Message             *string             `mo:"message"`
	MessageLocaleList   []string            `mo:"messageLocaleList"`
	SupportedLocaleList []string            `mo:"supportedLocaleList"`
	DefaultLocale       string              `mo:"defaultLocale"`
}

func (m SessionManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["SessionManager"] = reflect.TypeOf((*SessionManager)(nil)).Elem()
}

type SimpleCommand struct {
	Self types.ManagedObjectReference

	EncodingType types.SimpleCommandEncoding     `mo:"encodingType"`
	Entity       types.ServiceManagerServiceInfo `mo:"entity"`
}

func (m SimpleCommand) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["SimpleCommand"] = reflect.TypeOf((*SimpleCommand)(nil)).Elem()
}

type StoragePod struct {
	Folder

	Summary            *types.StoragePodSummary  `mo:"summary"`
	PodStorageDrsEntry *types.PodStorageDrsEntry `mo:"podStorageDrsEntry"`
}

func init() {
	t["StoragePod"] = reflect.TypeOf((*StoragePod)(nil)).Elem()
}

type StorageResourceManager struct {
	Self types.ManagedObjectReference
}

func (m StorageResourceManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["StorageResourceManager"] = reflect.TypeOf((*StorageResourceManager)(nil)).Elem()
}

type Task struct {
	ExtensibleManagedObject

	Info types.TaskInfo `mo:"info"`
}

func init() {
	t["Task"] = reflect.TypeOf((*Task)(nil)).Elem()
}

type TaskHistoryCollector struct {
	HistoryCollector

	LatestPage []types.TaskInfo `mo:"latestPage"`
}

func init() {
	t["TaskHistoryCollector"] = reflect.TypeOf((*TaskHistoryCollector)(nil)).Elem()
}

type TaskManager struct {
	Self types.ManagedObjectReference

	RecentTask   []types.ManagedObjectReference `mo:"recentTask"`
	Description  types.TaskDescription          `mo:"description"`
	MaxCollector int32                          `mo:"maxCollector"`
}

func (m TaskManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["TaskManager"] = reflect.TypeOf((*TaskManager)(nil)).Elem()
}

type UserDirectory struct {
	Self types.ManagedObjectReference

	DomainList []string `mo:"domainList"`
}

func (m UserDirectory) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["UserDirectory"] = reflect.TypeOf((*UserDirectory)(nil)).Elem()
}

type VRPResourceManager struct {
	Self types.ManagedObjectReference
}

func (m VRPResourceManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["VRPResourceManager"] = reflect.TypeOf((*VRPResourceManager)(nil)).Elem()
}

type View struct {
	Self types.ManagedObjectReference
}

func (m View) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["View"] = reflect.TypeOf((*View)(nil)).Elem()
}

type ViewManager struct {
	Self types.ManagedObjectReference

	ViewList []types.ManagedObjectReference `mo:"viewList"`
}

func (m ViewManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["ViewManager"] = reflect.TypeOf((*ViewManager)(nil)).Elem()
}

type VirtualApp struct {
	ResourcePool

	ParentFolder *types.ManagedObjectReference  `mo:"parentFolder"`
	Datastore    []types.ManagedObjectReference `mo:"datastore"`
	Network      []types.ManagedObjectReference `mo:"network"`
	VAppConfig   *types.VAppConfigInfo          `mo:"vAppConfig"`
	ParentVApp   *types.ManagedObjectReference  `mo:"parentVApp"`
	ChildLink    []types.VirtualAppLinkInfo     `mo:"childLink"`
}

func init() {
	t["VirtualApp"] = reflect.TypeOf((*VirtualApp)(nil)).Elem()
}

type VirtualDiskManager struct {
	Self types.ManagedObjectReference
}

func (m VirtualDiskManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["VirtualDiskManager"] = reflect.TypeOf((*VirtualDiskManager)(nil)).Elem()
}

type VirtualMachine struct {
	ManagedEntity

	Capability           types.VirtualMachineCapability    `mo:"capability"`
	Config               *types.VirtualMachineConfigInfo   `mo:"config"`
	Layout               *types.VirtualMachineFileLayout   `mo:"layout"`
	LayoutEx             *types.VirtualMachineFileLayoutEx `mo:"layoutEx"`
	Storage              *types.VirtualMachineStorageInfo  `mo:"storage"`
	EnvironmentBrowser   types.ManagedObjectReference      `mo:"environmentBrowser"`
	ResourcePool         *types.ManagedObjectReference     `mo:"resourcePool"`
	ParentVApp           *types.ManagedObjectReference     `mo:"parentVApp"`
	ResourceConfig       *types.ResourceConfigSpec         `mo:"resourceConfig"`
	Runtime              types.VirtualMachineRuntimeInfo   `mo:"runtime"`
	Guest                *types.GuestInfo                  `mo:"guest"`
	Summary              types.VirtualMachineSummary       `mo:"summary"`
	Datastore            []types.ManagedObjectReference    `mo:"datastore"`
	Network              []types.ManagedObjectReference    `mo:"network"`
	Snapshot             *types.VirtualMachineSnapshotInfo `mo:"snapshot"`
	RootSnapshot         []types.ManagedObjectReference    `mo:"rootSnapshot"`
	GuestHeartbeatStatus types.ManagedEntityStatus         `mo:"guestHeartbeatStatus"`
}

func (m *VirtualMachine) Entity() *ManagedEntity {
	return &m.ManagedEntity
}

func init() {
	t["VirtualMachine"] = reflect.TypeOf((*VirtualMachine)(nil)).Elem()
}

type VirtualMachineCompatibilityChecker struct {
	Self types.ManagedObjectReference
}

func (m VirtualMachineCompatibilityChecker) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["VirtualMachineCompatibilityChecker"] = reflect.TypeOf((*VirtualMachineCompatibilityChecker)(nil)).Elem()
}

type VirtualMachineProvisioningChecker struct {
	Self types.ManagedObjectReference
}

func (m VirtualMachineProvisioningChecker) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["VirtualMachineProvisioningChecker"] = reflect.TypeOf((*VirtualMachineProvisioningChecker)(nil)).Elem()
}

type VirtualMachineSnapshot struct {
	ExtensibleManagedObject

	Config        types.VirtualMachineConfigInfo `mo:"config"`
	ChildSnapshot []types.ManagedObjectReference `mo:"childSnapshot"`
	Vm            types.ManagedObjectReference   `mo:"vm"`
}

func init() {
	t["VirtualMachineSnapshot"] = reflect.TypeOf((*VirtualMachineSnapshot)(nil)).Elem()
}

type VirtualizationManager struct {
	Self types.ManagedObjectReference
}

func (m VirtualizationManager) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["VirtualizationManager"] = reflect.TypeOf((*VirtualizationManager)(nil)).Elem()
}

type VmwareDistributedVirtualSwitch struct {
	DistributedVirtualSwitch
}

func init() {
	t["VmwareDistributedVirtualSwitch"] = reflect.TypeOf((*VmwareDistributedVirtualSwitch)(nil)).Elem()
}

type VsanUpgradeSystem struct {
	Self types.ManagedObjectReference
}

func (m VsanUpgradeSystem) Reference() types.ManagedObjectReference {
	return m.Self
}

func init() {
	t["VsanUpgradeSystem"] = reflect.TypeOf((*VsanUpgradeSystem)(nil)).Elem()
}
