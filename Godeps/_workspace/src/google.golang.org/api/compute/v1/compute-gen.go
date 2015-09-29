// Package compute provides access to the Compute Engine API.
//
// See https://developers.google.com/compute/docs/reference/latest/
//
// Usage example:
//
//   import "google.golang.org/api/compute/v1"
//   ...
//   computeService, err := compute.New(oauthHttpClient)
package compute

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"golang.org/x/net/context"
	"google.golang.org/api/googleapi"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// Always reference these packages, just in case the auto-generated code
// below doesn't.
var _ = bytes.NewBuffer
var _ = strconv.Itoa
var _ = fmt.Sprintf
var _ = json.NewDecoder
var _ = io.Copy
var _ = url.Parse
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace
var _ = context.Background

const apiId = "compute:v1"
const apiName = "compute"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/compute/v1/projects/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View and manage your Google Compute Engine resources
	ComputeScope = "https://www.googleapis.com/auth/compute"

	// View your Google Compute Engine resources
	ComputeReadonlyScope = "https://www.googleapis.com/auth/compute.readonly"

	// Manage your data and permissions in Google Cloud Storage
	DevstorageFullControlScope = "https://www.googleapis.com/auth/devstorage.full_control"

	// View your data in Google Cloud Storage
	DevstorageReadOnlyScope = "https://www.googleapis.com/auth/devstorage.read_only"

	// Manage your data in Google Cloud Storage
	DevstorageReadWriteScope = "https://www.googleapis.com/auth/devstorage.read_write"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Addresses = NewAddressesService(s)
	s.Autoscalers = NewAutoscalersService(s)
	s.BackendServices = NewBackendServicesService(s)
	s.DiskTypes = NewDiskTypesService(s)
	s.Disks = NewDisksService(s)
	s.Firewalls = NewFirewallsService(s)
	s.ForwardingRules = NewForwardingRulesService(s)
	s.GlobalAddresses = NewGlobalAddressesService(s)
	s.GlobalForwardingRules = NewGlobalForwardingRulesService(s)
	s.GlobalOperations = NewGlobalOperationsService(s)
	s.HttpHealthChecks = NewHttpHealthChecksService(s)
	s.Images = NewImagesService(s)
	s.InstanceGroupManagers = NewInstanceGroupManagersService(s)
	s.InstanceGroups = NewInstanceGroupsService(s)
	s.InstanceTemplates = NewInstanceTemplatesService(s)
	s.Instances = NewInstancesService(s)
	s.Licenses = NewLicensesService(s)
	s.MachineTypes = NewMachineTypesService(s)
	s.Networks = NewNetworksService(s)
	s.Projects = NewProjectsService(s)
	s.RegionOperations = NewRegionOperationsService(s)
	s.Regions = NewRegionsService(s)
	s.Routes = NewRoutesService(s)
	s.Snapshots = NewSnapshotsService(s)
	s.TargetHttpProxies = NewTargetHttpProxiesService(s)
	s.TargetInstances = NewTargetInstancesService(s)
	s.TargetPools = NewTargetPoolsService(s)
	s.TargetVpnGateways = NewTargetVpnGatewaysService(s)
	s.UrlMaps = NewUrlMapsService(s)
	s.VpnTunnels = NewVpnTunnelsService(s)
	s.ZoneOperations = NewZoneOperationsService(s)
	s.Zones = NewZonesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Addresses *AddressesService

	Autoscalers *AutoscalersService

	BackendServices *BackendServicesService

	DiskTypes *DiskTypesService

	Disks *DisksService

	Firewalls *FirewallsService

	ForwardingRules *ForwardingRulesService

	GlobalAddresses *GlobalAddressesService

	GlobalForwardingRules *GlobalForwardingRulesService

	GlobalOperations *GlobalOperationsService

	HttpHealthChecks *HttpHealthChecksService

	Images *ImagesService

	InstanceGroupManagers *InstanceGroupManagersService

	InstanceGroups *InstanceGroupsService

	InstanceTemplates *InstanceTemplatesService

	Instances *InstancesService

	Licenses *LicensesService

	MachineTypes *MachineTypesService

	Networks *NetworksService

	Projects *ProjectsService

	RegionOperations *RegionOperationsService

	Regions *RegionsService

	Routes *RoutesService

	Snapshots *SnapshotsService

	TargetHttpProxies *TargetHttpProxiesService

	TargetInstances *TargetInstancesService

	TargetPools *TargetPoolsService

	TargetVpnGateways *TargetVpnGatewaysService

	UrlMaps *UrlMapsService

	VpnTunnels *VpnTunnelsService

	ZoneOperations *ZoneOperationsService

	Zones *ZonesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewAddressesService(s *Service) *AddressesService {
	rs := &AddressesService{s: s}
	return rs
}

type AddressesService struct {
	s *Service
}

func NewAutoscalersService(s *Service) *AutoscalersService {
	rs := &AutoscalersService{s: s}
	return rs
}

type AutoscalersService struct {
	s *Service
}

func NewBackendServicesService(s *Service) *BackendServicesService {
	rs := &BackendServicesService{s: s}
	return rs
}

type BackendServicesService struct {
	s *Service
}

func NewDiskTypesService(s *Service) *DiskTypesService {
	rs := &DiskTypesService{s: s}
	return rs
}

type DiskTypesService struct {
	s *Service
}

func NewDisksService(s *Service) *DisksService {
	rs := &DisksService{s: s}
	return rs
}

type DisksService struct {
	s *Service
}

func NewFirewallsService(s *Service) *FirewallsService {
	rs := &FirewallsService{s: s}
	return rs
}

type FirewallsService struct {
	s *Service
}

func NewForwardingRulesService(s *Service) *ForwardingRulesService {
	rs := &ForwardingRulesService{s: s}
	return rs
}

type ForwardingRulesService struct {
	s *Service
}

func NewGlobalAddressesService(s *Service) *GlobalAddressesService {
	rs := &GlobalAddressesService{s: s}
	return rs
}

type GlobalAddressesService struct {
	s *Service
}

func NewGlobalForwardingRulesService(s *Service) *GlobalForwardingRulesService {
	rs := &GlobalForwardingRulesService{s: s}
	return rs
}

type GlobalForwardingRulesService struct {
	s *Service
}

func NewGlobalOperationsService(s *Service) *GlobalOperationsService {
	rs := &GlobalOperationsService{s: s}
	return rs
}

type GlobalOperationsService struct {
	s *Service
}

func NewHttpHealthChecksService(s *Service) *HttpHealthChecksService {
	rs := &HttpHealthChecksService{s: s}
	return rs
}

type HttpHealthChecksService struct {
	s *Service
}

func NewImagesService(s *Service) *ImagesService {
	rs := &ImagesService{s: s}
	return rs
}

type ImagesService struct {
	s *Service
}

func NewInstanceGroupManagersService(s *Service) *InstanceGroupManagersService {
	rs := &InstanceGroupManagersService{s: s}
	return rs
}

type InstanceGroupManagersService struct {
	s *Service
}

func NewInstanceGroupsService(s *Service) *InstanceGroupsService {
	rs := &InstanceGroupsService{s: s}
	return rs
}

type InstanceGroupsService struct {
	s *Service
}

func NewInstanceTemplatesService(s *Service) *InstanceTemplatesService {
	rs := &InstanceTemplatesService{s: s}
	return rs
}

type InstanceTemplatesService struct {
	s *Service
}

func NewInstancesService(s *Service) *InstancesService {
	rs := &InstancesService{s: s}
	return rs
}

type InstancesService struct {
	s *Service
}

func NewLicensesService(s *Service) *LicensesService {
	rs := &LicensesService{s: s}
	return rs
}

type LicensesService struct {
	s *Service
}

func NewMachineTypesService(s *Service) *MachineTypesService {
	rs := &MachineTypesService{s: s}
	return rs
}

type MachineTypesService struct {
	s *Service
}

func NewNetworksService(s *Service) *NetworksService {
	rs := &NetworksService{s: s}
	return rs
}

type NetworksService struct {
	s *Service
}

func NewProjectsService(s *Service) *ProjectsService {
	rs := &ProjectsService{s: s}
	return rs
}

type ProjectsService struct {
	s *Service
}

func NewRegionOperationsService(s *Service) *RegionOperationsService {
	rs := &RegionOperationsService{s: s}
	return rs
}

type RegionOperationsService struct {
	s *Service
}

func NewRegionsService(s *Service) *RegionsService {
	rs := &RegionsService{s: s}
	return rs
}

type RegionsService struct {
	s *Service
}

func NewRoutesService(s *Service) *RoutesService {
	rs := &RoutesService{s: s}
	return rs
}

type RoutesService struct {
	s *Service
}

func NewSnapshotsService(s *Service) *SnapshotsService {
	rs := &SnapshotsService{s: s}
	return rs
}

type SnapshotsService struct {
	s *Service
}

func NewTargetHttpProxiesService(s *Service) *TargetHttpProxiesService {
	rs := &TargetHttpProxiesService{s: s}
	return rs
}

type TargetHttpProxiesService struct {
	s *Service
}

func NewTargetInstancesService(s *Service) *TargetInstancesService {
	rs := &TargetInstancesService{s: s}
	return rs
}

type TargetInstancesService struct {
	s *Service
}

func NewTargetPoolsService(s *Service) *TargetPoolsService {
	rs := &TargetPoolsService{s: s}
	return rs
}

type TargetPoolsService struct {
	s *Service
}

func NewTargetVpnGatewaysService(s *Service) *TargetVpnGatewaysService {
	rs := &TargetVpnGatewaysService{s: s}
	return rs
}

type TargetVpnGatewaysService struct {
	s *Service
}

func NewUrlMapsService(s *Service) *UrlMapsService {
	rs := &UrlMapsService{s: s}
	return rs
}

type UrlMapsService struct {
	s *Service
}

func NewVpnTunnelsService(s *Service) *VpnTunnelsService {
	rs := &VpnTunnelsService{s: s}
	return rs
}

type VpnTunnelsService struct {
	s *Service
}

func NewZoneOperationsService(s *Service) *ZoneOperationsService {
	rs := &ZoneOperationsService{s: s}
	return rs
}

type ZoneOperationsService struct {
	s *Service
}

func NewZonesService(s *Service) *ZonesService {
	rs := &ZonesService{s: s}
	return rs
}

type ZonesService struct {
	s *Service
}

// AccessConfig: An access configuration attached to an instance's
// network interface.
type AccessConfig struct {
	// Kind: [Output Only] Type of the resource. Always compute#accessConfig
	// for access configs.
	Kind string `json:"kind,omitempty"`

	// Name: Name of this access configuration.
	Name string `json:"name,omitempty"`

	// NatIP: An external IP address associated with this instance. Specify
	// an unused static external IP address available to the project or
	// leave this field undefined to use an IP from a shared ephemeral IP
	// address pool. If you specify a static external IP address, it must
	// live in the same region as the zone of the instance.
	NatIP string `json:"natIP,omitempty"`

	// Type: The type of configuration. The default and only option is
	// ONE_TO_ONE_NAT.
	//
	// Possible values:
	//   "ONE_TO_ONE_NAT" (default)
	Type string `json:"type,omitempty"`
}

// Address: A reserved address resource.
type Address struct {
	// Address: The static external IP address represented by this resource.
	Address string `json:"address,omitempty"`

	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of the resource. Always compute#address for
	// addresses.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035. Specifically, the name must be 1-63 characters long and
	// match the regular expression [a-z]([-a-z0-9]*[a-z0-9])? which means
	// the first character must be a lowercase letter, and all following
	// characters must be a dash, lowercase letter, or digit, except the
	// last character, which cannot be a dash.
	Name string `json:"name,omitempty"`

	// Region: [Output Only] URL of the region where the regional address
	// resides. This field is not applicable to global addresses.
	Region string `json:"region,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Status: [Output Only] The status of the address, which can be either
	// IN_USE or RESERVED. An address that is RESERVED is currently reserved
	// and available to use. An IN_USE address is currently being used by
	// another resource and is not available.
	//
	// Possible values:
	//   "IN_USE"
	//   "RESERVED"
	Status string `json:"status,omitempty"`

	// Users: [Output Only] The URLs of the resources that are using this
	// address.
	Users []string `json:"users,omitempty"`
}

type AddressAggregatedList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A map of scoped address lists.
	Items map[string]AddressesScopedList `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always
	// compute#addressAggregatedList for aggregated lists of addresses.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// AddressList: Contains a list of address resources.
type AddressList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of Address resources.
	Items []*Address `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#addressList for
	// lists of addresses.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type AddressesScopedList struct {
	// Addresses: [Output Only] List of addresses contained in this scope.
	Addresses []*Address `json:"addresses,omitempty"`

	// Warning: [Output Only] Informational warning which replaces the list
	// of addresses when the list is empty.
	Warning *AddressesScopedListWarning `json:"warning,omitempty"`
}

// AddressesScopedListWarning: [Output Only] Informational warning which
// replaces the list of addresses when the list is empty.
type AddressesScopedListWarning struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*AddressesScopedListWarningData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type AddressesScopedListWarningData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

// AttachedDisk: An instance-attached disk resource.
type AttachedDisk struct {
	// AutoDelete: Specifies whether the disk will be auto-deleted when the
	// instance is deleted (but not when the disk is detached from the
	// instance).
	AutoDelete bool `json:"autoDelete,omitempty"`

	// Boot: Indicates that this is a boot disk. The virtual machine will
	// use the first partition of the disk for its root filesystem.
	Boot bool `json:"boot,omitempty"`

	// DeviceName: Specifies a unique device name of your choice that is
	// reflected into the /dev/ tree of a Linux operating system running
	// within the instance. This name can be used to reference the device
	// for mounting, resizing, and so on, from within the instance.
	//
	// If not specified, the server chooses a default device name to apply
	// to this disk, in the form persistent-disks-x, where x is a number
	// assigned by Google Compute Engine. This field is only applicable for
	// persistent disks.
	DeviceName string `json:"deviceName,omitempty"`

	// Index: Assigns a zero-based index to this disk, where 0 is reserved
	// for the boot disk. For example, if you have many disks attached to an
	// instance, each disk would have a unique index number. If not
	// specified, the server will choose an appropriate value.
	Index int64 `json:"index,omitempty"`

	// InitializeParams: [Input Only] Specifies the parameters for a new
	// disk that will be created alongside the new instance. Use
	// initialization parameters to create boot disks or local SSDs attached
	// to the new instance.
	//
	// This property is mutually exclusive with the source property; you can
	// only define one or the other, but not both.
	InitializeParams *AttachedDiskInitializeParams `json:"initializeParams,omitempty"`

	// Possible values:
	//   "NVME"
	//   "SCSI"
	Interface string `json:"interface,omitempty"`

	// Kind: [Output Only] Type of the resource. Always compute#attachedDisk
	// for attached disks.
	Kind string `json:"kind,omitempty"`

	// Licenses: [Output Only] Any valid publicly visible licenses.
	Licenses []string `json:"licenses,omitempty"`

	// Mode: The mode in which to attach this disk, either READ_WRITE or
	// READ_ONLY. If not specified, the default is to attach the disk in
	// READ_WRITE mode.
	//
	// Possible values:
	//   "READ_ONLY"
	//   "READ_WRITE"
	Mode string `json:"mode,omitempty"`

	// Source: Specifies a valid partial or full URL to an existing
	// Persistent Disk resource. This field is only applicable for
	// persistent disks.
	Source string `json:"source,omitempty"`

	// Type: Specifies the type of the disk, either SCRATCH or PERSISTENT.
	// If not specified, the default is PERSISTENT.
	//
	// Possible values:
	//   "PERSISTENT"
	//   "SCRATCH"
	Type string `json:"type,omitempty"`
}

// AttachedDiskInitializeParams: [Input Only] Specifies the parameters
// for a new disk that will be created alongside the new instance. Use
// initialization parameters to create boot disks or local SSDs attached
// to the new instance.
//
// This property is mutually exclusive with the source property; you can
// only define one or the other, but not both.
type AttachedDiskInitializeParams struct {
	// DiskName: Specifies the disk name. If not specified, the default is
	// to use the name of the instance.
	DiskName string `json:"diskName,omitempty"`

	// DiskSizeGb: Specifies the size of the disk in base-2 GB.
	DiskSizeGb int64 `json:"diskSizeGb,omitempty,string"`

	// DiskType: Specifies the disk type to use to create the instance. If
	// not specified, the default is pd-standard, specified using the full
	// URL. For
	// example:
	//
	// https://www.googleapis.com/compute/v1/projects/project/zones
	// /zone/diskTypes/pd-standard
	//
	// Other values include pd-ssd and local-ssd. If you define this field,
	// you can provide either the full or partial URL. For example, the
	// following are valid values:
	// -
	// https://www.googleapis.com/compute/v1/projects/project/zones/zone/diskTypes/diskType
	// - projects/project/zones/zone/diskTypes/diskType
	// - zones/zone/diskTypes/diskType
	DiskType string `json:"diskType,omitempty"`

	// SourceImage: A source image used to create the disk. You can provide
	// a private (custom) image, and Compute Engine will use the
	// corresponding image from your project. For
	// example:
	//
	// global/images/my-private-image
	//
	// Or you can provide an image from a publicly-available project. For
	// example, to use a Debian image from the debian-cloud project, make
	// sure to include the project in the
	// URL:
	//
	// projects/debian-cloud/global/images/debian-7-wheezy-vYYYYMMDD
	//
	// where vYYYYMMDD is the image version. The fully-qualified URL will
	// also work in both cases.
	SourceImage string `json:"sourceImage,omitempty"`
}

type Autoscaler struct {
	// AutoscalingPolicy: Autoscaling configuration.
	AutoscalingPolicy *AutoscalingPolicy `json:"autoscalingPolicy,omitempty"`

	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id uint64 `json:"id,omitempty,string"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035.
	Name string `json:"name,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// Target: URL of Instance Group Manager or Replica Pool which will be
	// controlled by Autoscaler.
	Target string `json:"target,omitempty"`

	// Zone: URL of the zone where the instance group resides (output only).
	Zone string `json:"zone,omitempty"`
}

type AutoscalerAggregatedList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A map of scoped autoscaler lists.
	Items map[string]AutoscalersScopedList `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

// AutoscalerList: Contains a list of persistent autoscaler resources.
type AutoscalerList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A list of Autoscaler resources.
	Items []*Autoscaler `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type AutoscalersScopedList struct {
	// Autoscalers: List of autoscalers contained in this scope.
	Autoscalers []*Autoscaler `json:"autoscalers,omitempty"`

	// Warning: Informational warning which replaces the list of autoscalers
	// when the list is empty.
	Warning *AutoscalersScopedListWarning `json:"warning,omitempty"`
}

// AutoscalersScopedListWarning: Informational warning which replaces
// the list of autoscalers when the list is empty.
type AutoscalersScopedListWarning struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*AutoscalersScopedListWarningData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type AutoscalersScopedListWarningData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

// AutoscalingPolicy: Cloud Autoscaler policy.
type AutoscalingPolicy struct {
	// CoolDownPeriodSec: The number of seconds that the Autoscaler should
	// wait between two succeeding changes to the number of virtual
	// machines. You should define an interval that is at least as long as
	// the initialization time of a virtual machine and the time it may take
	// for replica pool to create the virtual machine. The default is 60
	// seconds.
	CoolDownPeriodSec int64 `json:"coolDownPeriodSec,omitempty"`

	// CpuUtilization: TODO(jbartosik): Add support for scaling based on
	// muliple utilization metrics (take max recommendation). Exactly one
	// utilization policy should be provided. Configuration parameters of
	// CPU based autoscaling policy.
	CpuUtilization *AutoscalingPolicyCpuUtilization `json:"cpuUtilization,omitempty"`

	// CustomMetricUtilizations: Configuration parameters of autoscaling
	// based on custom metric.
	CustomMetricUtilizations []*AutoscalingPolicyCustomMetricUtilization `json:"customMetricUtilizations,omitempty"`

	// LoadBalancingUtilization: Configuration parameters of autoscaling
	// based on load balancer.
	LoadBalancingUtilization *AutoscalingPolicyLoadBalancingUtilization `json:"loadBalancingUtilization,omitempty"`

	// MaxNumReplicas: The maximum number of replicas that the Autoscaler
	// can scale up to. This field is required for config to be effective.
	// Maximum number of replicas should be not lower than minimal number of
	// replicas. Absolute limit for this value is defined in Autoscaler
	// backend.
	MaxNumReplicas int64 `json:"maxNumReplicas,omitempty"`

	// MinNumReplicas: The minimum number of replicas that the Autoscaler
	// can scale down to. Can't be less than 0. If not provided Autoscaler
	// will choose default value depending on maximal number of replicas.
	MinNumReplicas int64 `json:"minNumReplicas,omitempty"`
}

// AutoscalingPolicyCpuUtilization: CPU utilization policy.
type AutoscalingPolicyCpuUtilization struct {
	// UtilizationTarget: The target utilization that the Autoscaler should
	// maintain. It is represented as a fraction of used cores. For example:
	// 6 cores used in 8-core VM are represented here as 0.75. Must be a
	// float value between (0, 1]. If not defined, the default is 0.8.
	UtilizationTarget float64 `json:"utilizationTarget,omitempty"`
}

// AutoscalingPolicyCustomMetricUtilization: Custom utilization metric
// policy.
type AutoscalingPolicyCustomMetricUtilization struct {
	// Metric: Identifier of the metric. It should be a Cloud Monitoring
	// metric. The metric can not have negative values. The metric should be
	// an utilization metric (increasing number of VMs handling requests x
	// times should reduce average value of the metric roughly x times). For
	// example you could use:
	// compute.googleapis.com/instance/network/received_bytes_count.
	Metric string `json:"metric,omitempty"`

	// UtilizationTarget: Target value of the metric which Autoscaler should
	// maintain. Must be a positive value.
	UtilizationTarget float64 `json:"utilizationTarget,omitempty"`

	// UtilizationTargetType: Defines type in which utilization_target is
	// expressed.
	//
	// Possible values:
	//   "DELTA_PER_MINUTE"
	//   "DELTA_PER_SECOND"
	//   "GAUGE"
	UtilizationTargetType string `json:"utilizationTargetType,omitempty"`
}

// AutoscalingPolicyLoadBalancingUtilization: Load balancing utilization
// policy.
type AutoscalingPolicyLoadBalancingUtilization struct {
	// UtilizationTarget: Fraction of backend capacity utilization (set in
	// HTTP load balancing configuration) that Autoscaler should maintain.
	// Must be a positive float value. If not defined, the default is 0.8.
	// For example if your maxRatePerInstance capacity (in HTTP Load
	// Balancing configuration) is set at 10 and you would like to keep
	// number of instances such that each instance receives 7 QPS on
	// average, set this to 0.7.
	UtilizationTarget float64 `json:"utilizationTarget,omitempty"`
}

// Backend: Message containing information of one individual backend.
type Backend struct {
	// BalancingMode: Specifies the balancing mode for this backend. The
	// default is UTILIZATION but available values are UTILIZATION and RATE.
	//
	// Possible values:
	//   "RATE"
	//   "UTILIZATION"
	BalancingMode string `json:"balancingMode,omitempty"`

	// CapacityScaler: A multiplier applied to the group's maximum servicing
	// capacity (either UTILIZATION or RATE). Default value is 1, which
	// means the group will serve up to 100% of its configured CPU or RPS
	// (depending on balancingMode). A setting of 0 means the group is
	// completely drained, offering 0% of its available CPU or RPS. Valid
	// range is [0.0,1.0].
	CapacityScaler float64 `json:"capacityScaler,omitempty"`

	// Description: An optional textual description of the resource.
	// Provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// Group: The fully-qualified URL of a zonal Instance Group resource.
	// This instance group defines the list of instances that serve traffic.
	// Member virtual machine instances from each instance group must live
	// in the same zone as the instance group itself. No two backends in a
	// backend service are allowed to use same Instance Group
	// resource.
	//
	// Note that you must specify an Instance Group resource using the
	// fully-qualified URL, rather than a partial URL.
	Group string `json:"group,omitempty"`

	// MaxRate: The max RPS of the group. Can be used with either balancing
	// mode, but required if RATE mode. For RATE mode, either maxRate or
	// maxRatePerInstance must be set.
	MaxRate int64 `json:"maxRate,omitempty"`

	// MaxRatePerInstance: The max RPS that a single backed instance can
	// handle. This is used to calculate the capacity of the group. Can be
	// used in either balancing mode. For RATE mode, either maxRate or
	// maxRatePerInstance must be set.
	MaxRatePerInstance float64 `json:"maxRatePerInstance,omitempty"`

	// MaxUtilization: Used when balancingMode is UTILIZATION. This ratio
	// defines the CPU utilization target for the group. The default is 0.8.
	// Valid range is [0.0, 1.0].
	MaxUtilization float64 `json:"maxUtilization,omitempty"`
}

// BackendService: A BackendService resource. This resource defines a
// group of backend VMs together with their serving capacity.
type BackendService struct {
	// Backends: The list of backends that serve this BackendService.
	Backends []*Backend `json:"backends,omitempty"`

	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource.
	// Provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// Fingerprint: Fingerprint of this resource. A hash of the contents
	// stored in this object. This field is used in optimistic locking. This
	// field will be ignored when inserting a BackendService. An up-to-date
	// fingerprint must be provided in order to update the BackendService.
	Fingerprint string `json:"fingerprint,omitempty"`

	// HealthChecks: The list of URLs to the HttpHealthCheck resource for
	// health checking this BackendService. Currently at most one health
	// check can be specified, and a health check is required.
	HealthChecks []string `json:"healthChecks,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of resource. Always compute#backendService
	// for backend services.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource. Provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035. Specifically, the name must be 1-63 characters long and
	// match the regular expression [a-z]([-a-z0-9]*[a-z0-9])? which means
	// the first character must be a lowercase letter, and all following
	// characters must be a dash, lowercase letter, or digit, except the
	// last character, which cannot be a dash.
	Name string `json:"name,omitempty"`

	// Port: Deprecated in favor of port name. The TCP port to connect on
	// the backend. The default value is 80.
	Port int64 `json:"port,omitempty"`

	// PortName: Name of backend port. The same name should appear in the
	// resource views referenced by this service. Required.
	PortName string `json:"portName,omitempty"`

	// Possible values:
	//   "HTTP"
	Protocol string `json:"protocol,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// TimeoutSec: How many seconds to wait for the backend before
	// considering it a failed request. Default is 30 seconds.
	TimeoutSec int64 `json:"timeoutSec,omitempty"`
}

type BackendServiceGroupHealth struct {
	HealthStatus []*HealthStatus `json:"healthStatus,omitempty"`

	// Kind: [Output Only] Type of resource. Always
	// compute#backendServiceGroupHealth for the health of backend services.
	Kind string `json:"kind,omitempty"`
}

// BackendServiceList: Contains a list of BackendService resources.
type BackendServiceList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: A list of BackendService resources.
	Items []*BackendService `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always
	// compute#backendServiceList for lists of backend services.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server-defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// DeprecationStatus: Deprecation status for a public resource.
type DeprecationStatus struct {
	// Deleted: An optional RFC3339 timestamp on or after which the
	// deprecation state of this resource will be changed to DELETED.
	Deleted string `json:"deleted,omitempty"`

	// Deprecated: An optional RFC3339 timestamp on or after which the
	// deprecation state of this resource will be changed to DEPRECATED.
	Deprecated string `json:"deprecated,omitempty"`

	// Obsolete: An optional RFC3339 timestamp on or after which the
	// deprecation state of this resource will be changed to OBSOLETE.
	Obsolete string `json:"obsolete,omitempty"`

	// Replacement: The URL of the suggested replacement for a deprecated
	// resource. The suggested replacement resource must be the same kind of
	// resource as the deprecated resource.
	Replacement string `json:"replacement,omitempty"`

	// State: The deprecation state of this resource. This can be
	// DEPRECATED, OBSOLETE, or DELETED. Operations which create a new
	// resource using a DEPRECATED resource will return successfully, but
	// with a warning indicating the deprecated resource and recommending
	// its replacement. Operations which use OBSOLETE or DELETED resources
	// will be rejected and result in an error.
	//
	// Possible values:
	//   "DELETED"
	//   "DEPRECATED"
	//   "OBSOLETE"
	State string `json:"state,omitempty"`
}

// Disk: A Disk resource.
type Disk struct {
	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of the resource. Always compute#disk for
	// disks.
	Kind string `json:"kind,omitempty"`

	// LastAttachTimestamp: [Output Only] Last attach timestamp in RFC3339
	// text format.
	LastAttachTimestamp string `json:"lastAttachTimestamp,omitempty"`

	// LastDetachTimestamp: [Output Only] Last detach timestamp in RFC3339
	// text format.
	LastDetachTimestamp string `json:"lastDetachTimestamp,omitempty"`

	// Licenses: Any applicable publicly visible licenses.
	Licenses []string `json:"licenses,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035. Specifically, the name must be 1-63 characters long and
	// match the regular expression [a-z]([-a-z0-9]*[a-z0-9])? which means
	// the first character must be a lowercase letter, and all following
	// characters must be a dash, lowercase letter, or digit, except the
	// last character, which cannot be a dash.
	Name string `json:"name,omitempty"`

	// Options: Internal use only.
	Options string `json:"options,omitempty"`

	// SelfLink: [Output Only] Server-defined fully-qualified URL for this
	// resource.
	SelfLink string `json:"selfLink,omitempty"`

	// SizeGb: Size of the persistent disk, specified in GB. You can specify
	// this field when creating a persistent disk using the sourceImage or
	// sourceSnapshot parameter, or specify it alone to create an empty
	// persistent disk.
	//
	// If you specify this field along with sourceImage or sourceSnapshot,
	// the value of sizeGb must not be less than the size of the sourceImage
	// or the size of the snapshot.
	SizeGb int64 `json:"sizeGb,omitempty,string"`

	// SourceImage: The source image used to create this disk. If the source
	// image is deleted from the system, this field will not be set, even if
	// an image with the same name has been re-created.
	//
	// When creating a disk, you can provide a private (custom) image using
	// the following input, and Compute Engine will use the corresponding
	// image from your project. For example:
	//
	// global/images/my-private-image
	//
	// Or you can provide an image from a publicly-available project. For
	// example, to use a Debian image from the debian-cloud project, make
	// sure to include the project in the
	// URL:
	//
	// projects/debian-cloud/global/images/debian-7-wheezy-vYYYYMMDD
	//
	// where vYYYYMMDD is the image version. The fully-qualified URL will
	// also work in both cases.
	SourceImage string `json:"sourceImage,omitempty"`

	// SourceImageId: The ID value of the image used to create this disk.
	// This value identifies the exact image that was used to create this
	// persistent disk. For example, if you created the persistent disk from
	// an image that was later deleted and recreated under the same name,
	// the source image ID would identify the exact version of the image
	// that was used.
	SourceImageId string `json:"sourceImageId,omitempty"`

	// SourceSnapshot: The source snapshot used to create this disk. You can
	// provide this as a partial or full URL to the resource. For example,
	// the following are valid values:
	// -
	// https://www.googleapis.com/compute/v1/projects/project/global/snapshots/snapshot
	// - projects/project/global/snapshots/snapshot
	// - global/snapshots/snapshot
	SourceSnapshot string `json:"sourceSnapshot,omitempty"`

	// SourceSnapshotId: [Output Only] The unique ID of the snapshot used to
	// create this disk. This value identifies the exact snapshot that was
	// used to create this persistent disk. For example, if you created the
	// persistent disk from a snapshot that was later deleted and recreated
	// under the same name, the source snapshot ID would identify the exact
	// version of the snapshot that was used.
	SourceSnapshotId string `json:"sourceSnapshotId,omitempty"`

	// Status: [Output Only] The status of disk creation. Applicable
	// statuses includes: CREATING, FAILED, READY, RESTORING.
	//
	// Possible values:
	//   "CREATING"
	//   "FAILED"
	//   "READY"
	//   "RESTORING"
	Status string `json:"status,omitempty"`

	// Type: URL of the disk type resource describing which disk type to use
	// to create the disk; provided by the client when the disk is created.
	Type string `json:"type,omitempty"`

	// Users: Links to the users of the disk (attached instances) in form:
	// project/zones/zone/instances/instance
	Users []string `json:"users,omitempty"`

	// Zone: [Output Only] URL of the zone where the disk resides.
	Zone string `json:"zone,omitempty"`
}

type DiskAggregatedList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A map of scoped disk lists.
	Items map[string]DisksScopedList `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always
	// compute#diskAggregatedList for aggregated lists of persistent disks.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// DiskList: A list of Disk resources.
type DiskList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of persistent disks.
	Items []*Disk `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#diskList for
	// lists of disks.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type DiskMoveRequest struct {
	// DestinationZone: The URL of the destination zone to move the disk to.
	// This can be a full or partial URL. For example, the following are all
	// valid URLs to a zone:
	// - https://www.googleapis.com/compute/v1/projects/project/zones/zone
	//
	// - projects/project/zones/zone
	// - zones/zone
	DestinationZone string `json:"destinationZone,omitempty"`

	// TargetDisk: The URL of the target disk to move. This can be a full or
	// partial URL. For example, the following are all valid URLs to a disk:
	//
	// -
	// https://www.googleapis.com/compute/v1/projects/project/zones/zone/disks/disk
	// - projects/project/zones/zone/disks/disk
	// - zones/zone/disks/disk
	TargetDisk string `json:"targetDisk,omitempty"`
}

// DiskType: A disk type resource.
type DiskType struct {
	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// DefaultDiskSizeGb: [Output Only] Server defined default disk size in
	// GB.
	DefaultDiskSizeGb int64 `json:"defaultDiskSizeGb,omitempty,string"`

	// Deprecated: [Output Only] The deprecation status associated with this
	// disk type.
	Deprecated *DeprecationStatus `json:"deprecated,omitempty"`

	// Description: [Output Only] An optional textual description of the
	// resource.
	Description string `json:"description,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of the resource. Always compute#diskType for
	// disk types.
	Kind string `json:"kind,omitempty"`

	// Name: [Output Only] Name of the resource.
	Name string `json:"name,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// ValidDiskSize: [Output Only] An optional textual description of the
	// valid disk size, such as "10GB-10TB".
	ValidDiskSize string `json:"validDiskSize,omitempty"`

	// Zone: [Output Only] URL of the zone where the disk type resides.
	Zone string `json:"zone,omitempty"`
}

type DiskTypeAggregatedList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A map of scoped disk type lists.
	Items map[string]DiskTypesScopedList `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always
	// compute#diskTypeAggregatedList.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// DiskTypeList: Contains a list of disk type resources.
type DiskTypeList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of Disk Type resources.
	Items []*DiskType `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#diskTypeList for
	// disk types.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type DiskTypesScopedList struct {
	// DiskTypes: [Output Only] List of disk types contained in this scope.
	DiskTypes []*DiskType `json:"diskTypes,omitempty"`

	// Warning: [Output Only] Informational warning which replaces the list
	// of disk types when the list is empty.
	Warning *DiskTypesScopedListWarning `json:"warning,omitempty"`
}

// DiskTypesScopedListWarning: [Output Only] Informational warning which
// replaces the list of disk types when the list is empty.
type DiskTypesScopedListWarning struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*DiskTypesScopedListWarningData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type DiskTypesScopedListWarningData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type DisksScopedList struct {
	// Disks: [Output Only] List of disks contained in this scope.
	Disks []*Disk `json:"disks,omitempty"`

	// Warning: [Output Only] Informational warning which replaces the list
	// of disks when the list is empty.
	Warning *DisksScopedListWarning `json:"warning,omitempty"`
}

// DisksScopedListWarning: [Output Only] Informational warning which
// replaces the list of disks when the list is empty.
type DisksScopedListWarning struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*DisksScopedListWarningData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type DisksScopedListWarningData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

// Firewall: A Firewall resource.
type Firewall struct {
	// Allowed: The list of rules specified by this firewall. Each rule
	// specifies a protocol and port-range tuple that describes a permitted
	// connection.
	Allowed []*FirewallAllowed `json:"allowed,omitempty"`

	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Ony] Type of the resource. Always compute#firewall for
	// firewall rules.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035. Specifically, the name must be 1-63 characters long and
	// match the regular expression [a-z]([-a-z0-9]*[a-z0-9])? which means
	// the first character must be a lowercase letter, and all following
	// characters must be a dash, lowercase letter, or digit, except the
	// last character, which cannot be a dash.
	Name string `json:"name,omitempty"`

	// Network: URL of the network resource for this firewall rule. This
	// field is required for creating an instance but optional when creating
	// a firewall rule. If not specified when creating a firewall rule, the
	// default network is used:
	// global/networks/default
	// If you choose to specify this property, you can specify the network
	// as a full or partial URL. For example, the following are all valid
	// URLs:
	// -
	// https://www.googleapis.com/compute/v1/projects/myproject/global/networks/my-network
	// - projects/myproject/global/networks/my-network
	// - global/networks/default
	Network string `json:"network,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// SourceRanges: The IP address blocks that this rule applies to,
	// expressed in CIDR format. One or both of sourceRanges and sourceTags
	// may be set.
	//
	// If both properties are set, an inbound connection is allowed if the
	// range matches the sourceRanges OR the tag of the source matches the
	// sourceTags property. The connection does not need to match both
	// properties.
	SourceRanges []string `json:"sourceRanges,omitempty"`

	// SourceTags: A list of instance tags which this rule applies to. One
	// or both of sourceRanges and sourceTags may be set.
	//
	// If both properties are set, an inbound connection is allowed if the
	// range matches the sourceRanges OR the tag of the source matches the
	// sourceTags property. The connection does not need to match both
	// properties.
	SourceTags []string `json:"sourceTags,omitempty"`

	// TargetTags: A list of instance tags indicating sets of instances
	// located in the network that may make network connections as specified
	// in allowed[]. If no targetTags are specified, the firewall rule
	// applies to all instances on the specified network.
	TargetTags []string `json:"targetTags,omitempty"`
}

type FirewallAllowed struct {
	// IPProtocol: The IP protocol that is allowed for this rule. The
	// protocol type is required when creating a firewall. This value can
	// either be one of the following well known protocol strings (tcp, udp,
	// icmp, esp, ah, sctp), or the IP protocol number.
	IPProtocol string `json:"IPProtocol,omitempty"`

	// Ports: An optional list of ports which are allowed. This field is
	// only applicable for UDP or TCP protocol. Each entry must be either an
	// integer or a range. If not specified, connections through any port
	// are allowed
	//
	// Example inputs include: ["22"], ["80","443"], and ["12345-12349"].
	Ports []string `json:"ports,omitempty"`
}

// FirewallList: Contains a list of Firewall resources.
type FirewallList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of Firewall resources.
	Items []*Firewall `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#firewallList for
	// lists of firewalls.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// ForwardingRule: A ForwardingRule resource. A ForwardingRule resource
// specifies which pool of target VMs to forward a packet to if it
// matches the given [IPAddress, IPProtocol, portRange] tuple.
type ForwardingRule struct {
	// IPAddress: Value of the reserved IP address that this forwarding rule
	// is serving on behalf of. For global forwarding rules, the address
	// must be a global IP; for regional forwarding rules, the address must
	// live in the same region as the forwarding rule. If left empty
	// (default value), an ephemeral IP from the same scope (global or
	// regional) will be assigned.
	IPAddress string `json:"IPAddress,omitempty"`

	// IPProtocol: The IP protocol to which this rule applies, valid options
	// are TCP, UDP, ESP, AH or SCTP.
	//
	// Possible values:
	//   "AH"
	//   "ESP"
	//   "SCTP"
	//   "TCP"
	//   "UDP"
	IPProtocol string `json:"IPProtocol,omitempty"`

	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035. Specifically, the name must be 1-63 characters long and
	// match the regular expression [a-z]([-a-z0-9]*[a-z0-9])? which means
	// the first character must be a lowercase letter, and all following
	// characters must be a dash, lowercase letter, or digit, except the
	// last character, which cannot be a dash.
	Name string `json:"name,omitempty"`

	// PortRange: Applicable only when `IPProtocol` is TCP, UDP, or SCTP,
	// only packets addressed to ports in the specified range will be
	// forwarded to target. If portRange is left empty (default value), all
	// ports are forwarded. Forwarding rules with the same `[IPAddress,
	// IPProtocol]` pair must have disjoint port ranges.
	PortRange string `json:"portRange,omitempty"`

	// Region: [Output Only] URL of the region where the regional forwarding
	// rule resides. This field is not applicable to global forwarding
	// rules.
	Region string `json:"region,omitempty"`

	// SelfLink: [Output Only] Server-defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Target: The URL of the target resource to receive the matched
	// traffic. For regional forwarding rules, this target must live in the
	// same region as the forwarding rule. For global forwarding rules, this
	// target must be a global TargetHttpProxy or TargetHttpsProxy resource.
	Target string `json:"target,omitempty"`
}

type ForwardingRuleAggregatedList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: A map of scoped forwarding rule lists.
	Items map[string]ForwardingRulesScopedList `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// ForwardingRuleList: Contains a list of ForwardingRule resources.
type ForwardingRuleList struct {
	// Id: [Output Only] Unique identifier for the resource. Set by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: A list of ForwardingRule resources.
	Items []*ForwardingRule `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type ForwardingRulesScopedList struct {
	// ForwardingRules: List of forwarding rules contained in this scope.
	ForwardingRules []*ForwardingRule `json:"forwardingRules,omitempty"`

	// Warning: Informational warning which replaces the list of forwarding
	// rules when the list is empty.
	Warning *ForwardingRulesScopedListWarning `json:"warning,omitempty"`
}

// ForwardingRulesScopedListWarning: Informational warning which
// replaces the list of forwarding rules when the list is empty.
type ForwardingRulesScopedListWarning struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*ForwardingRulesScopedListWarningData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type ForwardingRulesScopedListWarningData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type HealthCheckReference struct {
	HealthCheck string `json:"healthCheck,omitempty"`
}

type HealthStatus struct {
	// HealthState: Health state of the instance.
	//
	// Possible values:
	//   "HEALTHY"
	//   "UNHEALTHY"
	HealthState string `json:"healthState,omitempty"`

	// Instance: URL of the instance resource.
	Instance string `json:"instance,omitempty"`

	// IpAddress: The IP address represented by this resource.
	IpAddress string `json:"ipAddress,omitempty"`

	// Port: The port on the instance.
	Port int64 `json:"port,omitempty"`
}

// HostRule: UrlMaps A host-matching rule for a URL. If matched, will
// use the named PathMatcher to select the BackendService.
type HostRule struct {
	// Description: An optional textual description.
	Description string `json:"description,omitempty"`

	// Hosts: The list of host patterns to match. They must be valid
	// hostnames except that they may start with *. or *-. The * acts like a
	// glob and will match any string of atoms (separated by .s and -s) to
	// the left.
	Hosts []string `json:"hosts,omitempty"`

	// PathMatcher: The name of the PathMatcher to match the path portion of
	// the URL, if the this hostRule matches the URL's host portion.
	PathMatcher string `json:"pathMatcher,omitempty"`
}

// HttpHealthCheck: An HttpHealthCheck resource. This resource defines a
// template for how individual VMs should be checked for health, via
// HTTP.
type HttpHealthCheck struct {
	// CheckIntervalSec: How often (in seconds) to send a health check. The
	// default value is 5 seconds.
	CheckIntervalSec int64 `json:"checkIntervalSec,omitempty"`

	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// HealthyThreshold: A so-far unhealthy VM will be marked healthy after
	// this many consecutive successes. The default value is 2.
	HealthyThreshold int64 `json:"healthyThreshold,omitempty"`

	// Host: The value of the host header in the HTTP health check request.
	// If left empty (default value), the public IP on behalf of which this
	// health check is performed will be used.
	Host string `json:"host,omitempty"`

	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id uint64 `json:"id,omitempty,string"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035.
	Name string `json:"name,omitempty"`

	// Port: The TCP port number for the HTTP health check request. The
	// default value is 80.
	Port int64 `json:"port,omitempty"`

	// RequestPath: The request path of the HTTP health check request. The
	// default value is "/".
	RequestPath string `json:"requestPath,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// TimeoutSec: How long (in seconds) to wait before claiming failure.
	// The default value is 5 seconds. It is invalid for timeoutSec to have
	// greater value than checkIntervalSec.
	TimeoutSec int64 `json:"timeoutSec,omitempty"`

	// UnhealthyThreshold: A so-far healthy VM will be marked unhealthy
	// after this many consecutive failures. The default value is 2.
	UnhealthyThreshold int64 `json:"unhealthyThreshold,omitempty"`
}

// HttpHealthCheckList: Contains a list of HttpHealthCheck resources.
type HttpHealthCheckList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A list of HttpHealthCheck resources.
	Items []*HttpHealthCheck `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

// Image: An Image resource.
type Image struct {
	// ArchiveSizeBytes: Size of the image tar.gz archive stored in Google
	// Cloud Storage (in bytes).
	ArchiveSizeBytes int64 `json:"archiveSizeBytes,omitempty,string"`

	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Deprecated: The deprecation status associated with this image.
	Deprecated *DeprecationStatus `json:"deprecated,omitempty"`

	// Description: Textual description of the resource; provided by the
	// client when the resource is created.
	Description string `json:"description,omitempty"`

	// DiskSizeGb: Size of the image when restored onto a persistent disk
	// (in GB).
	DiskSizeGb int64 `json:"diskSizeGb,omitempty,string"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of the resource. Always compute#image for
	// images.
	Kind string `json:"kind,omitempty"`

	// Licenses: Any applicable publicly visible licenses.
	Licenses []string `json:"licenses,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035. Specifically, the name must be 1-63 characters long and
	// match the regular expression [a-z]([-a-z0-9]*[a-z0-9])? which means
	// the first character must be a lowercase letter, and all following
	// characters must be a dash, lowercase letter, or digit, except the
	// last character, which cannot be a dash.
	Name string `json:"name,omitempty"`

	// RawDisk: The parameters of the raw disk image.
	RawDisk *ImageRawDisk `json:"rawDisk,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// SourceDisk: URL of the The source disk used to create this image.
	// This can be a full or valid partial URL. You must provide either this
	// property or the rawDisk.source property but not both to create an
	// image. For example, the following are valid values:
	// -
	// https://www.googleapis.com/compute/v1/projects/project/zones/zone/disk/disk
	// - projects/project/zones/zone/disk/disk
	// - zones/zone/disks/disk
	SourceDisk string `json:"sourceDisk,omitempty"`

	// SourceDiskId: The ID value of the disk used to create this image.
	// This value may be used to determine whether the image was taken from
	// the current or a previous instance of a given disk name.
	SourceDiskId string `json:"sourceDiskId,omitempty"`

	// SourceType: The type of the image used to create this disk. The
	// default and only value is RAW
	//
	// Possible values:
	//   "RAW" (default)
	SourceType string `json:"sourceType,omitempty"`

	// Status: [Output Only] The status of the image. An image can be used
	// to create other resources, such as instances, only after the image
	// has been successfully created and the status is set to READY.
	// Possible values are FAILED, PENDING, or READY.
	//
	// Possible values:
	//   "FAILED"
	//   "PENDING"
	//   "READY"
	Status string `json:"status,omitempty"`
}

// ImageRawDisk: The parameters of the raw disk image.
type ImageRawDisk struct {
	// ContainerType: The format used to encode and transmit the block
	// device, which should be TAR. This is just a container and
	// transmission format and not a runtime format. Provided by the client
	// when the disk image is created.
	//
	// Possible values:
	//   "TAR"
	ContainerType string `json:"containerType,omitempty"`

	// Sha1Checksum: An optional SHA1 checksum of the disk image before
	// unpackaging; provided by the client when the disk image is created.
	Sha1Checksum string `json:"sha1Checksum,omitempty"`

	// Source: The full Google Cloud Storage URL where the disk image is
	// stored. You must provide either this property or the sourceDisk
	// property but not both.
	Source string `json:"source,omitempty"`
}

// ImageList: Contains a list of Image resources.
type ImageList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A list of Image resources.
	Items []*Image `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

// Instance: An Instance resource.
type Instance struct {
	// CanIpForward: Allows this instance to send and receive packets with
	// non-matching destination or source IPs. This is required if you plan
	// to use this instance to forward routes. For more information, see
	// Enabling IP Forwarding.
	CanIpForward bool `json:"canIpForward,omitempty"`

	// CpuPlatform: [Output Only] The CPU platform used by this instance.
	CpuPlatform string `json:"cpuPlatform,omitempty"`

	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// Disks: Array of disks associated with this instance. Persistent disks
	// must be created before you can assign them.
	Disks []*AttachedDisk `json:"disks,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of the resource. Always compute#instance for
	// instances.
	Kind string `json:"kind,omitempty"`

	// MachineType: Full or partial URL of the machine type resource to use
	// for this instance. This is provided by the client when the instance
	// is created. For example, the following is a valid partial
	// url:
	//
	// zones/zone/machineTypes/machine-type
	MachineType string `json:"machineType,omitempty"`

	// Metadata: The metadata key/value pairs assigned to this instance.
	// This includes custom metadata and predefined keys.
	Metadata *Metadata `json:"metadata,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035. Specifically, the name must be 1-63 characters long and
	// match the regular expression [a-z]([-a-z0-9]*[a-z0-9])? which means
	// the first character must be a lowercase letter, and all following
	// characters must be a dash, lowercase letter, or digit, except the
	// last character, which cannot be a dash.
	Name string `json:"name,omitempty"`

	// NetworkInterfaces: An array of configurations for this interface.
	// This specifies how this interface is configured to interact with
	// other network services, such as connecting to the internet.
	NetworkInterfaces []*NetworkInterface `json:"networkInterfaces,omitempty"`

	// Scheduling: Scheduling options for this instance.
	Scheduling *Scheduling `json:"scheduling,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// ServiceAccounts: A list of service accounts, with their specified
	// scopes, authorized for this instance. Service accounts generate
	// access tokens that can be accessed through the metadata server and
	// used to authenticate applications on the instance. See Authenticating
	// from Google Compute Engine for more information.
	ServiceAccounts []*ServiceAccount `json:"serviceAccounts,omitempty"`

	// Status: [Output Only] The status of the instance. One of the
	// following values: PROVISIONING, STAGING, RUNNING, STOPPING, and
	// TERMINATED.
	//
	// Possible values:
	//   "PROVISIONING"
	//   "RUNNING"
	//   "STAGING"
	//   "STOPPED"
	//   "STOPPING"
	//   "SUSPENDED"
	//   "SUSPENDING"
	//   "TERMINATED"
	Status string `json:"status,omitempty"`

	// StatusMessage: [Output Only] An optional, human-readable explanation
	// of the status.
	StatusMessage string `json:"statusMessage,omitempty"`

	// Tags: A list of tags to appy to this instance. Tags are used to
	// identify valid sources or targets for network firewalls and are
	// specified by the client during instance creation. The tags can be
	// later modified by the setTags method. Each tag within the list must
	// comply with RFC1035.
	Tags *Tags `json:"tags,omitempty"`

	// Zone: [Output Only] URL of the zone where the instance resides.
	Zone string `json:"zone,omitempty"`
}

type InstanceAggregatedList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A map of scoped instance lists.
	Items map[string]InstancesScopedList `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always
	// compute#instanceAggregatedList for aggregated lists of Instance
	// resources.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type InstanceGroup struct {
	// CreationTimestamp: [Output Only] The creation timestamp for this
	// instance group in RFC3339 text format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional text description for the instance group.
	Description string `json:"description,omitempty"`

	// Fingerprint: [Output Only] The fingerprint of the named ports
	// information. The system uses this fingerprint to detect conflicts
	// when multiple users change the named ports information concurrently.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Id: [Output Only] A unique identifier for this instance group. The
	// server defines this identifier.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] The resource type, which is always
	// compute#instanceGroup for instance groups.
	Kind string `json:"kind,omitempty"`

	// Name: The name of the instance group. The name must be 1-63
	// characters long, and comply with RFC1035.
	Name string `json:"name,omitempty"`

	// NamedPorts: Assigns a name to a port number. For example: {name:
	// ?http?, port: 80} This allows the system to reference ports by the
	// assigned name instead of a port number. Named ports can also contain
	// multiple ports. For example: [{name: ?http?, port: 80},{name: "http",
	// port: 8080}] Named ports apply to all instances in this instance
	// group.
	NamedPorts []*NamedPort `json:"namedPorts,omitempty"`

	// Network: The URL of the network to which all instances in the
	// instance group belong.
	Network string `json:"network,omitempty"`

	// SelfLink: [Output Only] The URL for this instance group. The server
	// defines this URL.
	SelfLink string `json:"selfLink,omitempty"`

	// Size: [Output Only] The total number of instances in the instance
	// group.
	Size int64 `json:"size,omitempty"`

	// Zone: The URL of the zone where the instance group is located.
	Zone string `json:"zone,omitempty"`
}

type InstanceGroupAggregatedList struct {
	// Id: [Output Only] A unique identifier for this aggregated list of
	// instance groups. The server defines this identifier.
	Id string `json:"id,omitempty"`

	// Items: A map of scoped instance group lists.
	Items map[string]InstanceGroupsScopedList `json:"items,omitempty"`

	// Kind: [Output Only] The resource type, which is always
	// compute#instanceGroupAggregatedList for aggregated lists of instance
	// groups.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token that is used to continue a
	// truncated list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] A unique identifier for this aggregated list
	// of instance groups. The server defines this identifier.
	SelfLink string `json:"selfLink,omitempty"`
}

// InstanceGroupList: A list of InstanceGroup resources.
type InstanceGroupList struct {
	// Id: [Output Only] A unique identifier for this list of instance
	// groups. The server defines this identifier.
	Id string `json:"id,omitempty"`

	// Items: A list of InstanceGroup resources.
	Items []*InstanceGroup `json:"items,omitempty"`

	// Kind: [Output Only] The resource type, which is always
	// compute#instanceGroupList for instance group lists.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token that is used to continue a
	// truncated list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] The URL for this instance group. The server
	// defines this URL.
	SelfLink string `json:"selfLink,omitempty"`
}

// InstanceGroupManager: InstanceGroupManagers
//
// Next available tag: 17
type InstanceGroupManager struct {
	// AutoHealingPolicies: The autohealing policy for this managed instance
	// group. You can specify only one value.
	AutoHealingPolicies []*InstanceGroupManagerAutoHealingPolicy `json:"autoHealingPolicies,omitempty"`

	// BaseInstanceName: The base instance name to use for instances in this
	// group. The value must be 1-58 characters long. Instances are named by
	// appending a hyphen and a random four-character string to the base
	// instance name. The base instance name must comply with RFC1035.
	BaseInstanceName string `json:"baseInstanceName,omitempty"`

	// CreationTimestamp: [Output Only] The creation timestamp for this
	// managed instance group in RFC3339 text format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// CurrentActions: [Output Only] The list of instance actions and the
	// number of instances in this managed instance group that are scheduled
	// for those actions.
	CurrentActions *InstanceGroupManagerActionsSummary `json:"currentActions,omitempty"`

	// Description: An optional text description for the managed instance
	// group.
	Description string `json:"description,omitempty"`

	// Fingerprint: [Output Only] The fingerprint of the target pools
	// information, which is a hash of the contents. This field is used for
	// optimistic locking when updating the target pool entries.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Id: [Output Only] A unique identifier for this managed instance
	// group. The server defines this identifier.
	Id uint64 `json:"id,omitempty,string"`

	// InstanceGroup: [Output Only] The URL of the InstanceGroup resource.
	InstanceGroup string `json:"instanceGroup,omitempty"`

	// InstanceTemplate: The URL of the instance template that is specified
	// for this managed instance group. The group uses this template to
	// create all new instances in the managed instance group.
	InstanceTemplate string `json:"instanceTemplate,omitempty"`

	// Kind: [Output Only] The resource type, which is always
	// compute#instanceGroupManager for managed instance groups.
	Kind string `json:"kind,omitempty"`

	// Name: The name of the managed instance group. The name must be 1-63
	// characters long, and comply with RFC1035.
	Name string `json:"name,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this managed instance
	// group.
	SelfLink string `json:"selfLink,omitempty"`

	// TargetPools: The URL of all TargetPool resources to which new
	// instances in the instanceGroup field are added. Updating the target
	// pool values does not affect existing instances.
	TargetPools []string `json:"targetPools,omitempty"`

	// TargetSize: The target number of running instances for this managed
	// instance group. Deleting or abandoning instances reduces this number.
	// Resizing the group changes this number.
	TargetSize int64 `json:"targetSize,omitempty"`

	// Zone: The URL of the zone where the managed instance group is
	// located.
	Zone string `json:"zone,omitempty"`
}

type InstanceGroupManagerActionsSummary struct {
	Abandoning int64 `json:"abandoning,omitempty"`

	Creating int64 `json:"creating,omitempty"`

	Deleting int64 `json:"deleting,omitempty"`

	None int64 `json:"none,omitempty"`

	Recreating int64 `json:"recreating,omitempty"`

	Refreshing int64 `json:"refreshing,omitempty"`

	Restarting int64 `json:"restarting,omitempty"`
}

type InstanceGroupManagerAggregatedList struct {
	// Id: [Output Only] A unique identifier for this aggregated list of
	// managed instance groups. The server defines this identifier.
	Id string `json:"id,omitempty"`

	// Items: A map of filtered managed instance group lists.
	Items map[string]InstanceGroupManagersScopedList `json:"items,omitempty"`

	// Kind: [Output Only] Type of the resource. Always
	// compute#instanceGroupManagerAggregatedList for an aggregated list of
	// managed instance groups.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token that is used to continue a
	// truncated list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] The URL for this aggregated list of managed
	// instance groups. The server defines this URL.
	SelfLink string `json:"selfLink,omitempty"`
}

type InstanceGroupManagerAutoHealingPolicy struct {
	// ActionType: The action to perform when an instance becomes unhealthy.
	// Possible values are RECREATE or RESTART. RECREATE replaces an
	// unhealthy instance with a new instance that is based on the instance
	// template for this managed instance group. RESTART performs a soft
	// restart on an instance. If the instance cannot restart softly, the
	// instance performs a hard restart.
	//
	// Possible values:
	//   "RECREATE"
	//   "RESTART"
	ActionType string `json:"actionType,omitempty"`

	// HealthCheck: The URL for the HealthCheck that signals autohealing.
	HealthCheck string `json:"healthCheck,omitempty"`
}

// InstanceGroupManagerList: [Output Only] A list of
// InstanceGroupManager resources.
type InstanceGroupManagerList struct {
	// Id: [Output Only] A unique identifier for this managed instance
	// group. The server defines this identifier.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of managed instance group resources.
	Items []*InstanceGroupManager `json:"items,omitempty"`

	// Kind: [Output Only] Type of the resource. Always
	// compute#instanceGroupManagerList for a list of managed instance group
	// resources.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token that is used to continue a
	// truncated list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] The URL for this managed instance group. The
	// server defines this URL.
	SelfLink string `json:"selfLink,omitempty"`
}

type InstanceGroupManagersAbandonInstancesRequest struct {
	// Instances: The names of instances to abandon from the managed
	// instance group.
	Instances []string `json:"instances,omitempty"`
}

type InstanceGroupManagersDeleteInstancesRequest struct {
	// Instances: The names of one or more instances to delete.
	Instances []string `json:"instances,omitempty"`
}

type InstanceGroupManagersListManagedInstancesResponse struct {
	// ManagedInstances: List of managed instances. If empty - all instances
	// are listed.
	ManagedInstances []*ManagedInstance `json:"managedInstances,omitempty"`
}

type InstanceGroupManagersRecreateInstancesRequest struct {
	// Instances: The names of one or more instances to recreate.
	Instances []string `json:"instances,omitempty"`
}

type InstanceGroupManagersScopedList struct {
	// InstanceGroupManagers: [Output Only] The list of managed instance
	// groups that are contained in the specified project and zone.
	InstanceGroupManagers []*InstanceGroupManager `json:"instanceGroupManagers,omitempty"`

	// Warning: [Output Only] The warning that replaces the list of managed
	// instance groups when the list is empty.
	Warning *InstanceGroupManagersScopedListWarning `json:"warning,omitempty"`
}

// InstanceGroupManagersScopedListWarning: [Output Only] The warning
// that replaces the list of managed instance groups when the list is
// empty.
type InstanceGroupManagersScopedListWarning struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*InstanceGroupManagersScopedListWarningData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type InstanceGroupManagersScopedListWarningData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type InstanceGroupManagersSetInstanceTemplateRequest struct {
	// InstanceTemplate: The URL of the instance template that is specified
	// for this managed instance group. The group uses this template to
	// create all new instances in the managed instance group.
	InstanceTemplate string `json:"instanceTemplate,omitempty"`
}

type InstanceGroupManagersSetTargetPoolsRequest struct {
	// Fingerprint: The fingerprint of the target pools information, which
	// is a hash of the contents. This field is used for optimistic locking
	// when updating the target pool entries.
	Fingerprint string `json:"fingerprint,omitempty"`

	// TargetPools: The list of target pool URLs that instances in this
	// managed instance group belong to. When the managed instance group
	// creates new instances, the group automatically adds those instances
	// to the target pools that are specified in this parameter. Changing
	// the value of this parameter does not change the target pools of
	// existing instances in this managed instance group.
	TargetPools []string `json:"targetPools,omitempty"`
}

type InstanceGroupsAddInstancesRequest struct {
	// Instances: The instances to add to the instance group.
	Instances []*InstanceReference `json:"instances,omitempty"`
}

type InstanceGroupsListInstances struct {
	// Id: [Output Only] A unique identifier for this list of instance
	// groups. The server defines this identifier.
	Id string `json:"id,omitempty"`

	// Items: A list of InstanceWithNamedPorts resources, which contains all
	// named ports for the given instance.
	Items []*InstanceWithNamedPorts `json:"items,omitempty"`

	// Kind: [Output Only] The resource type, which is always
	// compute#instanceGroupsListInstances for lists of instance groups.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token that is used to continue a
	// truncated list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] The URL for this list of instance groups. The
	// server defines this URL.
	SelfLink string `json:"selfLink,omitempty"`
}

type InstanceGroupsListInstancesRequest struct {
	// InstanceState: A filter for the state of the instances in the
	// instance group. Valid options are ALL or RUNNING. If you do not
	// specify this parameter the list includes all instances regardless of
	// their state.
	//
	// Possible values:
	//   "ALL"
	//   "RUNNING"
	InstanceState string `json:"instanceState,omitempty"`

	// PortName: A filter for the named ports that are associated with
	// instances in the instance group. If you specify this parameter, the
	// generated list includes only instances that are associated with the
	// specified named ports. If you do not specify this parameter, the
	// generated list includes all instances regardless of their named
	// ports.
	PortName string `json:"portName,omitempty"`
}

type InstanceGroupsRemoveInstancesRequest struct {
	// Instances: The instances to remove from the instance group.
	Instances []*InstanceReference `json:"instances,omitempty"`
}

type InstanceGroupsScopedList struct {
	// InstanceGroups: [Output Only] The list of instance groups that are
	// contained in this scope.
	InstanceGroups []*InstanceGroup `json:"instanceGroups,omitempty"`

	// Warning: [Output Only] An informational warning that replaces the
	// list of instance groups when the list is empty.
	Warning *InstanceGroupsScopedListWarning `json:"warning,omitempty"`
}

// InstanceGroupsScopedListWarning: [Output Only] An informational
// warning that replaces the list of instance groups when the list is
// empty.
type InstanceGroupsScopedListWarning struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*InstanceGroupsScopedListWarningData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type InstanceGroupsScopedListWarningData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type InstanceGroupsSetNamedPortsRequest struct {
	// Fingerprint: The fingerprint of the named ports information, which is
	// a hash of the contents. Use this field for optimistic locking when
	// you update the named ports entries.
	Fingerprint string `json:"fingerprint,omitempty"`

	// NamedPorts: The list of named ports to set for this instance group.
	NamedPorts []*NamedPort `json:"namedPorts,omitempty"`
}

// InstanceList: Contains a list of instance resources.
type InstanceList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of Instance resources.
	Items []*Instance `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#instanceList for
	// lists of Instance resources.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type InstanceMoveRequest struct {
	// DestinationZone: The URL of the destination zone to move the instance
	// to. This can be a full or partial URL. For example, the following are
	// all valid URLs to a zone:
	// - https://www.googleapis.com/compute/v1/projects/project/zones/zone
	//
	// - projects/project/zones/zone
	// - zones/zone
	DestinationZone string `json:"destinationZone,omitempty"`

	// TargetInstance: The URL of the target instance to move. This can be a
	// full or partial URL. For example, the following are all valid URLs to
	// an instance:
	// -
	// https://www.googleapis.com/compute/v1/projects/project/zones/zone/instances/instance
	// - projects/project/zones/zone/instances/instance
	// - zones/zone/instances/instance
	TargetInstance string `json:"targetInstance,omitempty"`
}

type InstanceProperties struct {
	// CanIpForward: A boolean that specifies if instances created from this
	// template can send packets with source IP addresses other than their
	// own or receive packets with destination IP addresses other than their
	// own. If you use these instances as an IP gateway or as the next-hop
	// in a Route resource, specify true. Otherwise, specify false.
	CanIpForward bool `json:"canIpForward,omitempty"`

	// Description: An optional text description for the instances that are
	// created from this instance template.
	Description string `json:"description,omitempty"`

	// Disks: An array of disks that are associated with the instances that
	// are created from this template.
	Disks []*AttachedDisk `json:"disks,omitempty"`

	// MachineType: The machine type to use for instances that are created
	// from this template.
	MachineType string `json:"machineType,omitempty"`

	// Metadata: The metadata key/value pairs to assign to instances that
	// are created from this template. These pairs can consist of custom
	// metadata or predefined keys. See Project and instance metadata for
	// more information.
	Metadata *Metadata `json:"metadata,omitempty"`

	// NetworkInterfaces: An array of network access configurations for this
	// interface. This specifies how this interface is configured to
	// interact with other network services, such as connecting to the
	// internet. Currently, ONE_TO_ONE_NAT is the only supported access
	// configuration. If you do not specify any access configurations, the
	// instances that are created from this template will have no external
	// internet access.
	NetworkInterfaces []*NetworkInterface `json:"networkInterfaces,omitempty"`

	// Scheduling: A list of scheduling options for the instances that are
	// created from this template.
	Scheduling *Scheduling `json:"scheduling,omitempty"`

	// ServiceAccounts: A list of service accounts with specified scopes.
	// Access tokens for these service accounts are available to the
	// instances that are created from this template. Use metadata queries
	// to obtain the access tokens for these instances.
	ServiceAccounts []*ServiceAccount `json:"serviceAccounts,omitempty"`

	// Tags: A list of tags to apply to the instances that are created from
	// this template. The tags identify valid sources or targets for network
	// firewalls. The setTags method can modify this list of tags. Each tag
	// within the list must comply with RFC1035.
	Tags *Tags `json:"tags,omitempty"`
}

type InstanceReference struct {
	Instance string `json:"instance,omitempty"`
}

// InstanceTemplate: An Instance Template resource.
type InstanceTemplate struct {
	// CreationTimestamp: [Output Only] The creation timestamp for this
	// instance template in RFC3339 text format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional text description for the instance template.
	Description string `json:"description,omitempty"`

	// Id: [Output Only] A unique identifier for this instance template. The
	// server defines this identifier.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] The resource type, which is always
	// compute#instanceTemplate for instance templates.
	Kind string `json:"kind,omitempty"`

	// Name: The name of the instance template. The name must be 1-63
	// characters long, and comply with RFC1035.
	Name string `json:"name,omitempty"`

	// Properties: The instance properties for the instance template
	// resource.
	Properties *InstanceProperties `json:"properties,omitempty"`

	// SelfLink: [Output Only] The URL for this instance template. The
	// server defines this URL.
	SelfLink string `json:"selfLink,omitempty"`
}

// InstanceTemplateList: A list of instance templates.
type InstanceTemplateList struct {
	// Id: [Output Only] A unique identifier for this instance template. The
	// server defines this identifier.
	Id string `json:"id,omitempty"`

	// Items: A list of InstanceTemplate resources.
	Items []*InstanceTemplate `json:"items,omitempty"`

	// Kind: [Output Only] The resource type, which is always
	// compute#instanceTemplatesListResponse for instance template lists.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token that is used to continue a
	// truncated list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] The URL for this instance template list. The
	// server defines this URL.
	SelfLink string `json:"selfLink,omitempty"`
}

type InstanceWithNamedPorts struct {
	// Instance: The URL of the instance.
	Instance string `json:"instance,omitempty"`

	// NamedPorts: The named ports that belong to this instance group.
	NamedPorts []*NamedPort `json:"namedPorts,omitempty"`

	// Status: The status of the instance.
	//
	// Possible values:
	//   "PROVISIONING"
	//   "RUNNING"
	//   "STAGING"
	//   "STOPPED"
	//   "STOPPING"
	//   "SUSPENDED"
	//   "SUSPENDING"
	//   "TERMINATED"
	Status string `json:"status,omitempty"`
}

type InstancesScopedList struct {
	// Instances: [Output Only] List of instances contained in this scope.
	Instances []*Instance `json:"instances,omitempty"`

	// Warning: [Output Only] Informational warning which replaces the list
	// of instances when the list is empty.
	Warning *InstancesScopedListWarning `json:"warning,omitempty"`
}

// InstancesScopedListWarning: [Output Only] Informational warning which
// replaces the list of instances when the list is empty.
type InstancesScopedListWarning struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*InstancesScopedListWarningData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type InstancesScopedListWarningData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

// License: A license resource.
type License struct {
	// ChargesUseFee: If true, the customer will be charged license fee for
	// running software that contains this license on an instance.
	ChargesUseFee bool `json:"chargesUseFee,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#license for
	// licenses.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource. The name must be 1-63 characters long,
	// and comply with RCF1035.
	Name string `json:"name,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// MachineType: A Machine Type resource.
type MachineType struct {
	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Deprecated: [Output Only] The deprecation status associated with this
	// machine type.
	Deprecated *DeprecationStatus `json:"deprecated,omitempty"`

	// Description: [Output Only] An optional textual description of the
	// resource.
	Description string `json:"description,omitempty"`

	// GuestCpus: [Output Only] The tumber of CPUs exposed to the instance.
	GuestCpus int64 `json:"guestCpus,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// ImageSpaceGb: [Deprecated] This property is deprecated and will never
	// be populated with any relevant values.
	ImageSpaceGb int64 `json:"imageSpaceGb,omitempty"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// MaximumPersistentDisks: [Output Only] Maximum persistent disks
	// allowed.
	MaximumPersistentDisks int64 `json:"maximumPersistentDisks,omitempty"`

	// MaximumPersistentDisksSizeGb: [Output Only] Maximum total persistent
	// disks size (GB) allowed.
	MaximumPersistentDisksSizeGb int64 `json:"maximumPersistentDisksSizeGb,omitempty,string"`

	// MemoryMb: [Output Only] The amount of physical memory available to
	// the instance, defined in MB.
	MemoryMb int64 `json:"memoryMb,omitempty"`

	// Name: [Output Only] Name of the resource.
	Name string `json:"name,omitempty"`

	// ScratchDisks: [Output Only] List of extended scratch disks assigned
	// to the instance.
	ScratchDisks []*MachineTypeScratchDisks `json:"scratchDisks,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Zone: [Output Only] The name of the zone where the machine type
	// resides, such as us-central1-a.
	Zone string `json:"zone,omitempty"`
}

type MachineTypeScratchDisks struct {
	// DiskGb: Size of the scratch disk, defined in GB.
	DiskGb int64 `json:"diskGb,omitempty"`
}

type MachineTypeAggregatedList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A map of scoped machine type lists.
	Items map[string]MachineTypesScopedList `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always
	// compute#machineTypeAggregatedList for aggregated lists of machine
	// types.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// MachineTypeList: Contains a list of Machine Type resources.
type MachineTypeList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of Machine Type resources.
	Items []*MachineType `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#machineTypeList
	// for lists of machine types.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type MachineTypesScopedList struct {
	// MachineTypes: [Output Only] List of machine types contained in this
	// scope.
	MachineTypes []*MachineType `json:"machineTypes,omitempty"`

	// Warning: [Output Only] An informational warning that appears when the
	// machine types list is empty.
	Warning *MachineTypesScopedListWarning `json:"warning,omitempty"`
}

// MachineTypesScopedListWarning: [Output Only] An informational warning
// that appears when the machine types list is empty.
type MachineTypesScopedListWarning struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*MachineTypesScopedListWarningData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type MachineTypesScopedListWarningData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type ManagedInstance struct {
	// CurrentAction: The current action that the managed instance group has
	// scheduled for the instance.
	//
	// Possible values:
	//   "ABANDONING"
	//   "CREATING"
	//   "DELETING"
	//   "NONE"
	//   "RECREATING"
	//   "REFRESHING"
	//   "RESTARTING"
	CurrentAction string `json:"currentAction,omitempty"`

	// Id: The unique identifier for this resource (empty when instance does
	// not exist).
	Id uint64 `json:"id,omitempty,string"`

	// Instance: The URL of the instance (set even though instance does not
	// exist yet).
	Instance string `json:"instance,omitempty"`

	// InstanceStatus: The status of the instance (empty when instance does
	// not exist).
	//
	// Possible values:
	//   "PROVISIONING"
	//   "RUNNING"
	//   "STAGING"
	//   "STOPPED"
	//   "STOPPING"
	//   "SUSPENDED"
	//   "SUSPENDING"
	//   "TERMINATED"
	InstanceStatus string `json:"instanceStatus,omitempty"`

	// LastAttempt: Information about the last attempt to create or delete
	// the instance.
	LastAttempt *ManagedInstanceLastAttempt `json:"lastAttempt,omitempty"`
}

type ManagedInstanceLastAttempt struct {
	// Errors: Encountered errors during the last attempt to create or
	// delete the instance.
	Errors *ManagedInstanceLastAttemptErrors `json:"errors,omitempty"`
}

// ManagedInstanceLastAttemptErrors: Encountered errors during the last
// attempt to create or delete the instance.
type ManagedInstanceLastAttemptErrors struct {
	// Errors: [Output Only] The array of errors encountered while
	// processing this operation.
	Errors []*ManagedInstanceLastAttemptErrorsErrors `json:"errors,omitempty"`
}

type ManagedInstanceLastAttemptErrorsErrors struct {
	// Code: [Output Only] The error type identifier for this error.
	Code string `json:"code,omitempty"`

	// Location: [Output Only] Indicates the field in the request which
	// caused the error. This property is optional.
	Location string `json:"location,omitempty"`

	// Message: [Output Only] An optional, human-readable error message.
	Message string `json:"message,omitempty"`
}

// Metadata: A metadata key/value entry.
type Metadata struct {
	// Fingerprint: Specifies a fingerprint for this request, which is
	// essentially a hash of the metadata's contents and used for optimistic
	// locking. The fingerprint is initially generated by Compute Engine and
	// changes after every request to modify or update metadata. You must
	// always provide an up-to-date fingerprint hash in order to update or
	// change metadata.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Items: Array of key/value pairs. The total size of all keys and
	// values must be less than 512 KB.
	Items []*MetadataItems `json:"items,omitempty"`

	// Kind: [Output Only] Type of the resource. Always compute#metadata for
	// metadata.
	Kind string `json:"kind,omitempty"`
}

type MetadataItems struct {
	// Key: Key for the metadata entry. Keys must conform to the following
	// regexp: [a-zA-Z0-9-_]+, and be less than 128 bytes in length. This is
	// reflected as part of a URL in the metadata server. Additionally, to
	// avoid ambiguity, keys must not conflict with any other metadata keys
	// for the project.
	Key string `json:"key,omitempty"`

	// Value: Value for the metadata entry. These are free-form strings, and
	// only have meaning as interpreted by the image running in the
	// instance. The only restriction placed on values is that their size
	// must be less than or equal to 32768 bytes.
	Value string `json:"value,omitempty"`
}

// NamedPort: The named port information. For example: .
type NamedPort struct {
	// Name: The name for this NamedPort.
	Name string `json:"name,omitempty"`

	// Port: The port number, which can be a value between 1 and 65535.
	Port int64 `json:"port,omitempty"`
}

// Network: A network resource.
type Network struct {
	// IPv4Range: The range of internal addresses that are legal on this
	// network. This range is a CIDR specification, for example:
	// 192.168.0.0/16. Provided by the client when the network is created.
	IPv4Range string `json:"IPv4Range,omitempty"`

	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource.
	// Provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// GatewayIPv4: A gateway address for default routing to other networks.
	// This value is read only and is selected by the Google Compute Engine,
	// typically as the first usable address in the IPv4Range.
	GatewayIPv4 string `json:"gatewayIPv4,omitempty"`

	// Id: [Output Only] Unique identifier for the resource. Defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of the resource. Always compute#network for
	// networks.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource. Provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035. Specifically, the name must be 1-63 characters long and
	// match the regular expression [a-z]([-a-z0-9]*[a-z0-9])? which means
	// the first character must be a lowercase letter, and all following
	// characters must be a dash, lowercase letter, or digit, except the
	// last character, which cannot be a dash.
	Name string `json:"name,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// NetworkInterface: A network interface resource attached to an
// instance.
type NetworkInterface struct {
	// AccessConfigs: An array of configurations for this interface.
	// Currently, <codeONE_TO_ONE_NAT is the only access config supported.
	// If there are no accessConfigs specified, then this instance will have
	// no external internet access.
	AccessConfigs []*AccessConfig `json:"accessConfigs,omitempty"`

	// Name: [Output Only] The name of the network interface, generated by
	// the server. For network devices, these are eth0, eth1, etc.
	Name string `json:"name,omitempty"`

	// Network: URL of the network resource for this instance. This is
	// required for creating an instance but optional when creating a
	// firewall rule. If not specified when creating a firewall rule, the
	// default network is used:
	//
	// global/networks/default
	//
	// If you specify this property, you can specify the network as a full
	// or partial URL. For example, the following are all valid URLs:
	// -
	// https://www.googleapis.com/compute/v1/projects/project/global/networks/network
	// - projects/project/global/networks/network
	// - global/networks/default
	Network string `json:"network,omitempty"`

	// NetworkIP: [Output Only] An optional IPV4 internal network address
	// assigned to the instance for this network interface.
	NetworkIP string `json:"networkIP,omitempty"`
}

// NetworkList: Contains a list of Network resources.
type NetworkList struct {
	// Id: [Output Only] Unique identifier for the resource. Defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of Network resources.
	Items []*Network `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#networkList for
	// lists of networks.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource .
	SelfLink string `json:"selfLink,omitempty"`
}

// Operation: An Operation resource, used to manage asynchronous API
// requests.
type Operation struct {
	// ClientOperationId: [Output Only] An optional identifier specified by
	// the client when the mutation was initiated. Must be unique for all
	// Operation resources in the project.
	ClientOperationId string `json:"clientOperationId,omitempty"`

	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// EndTime: [Output Only] The time that this operation was completed.
	// This is in RFC3339 text format.
	EndTime string `json:"endTime,omitempty"`

	// Error: [Output Only] If errors are generated during processing of the
	// operation, this field will be populated.
	Error *OperationError `json:"error,omitempty"`

	// HttpErrorMessage: [Output Only] If the operation fails, this field
	// contains the HTTP error message that was returned, such as NOT FOUND.
	HttpErrorMessage string `json:"httpErrorMessage,omitempty"`

	// HttpErrorStatusCode: [Output Only] If the operation fails, this field
	// contains the HTTP error message that was returned, such as 404.
	HttpErrorStatusCode int64 `json:"httpErrorStatusCode,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// InsertTime: [Output Only] The time that this operation was requested.
	// This is in RFC3339 text format.
	InsertTime string `json:"insertTime,omitempty"`

	// Kind: [Output Only] Type of the resource. Always compute#Operation
	// for Operation resources.
	Kind string `json:"kind,omitempty"`

	// Name: [Output Only] Name of the resource.
	Name string `json:"name,omitempty"`

	// OperationType: [Output Only] Type of the operation, such as insert,
	// update, and delete.
	OperationType string `json:"operationType,omitempty"`

	// Progress: [Output Only] An optional progress indicator that ranges
	// from 0 to 100. There is no requirement that this be linear or support
	// any granularity of operations. This should not be used to guess at
	// when the operation will be complete. This number should monotonically
	// increase as the operation progresses.
	Progress int64 `json:"progress,omitempty"`

	// Region: [Output Only] URL of the region where the operation resides.
	// Only applicable for regional resources.
	Region string `json:"region,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// StartTime: [Output Only] The time that this operation was started by
	// the server. This is in RFC3339 text format.
	StartTime string `json:"startTime,omitempty"`

	// Status: [Output Only] Status of the operation. Can be one of the
	// following: PENDING, RUNNING, or DONE.
	//
	// Possible values:
	//   "DONE"
	//   "PENDING"
	//   "RUNNING"
	Status string `json:"status,omitempty"`

	// StatusMessage: [Output Only] An optional textual description of the
	// current status of the operation.
	StatusMessage string `json:"statusMessage,omitempty"`

	// TargetId: [Output Only] Unique target ID which identifies a
	// particular incarnation of the target.
	TargetId uint64 `json:"targetId,omitempty,string"`

	// TargetLink: [Output Only] URL of the resource the operation is
	// mutating.
	TargetLink string `json:"targetLink,omitempty"`

	// User: [Output Only] User who requested the operation, for example:
	// user@example.com.
	User string `json:"user,omitempty"`

	// Warnings: [Output Only] If warning messages are generated during
	// processing of the operation, this field will be populated.
	Warnings []*OperationWarnings `json:"warnings,omitempty"`

	// Zone: [Output Only] URL of the zone where the operation resides.
	Zone string `json:"zone,omitempty"`
}

// OperationError: [Output Only] If errors are generated during
// processing of the operation, this field will be populated.
type OperationError struct {
	// Errors: [Output Only] The array of errors encountered while
	// processing this operation.
	Errors []*OperationErrorErrors `json:"errors,omitempty"`
}

type OperationErrorErrors struct {
	// Code: [Output Only] The error type identifier for this error.
	Code string `json:"code,omitempty"`

	// Location: [Output Only] Indicates the field in the request which
	// caused the error. This property is optional.
	Location string `json:"location,omitempty"`

	// Message: [Output Only] An optional, human-readable error message.
	Message string `json:"message,omitempty"`
}

type OperationWarnings struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*OperationWarningsData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type OperationWarningsData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type OperationAggregatedList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A map of scoped operation lists.
	Items map[string]OperationsScopedList `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always
	// compute#operationAggregatedList for aggregated lists of operations.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// OperationList: Contains a list of Operation resources.
type OperationList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] The Operation resources.
	Items []*Operation `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#operations for
	// Operations resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncate.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type OperationsScopedList struct {
	// Operations: [Output Only] List of operations contained in this scope.
	Operations []*Operation `json:"operations,omitempty"`

	// Warning: [Output Only] Informational warning which replaces the list
	// of operations when the list is empty.
	Warning *OperationsScopedListWarning `json:"warning,omitempty"`
}

// OperationsScopedListWarning: [Output Only] Informational warning
// which replaces the list of operations when the list is empty.
type OperationsScopedListWarning struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*OperationsScopedListWarningData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type OperationsScopedListWarningData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

// PathMatcher: A matcher for the path portion of the URL. The
// BackendService from the longest-matched rule will serve the URL. If
// no rule was matched, the default_service will be used.
type PathMatcher struct {
	// DefaultService: The URL to the BackendService resource. This will be
	// used if none of the 'pathRules' defined by this PathMatcher is met by
	// the URL's path portion.
	DefaultService string `json:"defaultService,omitempty"`

	// Description: An optional textual description of the resource.
	Description string `json:"description,omitempty"`

	// Name: The name to which this PathMatcher is referred by the HostRule.
	Name string `json:"name,omitempty"`

	// PathRules: The list of path rules.
	PathRules []*PathRule `json:"pathRules,omitempty"`
}

// PathRule: A path-matching rule for a URL. If matched, will use the
// specified BackendService to handle the traffic arriving at this URL.
type PathRule struct {
	// Paths: The list of path patterns to match. Each must start with / and
	// the only place a * is allowed is at the end following a /. The string
	// fed to the path matcher does not include any text after the first ?
	// or #, and those chars are not allowed here.
	Paths []string `json:"paths,omitempty"`

	// Service: The URL of the BackendService resource if this rule is
	// matched.
	Service string `json:"service,omitempty"`
}

// Project: A Project resource. Projects can only be created in the
// Google Developers Console. Unless marked otherwise, values can only
// be modified in the console.
type Project struct {
	// CommonInstanceMetadata: Metadata key/value pairs available to all
	// instances contained in this project. See Custom metadata for more
	// information.
	CommonInstanceMetadata *Metadata `json:"commonInstanceMetadata,omitempty"`

	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource.
	Description string `json:"description,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of the resource. Always compute#project for
	// projects.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource.
	Name string `json:"name,omitempty"`

	// Quotas: [Output Only] Quotas assigned to this project.
	Quotas []*Quota `json:"quotas,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// UsageExportLocation: The location in Cloud Storage and naming method
	// of the daily usage report.
	UsageExportLocation *UsageExportLocation `json:"usageExportLocation,omitempty"`
}

// Quota: A quotas entry.
type Quota struct {
	// Limit: [Output Only] Quota limit for this metric.
	Limit float64 `json:"limit,omitempty"`

	// Metric: [Output Only] Name of the quota metric.
	//
	// Possible values:
	//   "BACKEND_SERVICES"
	//   "CPUS"
	//   "DISKS_TOTAL_GB"
	//   "FIREWALLS"
	//   "FORWARDING_RULES"
	//   "HEALTH_CHECKS"
	//   "IMAGES"
	//   "INSTANCES"
	//   "INSTANCE_GROUPS"
	//   "INSTANCE_GROUP_MANAGERS"
	//   "INSTANCE_TEMPLATES"
	//   "IN_USE_ADDRESSES"
	//   "LOCAL_SSD_TOTAL_GB"
	//   "NETWORKS"
	//   "ROUTES"
	//   "SNAPSHOTS"
	//   "SSD_TOTAL_GB"
	//   "STATIC_ADDRESSES"
	//   "TARGET_HTTP_PROXIES"
	//   "TARGET_INSTANCES"
	//   "TARGET_POOLS"
	//   "TARGET_VPN_GATEWAYS"
	//   "URL_MAPS"
	//   "VPN_TUNNELS"
	Metric string `json:"metric,omitempty"`

	// Usage: [Output Only] Current usage of this metric.
	Usage float64 `json:"usage,omitempty"`
}

// Region: Region resource.
type Region struct {
	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Deprecated: [Output Only] The deprecation status associated with this
	// region.
	Deprecated *DeprecationStatus `json:"deprecated,omitempty"`

	// Description: [Output Only] Textual description of the resource.
	Description string `json:"description,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server .
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of the resource. Always compute#region for
	// regions.
	Kind string `json:"kind,omitempty"`

	// Name: [Output Only] Name of the resource.
	Name string `json:"name,omitempty"`

	// Quotas: [Output Only] Quotas assigned to this region.
	Quotas []*Quota `json:"quotas,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Status: [Output Only] Status of the region, either UP or DOWN.
	//
	// Possible values:
	//   "DOWN"
	//   "UP"
	Status string `json:"status,omitempty"`

	// Zones: [Output Only] A list of zones available in this region, in the
	// form of resource URLs.
	Zones []string `json:"zones,omitempty"`
}

// RegionList: Contains a list of region resources.
type RegionList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of Region resources.
	Items []*Region `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#regionList for
	// lists of regions.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type ResourceGroupReference struct {
	// Group: A URI referencing one of the resource views listed in the
	// backend service.
	Group string `json:"group,omitempty"`
}

// Route: The route resource. A Route is a rule that specifies how
// certain packets should be handled by the virtual network. Routes are
// associated with VMs by tag and the set of Routes for a particular VM
// is called its routing table. For each packet leaving a VM, the system
// searches that VM's routing table for a single best matching Route.
// Routes match packets by destination IP address, preferring smaller or
// more specific ranges over larger ones. If there is a tie, the system
// selects the Route with the smallest priority value. If there is still
// a tie, it uses the layer three and four packet headers to select just
// one of the remaining matching Routes. The packet is then forwarded as
// specified by the nextHop field of the winning Route -- either to
// another VM destination, a VM gateway or a GCE operated gateway.
// Packets that do not match any Route in the sending VM's routing table
// are dropped.
type Route struct {
	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource.
	// Provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// DestRange: The destination range of outgoing packets that this route
	// applies to.
	DestRange string `json:"destRange,omitempty"`

	// Id: [Output Only] Unique identifier for the resource. Defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of this resource. Always compute#routes for
	// Route resources.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035. Specifically, the name must be 1-63 characters long and
	// match the regular expression [a-z]([-a-z0-9]*[a-z0-9])? which means
	// the first character must be a lowercase letter, and all following
	// characters must be a dash, lowercase letter, or digit, except the
	// last character, which cannot be a dash.
	Name string `json:"name,omitempty"`

	// Network: Fully-qualified URL of the network that this route applies
	// to.
	Network string `json:"network,omitempty"`

	// NextHopGateway: The URL to a gateway that should handle matching
	// packets. Currently, this is only the internet gateway:
	// projects/<project-id>/global/gateways/default-internet-gateway
	NextHopGateway string `json:"nextHopGateway,omitempty"`

	// NextHopInstance: The fully-qualified URL to an instance that should
	// handle matching packets. For
	// example:
	// https://www.googleapis.com/compute/v1/projects/project/zones/
	// zone/instances/
	NextHopInstance string `json:"nextHopInstance,omitempty"`

	// NextHopIp: The network IP address of an instance that should handle
	// matching packets.
	NextHopIp string `json:"nextHopIp,omitempty"`

	// NextHopNetwork: The URL of the local network if it should handle
	// matching packets.
	NextHopNetwork string `json:"nextHopNetwork,omitempty"`

	// NextHopVpnTunnel: The URL to a VpnTunnel that should handle matching
	// packets.
	NextHopVpnTunnel string `json:"nextHopVpnTunnel,omitempty"`

	// Priority: Breaks ties between Routes of equal specificity. Routes
	// with smaller values win when tied with routes with larger values.
	// Default value is 1000. A valid range is between 0 and 65535.
	Priority int64 `json:"priority,omitempty"`

	// SelfLink: [Output Only] Server-defined fully-qualified URL for this
	// resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Tags: A list of instance tags to which this route applies.
	Tags []string `json:"tags,omitempty"`

	// Warnings: [Output Only] If potential misconfigurations are detected
	// for this route, this field will be populated with warning messages.
	Warnings []*RouteWarnings `json:"warnings,omitempty"`
}

type RouteWarnings struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*RouteWarningsData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type RouteWarningsData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

// RouteList: Contains a list of route resources.
type RouteList struct {
	// Id: [Output Only] Unique identifier for the resource. Defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: A list of Route resources.
	Items []*Route `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// Scheduling: Sets the scheduling options for an Instance.
type Scheduling struct {
	// AutomaticRestart: Specifies whether the instance should be
	// automatically restarted if it is terminated by Compute Engine (not
	// terminated by a user).
	AutomaticRestart bool `json:"automaticRestart,omitempty"`

	// OnHostMaintenance: Defines the maintenance behavior for this
	// instance. The default behavior is MIGRATE. For more information, see
	// Setting maintenance behavior.
	//
	// Possible values:
	//   "MIGRATE"
	//   "TERMINATE"
	OnHostMaintenance string `json:"onHostMaintenance,omitempty"`

	// Preemptible: Whether the Instance is preemptible.
	Preemptible bool `json:"preemptible,omitempty"`
}

// SerialPortOutput: An instance's serial console output.
type SerialPortOutput struct {
	// Contents: [Output Only] The contents of the console output.
	Contents string `json:"contents,omitempty"`

	// Kind: [Output Only] Type of the resource. Always
	// compute#serialPortOutput for serial port output.
	Kind string `json:"kind,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// ServiceAccount: A service account.
type ServiceAccount struct {
	// Email: Email address of the service account.
	Email string `json:"email,omitempty"`

	// Scopes: The list of scopes to be made available for this service
	// account.
	Scopes []string `json:"scopes,omitempty"`
}

// Snapshot: A persistent disk snapshot resource.
type Snapshot struct {
	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// DiskSizeGb: [Output Only] Size of the snapshot, specified in GB.
	DiskSizeGb int64 `json:"diskSizeGb,omitempty,string"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of the resource. Always compute#snapshot for
	// Snapshot resources.
	Kind string `json:"kind,omitempty"`

	// Licenses: Public visible licenses.
	Licenses []string `json:"licenses,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035. Specifically, the name must be 1-63 characters long and
	// match the regular expression [a-z]([-a-z0-9]*[a-z0-9])? which means
	// the first character must be a lowercase letter, and all following
	// characters must be a dash, lowercase letter, or digit, except the
	// last character, which cannot be a dash.
	Name string `json:"name,omitempty"`

	// SelfLink: [Output Only] Server-defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// SourceDisk: The source disk used to create this snapshot.
	SourceDisk string `json:"sourceDisk,omitempty"`

	// SourceDiskId: [Output Only] The ID value of the disk used to create
	// this snapshot. This value may be used to determine whether the
	// snapshot was taken from the current or a previous instance of a given
	// disk name.
	SourceDiskId string `json:"sourceDiskId,omitempty"`

	// Status: [Output Only] The status of the snapshot.
	//
	// Possible values:
	//   "CREATING"
	//   "DELETING"
	//   "FAILED"
	//   "READY"
	//   "UPLOADING"
	Status string `json:"status,omitempty"`

	// StorageBytes: [Output Only] A size of the the storage used by the
	// snapshot. As snapshots share storage, this number is expected to
	// change with snapshot creation/deletion.
	StorageBytes int64 `json:"storageBytes,omitempty,string"`

	// StorageBytesStatus: [Output Only] An indicator whether storageBytes
	// is in a stable state or it is being adjusted as a result of shared
	// storage reallocation.
	//
	// Possible values:
	//   "UPDATING"
	//   "UP_TO_DATE"
	StorageBytesStatus string `json:"storageBytesStatus,omitempty"`
}

// SnapshotList: Contains a list of Snapshot resources.
type SnapshotList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A list of Snapshot resources.
	Items []*Snapshot `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

// Tags: A set of instance tags.
type Tags struct {
	// Fingerprint: Specifies a fingerprint for this request, which is
	// essentially a hash of the metadata's contents and used for optimistic
	// locking. The fingerprint is initially generated by Compute Engine and
	// changes after every request to modify or update metadata. You must
	// always provide an up-to-date fingerprint hash in order to update or
	// change metadata.
	//
	// To see the latest fingerprint, make get() request to the instance.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Items: An array of tags. Each tag must be 1-63 characters long, and
	// comply with RFC1035.
	Items []string `json:"items,omitempty"`
}

// TargetHttpProxy: A TargetHttpProxy resource. This resource defines an
// HTTP proxy.
type TargetHttpProxy struct {
	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// Id: [Output Only] Unique identifier for the resource. Defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of resource. Always compute#Operation for
	// Operation resources.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035. Specifically, the name must be 1-63 characters long and
	// match the regular expression [a-z]([-a-z0-9]*[a-z0-9])? which means
	// the first character must be a lowercase letter, and all following
	// characters must be a dash, lowercase letter, or digit, except the
	// last character, which cannot be a dash.
	Name string `json:"name,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// UrlMap: URL to the UrlMap resource that defines the mapping from URL
	// to the BackendService.
	UrlMap string `json:"urlMap,omitempty"`
}

// TargetHttpProxyList: A list of TargetHttpProxy resources.
type TargetHttpProxyList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: A list of TargetHttpProxy resources.
	Items []*TargetHttpProxy `json:"items,omitempty"`

	// Kind: Type of resource. Always compute#targetHttpProxyList for lists
	// of Target HTTP proxies.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server-defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// TargetInstance: A TargetInstance resource. This resource defines an
// endpoint VM that terminates traffic of certain protocols.
type TargetInstance struct {
	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id uint64 `json:"id,omitempty,string"`

	// Instance: The URL to the instance that terminates the relevant
	// traffic.
	Instance string `json:"instance,omitempty"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035.
	Name string `json:"name,omitempty"`

	// NatPolicy: NAT option controlling how IPs are NAT'ed to the VM.
	// Currently only NO_NAT (default value) is supported.
	//
	// Possible values:
	//   "NO_NAT"
	NatPolicy string `json:"natPolicy,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// Zone: URL of the zone where the target instance resides (output
	// only).
	Zone string `json:"zone,omitempty"`
}

type TargetInstanceAggregatedList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A map of scoped target instance lists.
	Items map[string]TargetInstancesScopedList `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

// TargetInstanceList: Contains a list of TargetInstance resources.
type TargetInstanceList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A list of TargetInstance resources.
	Items []*TargetInstance `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type TargetInstancesScopedList struct {
	// TargetInstances: List of target instances contained in this scope.
	TargetInstances []*TargetInstance `json:"targetInstances,omitempty"`

	// Warning: Informational warning which replaces the list of addresses
	// when the list is empty.
	Warning *TargetInstancesScopedListWarning `json:"warning,omitempty"`
}

// TargetInstancesScopedListWarning: Informational warning which
// replaces the list of addresses when the list is empty.
type TargetInstancesScopedListWarning struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*TargetInstancesScopedListWarningData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type TargetInstancesScopedListWarningData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

// TargetPool: A TargetPool resource. This resource defines a pool of
// VMs, associated HttpHealthCheck resources, and the fallback
// TargetPool.
type TargetPool struct {
	// BackupPool: This field is applicable only when the containing target
	// pool is serving a forwarding rule as the primary pool, and its
	// 'failoverRatio' field is properly set to a value between [0,
	// 1].
	//
	// 'backupPool' and 'failoverRatio' together define the fallback
	// behavior of the primary target pool: if the ratio of the healthy VMs
	// in the primary pool is at or below 'failoverRatio', traffic arriving
	// at the load-balanced IP will be directed to the backup pool.
	//
	// In case where 'failoverRatio' and 'backupPool' are not set, or all
	// the VMs in the backup pool are unhealthy, the traffic will be
	// directed back to the primary pool in the "force" mode, where traffic
	// will be spread to the healthy VMs with the best effort, or to all VMs
	// when no VM is healthy.
	BackupPool string `json:"backupPool,omitempty"`

	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// FailoverRatio: This field is applicable only when the containing
	// target pool is serving a forwarding rule as the primary pool (i.e.,
	// not as a backup pool to some other target pool). The value of the
	// field must be in [0, 1].
	//
	// If set, 'backupPool' must also be set. They together define the
	// fallback behavior of the primary target pool: if the ratio of the
	// healthy VMs in the primary pool is at or below this number, traffic
	// arriving at the load-balanced IP will be directed to the backup
	// pool.
	//
	// In case where 'failoverRatio' is not set or all the VMs in the backup
	// pool are unhealthy, the traffic will be directed back to the primary
	// pool in the "force" mode, where traffic will be spread to the healthy
	// VMs with the best effort, or to all VMs when no VM is healthy.
	FailoverRatio float64 `json:"failoverRatio,omitempty"`

	// HealthChecks: A list of URLs to the HttpHealthCheck resource. A
	// member VM in this pool is considered healthy if and only if all
	// specified health checks pass. An empty list means all member VMs will
	// be considered healthy at all times.
	HealthChecks []string `json:"healthChecks,omitempty"`

	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id uint64 `json:"id,omitempty,string"`

	// Instances: A list of resource URLs to the member VMs serving this
	// pool. They must live in zones contained in the same region as this
	// pool.
	Instances []string `json:"instances,omitempty"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035.
	Name string `json:"name,omitempty"`

	// Region: URL of the region where the target pool resides (output
	// only).
	Region string `json:"region,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// SessionAffinity: Sesssion affinity option, must be one of the
	// following values: 'NONE': Connections from the same client IP may go
	// to any VM in the pool; 'CLIENT_IP': Connections from the same client
	// IP will go to the same VM in the pool while that VM remains healthy.
	// 'CLIENT_IP_PROTO': Connections from the same client IP with the same
	// IP protocol will go to the same VM in the pool while that VM remains
	// healthy.
	//
	// Possible values:
	//   "CLIENT_IP"
	//   "CLIENT_IP_PROTO"
	//   "NONE"
	SessionAffinity string `json:"sessionAffinity,omitempty"`
}

type TargetPoolAggregatedList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A map of scoped target pool lists.
	Items map[string]TargetPoolsScopedList `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type TargetPoolInstanceHealth struct {
	HealthStatus []*HealthStatus `json:"healthStatus,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`
}

// TargetPoolList: Contains a list of TargetPool resources.
type TargetPoolList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A list of TargetPool resources.
	Items []*TargetPool `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type TargetPoolsAddHealthCheckRequest struct {
	// HealthChecks: Health check URLs to be added to targetPool.
	HealthChecks []*HealthCheckReference `json:"healthChecks,omitempty"`
}

type TargetPoolsAddInstanceRequest struct {
	// Instances: URLs of the instances to be added to targetPool.
	Instances []*InstanceReference `json:"instances,omitempty"`
}

type TargetPoolsRemoveHealthCheckRequest struct {
	// HealthChecks: Health check URLs to be removed from targetPool.
	HealthChecks []*HealthCheckReference `json:"healthChecks,omitempty"`
}

type TargetPoolsRemoveInstanceRequest struct {
	// Instances: URLs of the instances to be removed from targetPool.
	Instances []*InstanceReference `json:"instances,omitempty"`
}

type TargetPoolsScopedList struct {
	// TargetPools: List of target pools contained in this scope.
	TargetPools []*TargetPool `json:"targetPools,omitempty"`

	// Warning: Informational warning which replaces the list of addresses
	// when the list is empty.
	Warning *TargetPoolsScopedListWarning `json:"warning,omitempty"`
}

// TargetPoolsScopedListWarning: Informational warning which replaces
// the list of addresses when the list is empty.
type TargetPoolsScopedListWarning struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*TargetPoolsScopedListWarningData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type TargetPoolsScopedListWarningData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type TargetReference struct {
	Target string `json:"target,omitempty"`
}

type TargetVpnGateway struct {
	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource.
	// Provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// ForwardingRules: [Output Only] A list of URLs to the ForwardingRule
	// resources. ForwardingRules are created using
	// compute.forwardingRules.insert and associated to a VPN gateway.
	ForwardingRules []string `json:"forwardingRules,omitempty"`

	// Id: [Output Only] Unique identifier for the resource. Defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of resource. Always compute#targetVpnGateway
	// for target VPN gateways.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035. Specifically, the name must be 1-63 characters long and
	// match the regular expression [a-z]([-a-z0-9]*[a-z0-9])? which means
	// the first character must be a lowercase letter, and all following
	// characters must be a dash, lowercase letter, or digit, except the
	// last character, which cannot be a dash.
	Name string `json:"name,omitempty"`

	// Network: URL of the network to which this VPN gateway is attached.
	// Provided by the client when the VPN gateway is created.
	Network string `json:"network,omitempty"`

	// Region: [Output Only] URL of the region where the target VPN gateway
	// resides.
	Region string `json:"region,omitempty"`

	// SelfLink: [Output Only] Server-defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Status: [Output Only] The status of the VPN gateway.
	//
	// Possible values:
	//   "CREATING"
	//   "DELETING"
	//   "FAILED"
	//   "READY"
	Status string `json:"status,omitempty"`

	// Tunnels: [Output Only] A list of URLs to VpnTunnel resources.
	// VpnTunnels are created using compute.vpntunnels.insert and associated
	// to a VPN gateway.
	Tunnels []string `json:"tunnels,omitempty"`
}

type TargetVpnGatewayAggregatedList struct {
	// Id: [Output Only] Unique identifier for the resource. Defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: A map of scoped target vpn gateway lists.
	Items map[string]TargetVpnGatewaysScopedList `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#targetVpnGateway
	// for target VPN gateways.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server-defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// TargetVpnGatewayList: Contains a list of TargetVpnGateway resources.
type TargetVpnGatewayList struct {
	// Id: [Output Only] Unique identifier for the resource. Defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of TargetVpnGateway resources.
	Items []*TargetVpnGateway `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#targetVpnGateway
	// for target VPN gateways.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server-defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type TargetVpnGatewaysScopedList struct {
	// TargetVpnGateways: [Output Only] List of target vpn gateways
	// contained in this scope.
	TargetVpnGateways []*TargetVpnGateway `json:"targetVpnGateways,omitempty"`

	// Warning: [Output Only] Informational warning which replaces the list
	// of addresses when the list is empty.
	Warning *TargetVpnGatewaysScopedListWarning `json:"warning,omitempty"`
}

// TargetVpnGatewaysScopedListWarning: [Output Only] Informational
// warning which replaces the list of addresses when the list is empty.
type TargetVpnGatewaysScopedListWarning struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*TargetVpnGatewaysScopedListWarningData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type TargetVpnGatewaysScopedListWarningData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type TestFailure struct {
	ActualService string `json:"actualService,omitempty"`

	ExpectedService string `json:"expectedService,omitempty"`

	Host string `json:"host,omitempty"`

	Path string `json:"path,omitempty"`
}

// UrlMap: A UrlMap resource. This resource defines the mapping from URL
// to the BackendService resource, based on the "longest-match" of the
// URL's host and path.
type UrlMap struct {
	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// DefaultService: The URL of the BackendService resource if none of the
	// hostRules match.
	DefaultService string `json:"defaultService,omitempty"`

	// Description: An optional textual description of the resource.
	// Provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// Fingerprint: Fingerprint of this resource. A hash of the contents
	// stored in this object. This field is used in optimistic locking. This
	// field will be ignored when inserting a UrlMap. An up-to-date
	// fingerprint must be provided in order to update the UrlMap.
	Fingerprint string `json:"fingerprint,omitempty"`

	// HostRules: The list of HostRules to use against the URL.
	HostRules []*HostRule `json:"hostRules,omitempty"`

	// Id: [Output Only] Unique identifier for the resource. Set by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource. Provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035. Specifically, the name must be 1-63 characters long and
	// match the regular expression [a-z]([-a-z0-9]*[a-z0-9])? which means
	// the first character must be a lowercase letter, and all following
	// characters must be a dash, lowercase letter, or digit, except the
	// last character, which cannot be a dash.
	Name string `json:"name,omitempty"`

	// PathMatchers: The list of named PathMatchers to use against the URL.
	PathMatchers []*PathMatcher `json:"pathMatchers,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Tests: The list of expected URL mappings. Request to update this
	// UrlMap will succeed only all of the test cases pass.
	Tests []*UrlMapTest `json:"tests,omitempty"`
}

// UrlMapList: Contains a list of UrlMap resources.
type UrlMapList struct {
	// Id: [Output Only] Unique identifier for the resource. Set by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: A list of UrlMap resources.
	Items []*UrlMap `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type UrlMapReference struct {
	UrlMap string `json:"urlMap,omitempty"`
}

// UrlMapTest: Message for the expected URL mappings.
type UrlMapTest struct {
	// Description: Description of this test case.
	Description string `json:"description,omitempty"`

	// Host: Host portion of the URL.
	Host string `json:"host,omitempty"`

	// Path: Path portion of the URL.
	Path string `json:"path,omitempty"`

	// Service: Expected BackendService resource the given URL should be
	// mapped to.
	Service string `json:"service,omitempty"`
}

// UrlMapValidationResult: Message representing the validation result
// for a UrlMap.
type UrlMapValidationResult struct {
	LoadErrors []string `json:"loadErrors,omitempty"`

	// LoadSucceeded: Whether the given UrlMap can be successfully loaded.
	// If false, 'loadErrors' indicates the reasons.
	LoadSucceeded bool `json:"loadSucceeded,omitempty"`

	TestFailures []*TestFailure `json:"testFailures,omitempty"`

	// TestPassed: If successfully loaded, this field indicates whether the
	// test passed. If false, 'testFailures's indicate the reason of
	// failure.
	TestPassed bool `json:"testPassed,omitempty"`
}

type UrlMapsValidateRequest struct {
	// Resource: Content of the UrlMap to be validated.
	Resource *UrlMap `json:"resource,omitempty"`
}

type UrlMapsValidateResponse struct {
	Result *UrlMapValidationResult `json:"result,omitempty"`
}

// UsageExportLocation: The location in Cloud Storage and naming method
// of the daily usage report. Contains bucket_name and report_name
// prefix.
type UsageExportLocation struct {
	// BucketName: The name of an existing bucket in Cloud Storage where the
	// usage report object is stored. The Google Service Account is granted
	// write access to this bucket. This is just the bucket name, with no
	// gs:// or https://storage.googleapis.com/ in front of it.
	BucketName string `json:"bucketName,omitempty"`

	// ReportNamePrefix: An optional prefix for the name of the usage report
	// object stored in bucketName. If not supplied, defaults to usage. The
	// report is stored as a CSV file named
	// report_name_prefix_gce_YYYYMMDD.csv where YYYYMMDD is the day of the
	// usage according to Pacific Time. If you supply a prefix, it should
	// conform to Cloud Storage object naming conventions.
	ReportNamePrefix string `json:"reportNamePrefix,omitempty"`
}

type VpnTunnel struct {
	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource.
	// Provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// DetailedStatus: [Output Only] Detailed status message for the VPN
	// tunnel.
	DetailedStatus string `json:"detailedStatus,omitempty"`

	// Id: [Output Only] Unique identifier for the resource. Defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// IkeNetworks: IKE networks to use when establishing the VPN tunnel
	// with peer VPN gateway. The value should be a CIDR formatted string,
	// for example: 192.168.0.0/16. The ranges should be disjoint.
	IkeNetworks []string `json:"ikeNetworks,omitempty"`

	// IkeVersion: IKE protocol version to use when establishing the VPN
	// tunnel with peer VPN gateway. Acceptable IKE versions are 1 or 2.
	// Default version is 2.
	IkeVersion int64 `json:"ikeVersion,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#vpnTunnel for
	// VPN tunnels.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035. Specifically, the name must be 1-63 characters long and
	// match the regular expression [a-z]([-a-z0-9]*[a-z0-9])? which means
	// the first character must be a lowercase letter, and all following
	// characters must be a dash, lowercase letter, or digit, except the
	// last character, which cannot be a dash.
	Name string `json:"name,omitempty"`

	// PeerIp: IP address of the peer VPN gateway.
	PeerIp string `json:"peerIp,omitempty"`

	// Region: [Output Only] URL of the region where the VPN tunnel resides.
	Region string `json:"region,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// SharedSecret: Shared secret used to set the secure session between
	// the GCE VPN gateway and the peer VPN gateway.
	SharedSecret string `json:"sharedSecret,omitempty"`

	// SharedSecretHash: Hash of the shared secret.
	SharedSecretHash string `json:"sharedSecretHash,omitempty"`

	// Status: [Output Only] The status of the VPN tunnel.
	//
	// Possible values:
	//   "AUTHORIZATION_ERROR"
	//   "DEPROVISIONING"
	//   "ESTABLISHED"
	//   "FAILED"
	//   "FIRST_HANDSHAKE"
	//   "NEGOTIATION_FAILURE"
	//   "NETWORK_ERROR"
	//   "NO_INCOMING_PACKETS"
	//   "PROVISIONING"
	//   "REJECTED"
	//   "WAITING_FOR_FULL_CONFIG"
	Status string `json:"status,omitempty"`

	// TargetVpnGateway: URL of the VPN gateway to which this VPN tunnel is
	// associated. Provided by the client when the VPN tunnel is created.
	TargetVpnGateway string `json:"targetVpnGateway,omitempty"`
}

type VpnTunnelAggregatedList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A map of scoped vpn tunnel lists.
	Items map[string]VpnTunnelsScopedList `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#vpnTunnel for
	// VPN tunnels.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

// VpnTunnelList: Contains a list of VpnTunnel resources.
type VpnTunnelList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of VpnTunnel resources.
	Items []*VpnTunnel `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#vpnTunnel for
	// VPN tunnels.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server-defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type VpnTunnelsScopedList struct {
	// VpnTunnels: List of vpn tunnels contained in this scope.
	VpnTunnels []*VpnTunnel `json:"vpnTunnels,omitempty"`

	// Warning: Informational warning which replaces the list of addresses
	// when the list is empty.
	Warning *VpnTunnelsScopedListWarning `json:"warning,omitempty"`
}

// VpnTunnelsScopedListWarning: Informational warning which replaces the
// list of addresses when the list is empty.
type VpnTunnelsScopedListWarning struct {
	// Code: [Output Only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata for this warning in key: value format.
	Data []*VpnTunnelsScopedListWarningData `json:"data,omitempty"`

	// Message: [Output Only] Optional human-readable details for this
	// warning.
	Message string `json:"message,omitempty"`
}

type VpnTunnelsScopedListWarningData struct {
	// Key: [Output Only] A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

// Zone: A Zone resource.
type Zone struct {
	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Deprecated: [Output Only] The deprecation status associated with this
	// zone.
	Deprecated *DeprecationStatus `json:"deprecated,omitempty"`

	// Description: [Output Only] Textual description of the resource.
	Description string `json:"description,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of the resource. Always kind#zone for zones.
	Kind string `json:"kind,omitempty"`

	// MaintenanceWindows: [Output Only] Any scheduled maintenance windows
	// for this zone. When the zone is in a maintenance window, all
	// resources which reside in the zone will be unavailable. For more
	// information, see Maintenance Windows
	MaintenanceWindows []*ZoneMaintenanceWindows `json:"maintenanceWindows,omitempty"`

	// Name: [Output Only] Name of the resource.
	Name string `json:"name,omitempty"`

	// Region: [Output Only] Full URL reference to the region which hosts
	// the zone.
	Region string `json:"region,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Status: [Output Only] Status of the zone, either UP or DOWN.
	//
	// Possible values:
	//   "DOWN"
	//   "UP"
	Status string `json:"status,omitempty"`
}

type ZoneMaintenanceWindows struct {
	// BeginTime: [Output Only] Starting time of the maintenance window, in
	// RFC3339 format.
	BeginTime string `json:"beginTime,omitempty"`

	// Description: [Output Only] Textual description of the maintenance
	// window.
	Description string `json:"description,omitempty"`

	// EndTime: [Output Only] Ending time of the maintenance window, in
	// RFC3339 format.
	EndTime string `json:"endTime,omitempty"`

	// Name: [Output Only] Name of the maintenance window.
	Name string `json:"name,omitempty"`
}

// ZoneList: Contains a list of zone resources.
type ZoneList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of Zone resources.
	Items []*Zone `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

// method id "compute.addresses.aggregatedList":

type AddressesAggregatedListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// AggregatedList: Retrieves the list of addresses grouped by scope.
// For details, see https://cloud.google.com/compute/docs/reference/latest/addresses/aggregatedList
func (r *AddressesService) AggregatedList(project string) *AddressesAggregatedListCall {
	c := &AddressesAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *AddressesAggregatedListCall) Filter(filter string) *AddressesAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *AddressesAggregatedListCall) MaxResults(maxResults int64) *AddressesAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *AddressesAggregatedListCall) PageToken(pageToken string) *AddressesAggregatedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AddressesAggregatedListCall) Fields(s ...googleapi.Field) *AddressesAggregatedListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *AddressesAggregatedListCall) Do() (*AddressAggregatedList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/aggregated/addresses")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *AddressAggregatedList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of addresses grouped by scope.",
	//   "httpMethod": "GET",
	//   "id": "compute.addresses.aggregatedList",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/aggregated/addresses",
	//   "response": {
	//     "$ref": "AddressAggregatedList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.addresses.delete":

type AddressesDeleteCall struct {
	s       *Service
	project string
	region  string
	address string
	opt_    map[string]interface{}
}

// Delete: Deletes the specified address resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/addresses/delete
func (r *AddressesService) Delete(project string, region string, address string) *AddressesDeleteCall {
	c := &AddressesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.address = address
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AddressesDeleteCall) Fields(s ...googleapi.Field) *AddressesDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *AddressesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/addresses/{address}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"region":  c.region,
		"address": c.address,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified address resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.addresses.delete",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "address"
	//   ],
	//   "parameters": {
	//     "address": {
	//       "description": "Name of the address resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The name of the region for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/addresses/{address}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.addresses.get":

type AddressesGetCall struct {
	s       *Service
	project string
	region  string
	address string
	opt_    map[string]interface{}
}

// Get: Returns the specified address resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/addresses/get
func (r *AddressesService) Get(project string, region string, address string) *AddressesGetCall {
	c := &AddressesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.address = address
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AddressesGetCall) Fields(s ...googleapi.Field) *AddressesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *AddressesGetCall) Do() (*Address, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/addresses/{address}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"region":  c.region,
		"address": c.address,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Address
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified address resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.addresses.get",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "address"
	//   ],
	//   "parameters": {
	//     "address": {
	//       "description": "Name of the address resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The name of the region for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/addresses/{address}",
	//   "response": {
	//     "$ref": "Address"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.addresses.insert":

type AddressesInsertCall struct {
	s       *Service
	project string
	region  string
	address *Address
	opt_    map[string]interface{}
}

// Insert: Creates an address resource in the specified project using
// the data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/addresses/insert
func (r *AddressesService) Insert(project string, region string, address *Address) *AddressesInsertCall {
	c := &AddressesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.address = address
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AddressesInsertCall) Fields(s ...googleapi.Field) *AddressesInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *AddressesInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.address)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/addresses")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"region":  c.region,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates an address resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.addresses.insert",
	//   "parameterOrder": [
	//     "project",
	//     "region"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The name of the region for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/addresses",
	//   "request": {
	//     "$ref": "Address"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.addresses.list":

type AddressesListCall struct {
	s       *Service
	project string
	region  string
	opt_    map[string]interface{}
}

// List: Retrieves the list of address resources contained within the
// specified region.
// For details, see https://cloud.google.com/compute/docs/reference/latest/addresses/list
func (r *AddressesService) List(project string, region string) *AddressesListCall {
	c := &AddressesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *AddressesListCall) Filter(filter string) *AddressesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *AddressesListCall) MaxResults(maxResults int64) *AddressesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *AddressesListCall) PageToken(pageToken string) *AddressesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AddressesListCall) Fields(s ...googleapi.Field) *AddressesListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *AddressesListCall) Do() (*AddressList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/addresses")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"region":  c.region,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *AddressList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of address resources contained within the specified region.",
	//   "httpMethod": "GET",
	//   "id": "compute.addresses.list",
	//   "parameterOrder": [
	//     "project",
	//     "region"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The name of the region for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/addresses",
	//   "response": {
	//     "$ref": "AddressList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.autoscalers.aggregatedList":

type AutoscalersAggregatedListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// AggregatedList: Retrieves the list of autoscalers grouped by scope.
func (r *AutoscalersService) AggregatedList(project string) *AutoscalersAggregatedListCall {
	c := &AutoscalersAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *AutoscalersAggregatedListCall) Filter(filter string) *AutoscalersAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *AutoscalersAggregatedListCall) MaxResults(maxResults int64) *AutoscalersAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *AutoscalersAggregatedListCall) PageToken(pageToken string) *AutoscalersAggregatedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AutoscalersAggregatedListCall) Fields(s ...googleapi.Field) *AutoscalersAggregatedListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *AutoscalersAggregatedListCall) Do() (*AutoscalerAggregatedList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/aggregated/autoscalers")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *AutoscalerAggregatedList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of autoscalers grouped by scope.",
	//   "httpMethod": "GET",
	//   "id": "compute.autoscalers.aggregatedList",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/aggregated/autoscalers",
	//   "response": {
	//     "$ref": "AutoscalerAggregatedList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.autoscalers.delete":

type AutoscalersDeleteCall struct {
	s          *Service
	project    string
	zone       string
	autoscaler string
	opt_       map[string]interface{}
}

// Delete: Deletes the specified autoscaler resource.
func (r *AutoscalersService) Delete(project string, zone string, autoscaler string) *AutoscalersDeleteCall {
	c := &AutoscalersDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.autoscaler = autoscaler
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AutoscalersDeleteCall) Fields(s ...googleapi.Field) *AutoscalersDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *AutoscalersDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/autoscalers/{autoscaler}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"zone":       c.zone,
		"autoscaler": c.autoscaler,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified autoscaler resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.autoscalers.delete",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "autoscaler"
	//   ],
	//   "parameters": {
	//     "autoscaler": {
	//       "description": "Name of the persistent autoscaler resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/autoscalers/{autoscaler}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.autoscalers.get":

type AutoscalersGetCall struct {
	s          *Service
	project    string
	zone       string
	autoscaler string
	opt_       map[string]interface{}
}

// Get: Returns the specified autoscaler resource.
func (r *AutoscalersService) Get(project string, zone string, autoscaler string) *AutoscalersGetCall {
	c := &AutoscalersGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.autoscaler = autoscaler
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AutoscalersGetCall) Fields(s ...googleapi.Field) *AutoscalersGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *AutoscalersGetCall) Do() (*Autoscaler, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/autoscalers/{autoscaler}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"zone":       c.zone,
		"autoscaler": c.autoscaler,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Autoscaler
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified autoscaler resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.autoscalers.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "autoscaler"
	//   ],
	//   "parameters": {
	//     "autoscaler": {
	//       "description": "Name of the persistent autoscaler resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/autoscalers/{autoscaler}",
	//   "response": {
	//     "$ref": "Autoscaler"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.autoscalers.insert":

type AutoscalersInsertCall struct {
	s          *Service
	project    string
	zone       string
	autoscaler *Autoscaler
	opt_       map[string]interface{}
}

// Insert: Creates an autoscaler resource in the specified project using
// the data included in the request.
func (r *AutoscalersService) Insert(project string, zone string, autoscaler *Autoscaler) *AutoscalersInsertCall {
	c := &AutoscalersInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.autoscaler = autoscaler
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AutoscalersInsertCall) Fields(s ...googleapi.Field) *AutoscalersInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *AutoscalersInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.autoscaler)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/autoscalers")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates an autoscaler resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.autoscalers.insert",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/autoscalers",
	//   "request": {
	//     "$ref": "Autoscaler"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.autoscalers.list":

type AutoscalersListCall struct {
	s       *Service
	project string
	zone    string
	opt_    map[string]interface{}
}

// List: Retrieves the list of autoscaler resources contained within the
// specified zone.
func (r *AutoscalersService) List(project string, zone string) *AutoscalersListCall {
	c := &AutoscalersListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *AutoscalersListCall) Filter(filter string) *AutoscalersListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *AutoscalersListCall) MaxResults(maxResults int64) *AutoscalersListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *AutoscalersListCall) PageToken(pageToken string) *AutoscalersListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AutoscalersListCall) Fields(s ...googleapi.Field) *AutoscalersListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *AutoscalersListCall) Do() (*AutoscalerList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/autoscalers")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *AutoscalerList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of autoscaler resources contained within the specified zone.",
	//   "httpMethod": "GET",
	//   "id": "compute.autoscalers.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/autoscalers",
	//   "response": {
	//     "$ref": "AutoscalerList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.autoscalers.patch":

type AutoscalersPatchCall struct {
	s           *Service
	project     string
	zone        string
	autoscaler  string
	autoscaler2 *Autoscaler
	opt_        map[string]interface{}
}

// Patch: Updates an autoscaler resource in the specified project using
// the data included in the request. This method supports patch
// semantics.
func (r *AutoscalersService) Patch(project string, zone string, autoscaler string, autoscaler2 *Autoscaler) *AutoscalersPatchCall {
	c := &AutoscalersPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.autoscaler = autoscaler
	c.autoscaler2 = autoscaler2
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AutoscalersPatchCall) Fields(s ...googleapi.Field) *AutoscalersPatchCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *AutoscalersPatchCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.autoscaler2)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("autoscaler", fmt.Sprintf("%v", c.autoscaler))
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/autoscalers")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an autoscaler resource in the specified project using the data included in the request. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "compute.autoscalers.patch",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "autoscaler"
	//   ],
	//   "parameters": {
	//     "autoscaler": {
	//       "description": "Name of the autoscaler resource to update.",
	//       "location": "query",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/autoscalers",
	//   "request": {
	//     "$ref": "Autoscaler"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.autoscalers.update":

type AutoscalersUpdateCall struct {
	s          *Service
	project    string
	zone       string
	autoscaler *Autoscaler
	opt_       map[string]interface{}
}

// Update: Updates an autoscaler resource in the specified project using
// the data included in the request.
func (r *AutoscalersService) Update(project string, zone string, autoscaler *Autoscaler) *AutoscalersUpdateCall {
	c := &AutoscalersUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.autoscaler = autoscaler
	return c
}

// Autoscaler sets the optional parameter "autoscaler": Name of the
// autoscaler resource to update.
func (c *AutoscalersUpdateCall) Autoscaler(autoscaler string) *AutoscalersUpdateCall {
	c.opt_["autoscaler"] = autoscaler
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AutoscalersUpdateCall) Fields(s ...googleapi.Field) *AutoscalersUpdateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *AutoscalersUpdateCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.autoscaler)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["autoscaler"]; ok {
		params.Set("autoscaler", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/autoscalers")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an autoscaler resource in the specified project using the data included in the request.",
	//   "httpMethod": "PUT",
	//   "id": "compute.autoscalers.update",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "autoscaler": {
	//       "description": "Name of the autoscaler resource to update.",
	//       "location": "query",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/autoscalers",
	//   "request": {
	//     "$ref": "Autoscaler"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.backendServices.delete":

type BackendServicesDeleteCall struct {
	s              *Service
	project        string
	backendService string
	opt_           map[string]interface{}
}

// Delete: Deletes the specified BackendService resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/backendServices/delete
func (r *BackendServicesService) Delete(project string, backendService string) *BackendServicesDeleteCall {
	c := &BackendServicesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.backendService = backendService
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BackendServicesDeleteCall) Fields(s ...googleapi.Field) *BackendServicesDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *BackendServicesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/backendServices/{backendService}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":        c.project,
		"backendService": c.backendService,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified BackendService resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.backendServices.delete",
	//   "parameterOrder": [
	//     "project",
	//     "backendService"
	//   ],
	//   "parameters": {
	//     "backendService": {
	//       "description": "Name of the BackendService resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/backendServices/{backendService}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.backendServices.get":

type BackendServicesGetCall struct {
	s              *Service
	project        string
	backendService string
	opt_           map[string]interface{}
}

// Get: Returns the specified BackendService resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/backendServices/get
func (r *BackendServicesService) Get(project string, backendService string) *BackendServicesGetCall {
	c := &BackendServicesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.backendService = backendService
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BackendServicesGetCall) Fields(s ...googleapi.Field) *BackendServicesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *BackendServicesGetCall) Do() (*BackendService, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/backendServices/{backendService}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":        c.project,
		"backendService": c.backendService,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *BackendService
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified BackendService resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.backendServices.get",
	//   "parameterOrder": [
	//     "project",
	//     "backendService"
	//   ],
	//   "parameters": {
	//     "backendService": {
	//       "description": "Name of the BackendService resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/backendServices/{backendService}",
	//   "response": {
	//     "$ref": "BackendService"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.backendServices.getHealth":

type BackendServicesGetHealthCall struct {
	s                      *Service
	project                string
	backendService         string
	resourcegroupreference *ResourceGroupReference
	opt_                   map[string]interface{}
}

// GetHealth: Gets the most recent health check results for this
// BackendService.
// For details, see https://cloud.google.com/compute/docs/reference/latest/backendServices/getHealth
func (r *BackendServicesService) GetHealth(project string, backendService string, resourcegroupreference *ResourceGroupReference) *BackendServicesGetHealthCall {
	c := &BackendServicesGetHealthCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.backendService = backendService
	c.resourcegroupreference = resourcegroupreference
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BackendServicesGetHealthCall) Fields(s ...googleapi.Field) *BackendServicesGetHealthCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *BackendServicesGetHealthCall) Do() (*BackendServiceGroupHealth, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.resourcegroupreference)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/backendServices/{backendService}/getHealth")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":        c.project,
		"backendService": c.backendService,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *BackendServiceGroupHealth
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the most recent health check results for this BackendService.",
	//   "httpMethod": "POST",
	//   "id": "compute.backendServices.getHealth",
	//   "parameterOrder": [
	//     "project",
	//     "backendService"
	//   ],
	//   "parameters": {
	//     "backendService": {
	//       "description": "Name of the BackendService resource to which the queried instance belongs.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/backendServices/{backendService}/getHealth",
	//   "request": {
	//     "$ref": "ResourceGroupReference"
	//   },
	//   "response": {
	//     "$ref": "BackendServiceGroupHealth"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.backendServices.insert":

type BackendServicesInsertCall struct {
	s              *Service
	project        string
	backendservice *BackendService
	opt_           map[string]interface{}
}

// Insert: Creates a BackendService resource in the specified project
// using the data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/backendServices/insert
func (r *BackendServicesService) Insert(project string, backendservice *BackendService) *BackendServicesInsertCall {
	c := &BackendServicesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.backendservice = backendservice
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BackendServicesInsertCall) Fields(s ...googleapi.Field) *BackendServicesInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *BackendServicesInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.backendservice)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/backendServices")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a BackendService resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.backendServices.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/backendServices",
	//   "request": {
	//     "$ref": "BackendService"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.backendServices.list":

type BackendServicesListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of BackendService resources available to the
// specified project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/backendServices/list
func (r *BackendServicesService) List(project string) *BackendServicesListCall {
	c := &BackendServicesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *BackendServicesListCall) Filter(filter string) *BackendServicesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *BackendServicesListCall) MaxResults(maxResults int64) *BackendServicesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *BackendServicesListCall) PageToken(pageToken string) *BackendServicesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BackendServicesListCall) Fields(s ...googleapi.Field) *BackendServicesListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *BackendServicesListCall) Do() (*BackendServiceList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/backendServices")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *BackendServiceList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of BackendService resources available to the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.backendServices.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/backendServices",
	//   "response": {
	//     "$ref": "BackendServiceList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.backendServices.patch":

type BackendServicesPatchCall struct {
	s              *Service
	project        string
	backendService string
	backendservice *BackendService
	opt_           map[string]interface{}
}

// Patch: Update the entire content of the BackendService resource. This
// method supports patch semantics.
// For details, see https://cloud.google.com/compute/docs/reference/latest/backendServices/patch
func (r *BackendServicesService) Patch(project string, backendService string, backendservice *BackendService) *BackendServicesPatchCall {
	c := &BackendServicesPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.backendService = backendService
	c.backendservice = backendservice
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BackendServicesPatchCall) Fields(s ...googleapi.Field) *BackendServicesPatchCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *BackendServicesPatchCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.backendservice)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/backendServices/{backendService}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":        c.project,
		"backendService": c.backendService,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update the entire content of the BackendService resource. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "compute.backendServices.patch",
	//   "parameterOrder": [
	//     "project",
	//     "backendService"
	//   ],
	//   "parameters": {
	//     "backendService": {
	//       "description": "Name of the BackendService resource to update.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/backendServices/{backendService}",
	//   "request": {
	//     "$ref": "BackendService"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.backendServices.update":

type BackendServicesUpdateCall struct {
	s              *Service
	project        string
	backendService string
	backendservice *BackendService
	opt_           map[string]interface{}
}

// Update: Update the entire content of the BackendService resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/backendServices/update
func (r *BackendServicesService) Update(project string, backendService string, backendservice *BackendService) *BackendServicesUpdateCall {
	c := &BackendServicesUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.backendService = backendService
	c.backendservice = backendservice
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BackendServicesUpdateCall) Fields(s ...googleapi.Field) *BackendServicesUpdateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *BackendServicesUpdateCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.backendservice)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/backendServices/{backendService}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":        c.project,
		"backendService": c.backendService,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update the entire content of the BackendService resource.",
	//   "httpMethod": "PUT",
	//   "id": "compute.backendServices.update",
	//   "parameterOrder": [
	//     "project",
	//     "backendService"
	//   ],
	//   "parameters": {
	//     "backendService": {
	//       "description": "Name of the BackendService resource to update.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/backendServices/{backendService}",
	//   "request": {
	//     "$ref": "BackendService"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.diskTypes.aggregatedList":

type DiskTypesAggregatedListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// AggregatedList: Retrieves the list of disk type resources grouped by
// scope.
// For details, see https://cloud.google.com/compute/docs/reference/latest/diskTypes/aggregatedList
func (r *DiskTypesService) AggregatedList(project string) *DiskTypesAggregatedListCall {
	c := &DiskTypesAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *DiskTypesAggregatedListCall) Filter(filter string) *DiskTypesAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *DiskTypesAggregatedListCall) MaxResults(maxResults int64) *DiskTypesAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *DiskTypesAggregatedListCall) PageToken(pageToken string) *DiskTypesAggregatedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DiskTypesAggregatedListCall) Fields(s ...googleapi.Field) *DiskTypesAggregatedListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *DiskTypesAggregatedListCall) Do() (*DiskTypeAggregatedList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/aggregated/diskTypes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *DiskTypeAggregatedList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of disk type resources grouped by scope.",
	//   "httpMethod": "GET",
	//   "id": "compute.diskTypes.aggregatedList",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/aggregated/diskTypes",
	//   "response": {
	//     "$ref": "DiskTypeAggregatedList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.diskTypes.get":

type DiskTypesGetCall struct {
	s        *Service
	project  string
	zone     string
	diskType string
	opt_     map[string]interface{}
}

// Get: Returns the specified disk type resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/diskTypes/get
func (r *DiskTypesService) Get(project string, zone string, diskType string) *DiskTypesGetCall {
	c := &DiskTypesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.diskType = diskType
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DiskTypesGetCall) Fields(s ...googleapi.Field) *DiskTypesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *DiskTypesGetCall) Do() (*DiskType, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/diskTypes/{diskType}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"diskType": c.diskType,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *DiskType
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified disk type resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.diskTypes.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "diskType"
	//   ],
	//   "parameters": {
	//     "diskType": {
	//       "description": "Name of the disk type resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/diskTypes/{diskType}",
	//   "response": {
	//     "$ref": "DiskType"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.diskTypes.list":

type DiskTypesListCall struct {
	s       *Service
	project string
	zone    string
	opt_    map[string]interface{}
}

// List: Retrieves the list of disk type resources available to the
// specified project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/diskTypes/list
func (r *DiskTypesService) List(project string, zone string) *DiskTypesListCall {
	c := &DiskTypesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *DiskTypesListCall) Filter(filter string) *DiskTypesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *DiskTypesListCall) MaxResults(maxResults int64) *DiskTypesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *DiskTypesListCall) PageToken(pageToken string) *DiskTypesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DiskTypesListCall) Fields(s ...googleapi.Field) *DiskTypesListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *DiskTypesListCall) Do() (*DiskTypeList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/diskTypes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *DiskTypeList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of disk type resources available to the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.diskTypes.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/diskTypes",
	//   "response": {
	//     "$ref": "DiskTypeList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.disks.aggregatedList":

type DisksAggregatedListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// AggregatedList: Retrieves the list of disks grouped by scope.
// For details, see https://cloud.google.com/compute/docs/reference/latest/disks/aggregatedList
func (r *DisksService) AggregatedList(project string) *DisksAggregatedListCall {
	c := &DisksAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *DisksAggregatedListCall) Filter(filter string) *DisksAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *DisksAggregatedListCall) MaxResults(maxResults int64) *DisksAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *DisksAggregatedListCall) PageToken(pageToken string) *DisksAggregatedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DisksAggregatedListCall) Fields(s ...googleapi.Field) *DisksAggregatedListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *DisksAggregatedListCall) Do() (*DiskAggregatedList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/aggregated/disks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *DiskAggregatedList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of disks grouped by scope.",
	//   "httpMethod": "GET",
	//   "id": "compute.disks.aggregatedList",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/aggregated/disks",
	//   "response": {
	//     "$ref": "DiskAggregatedList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.disks.createSnapshot":

type DisksCreateSnapshotCall struct {
	s        *Service
	project  string
	zone     string
	disk     string
	snapshot *Snapshot
	opt_     map[string]interface{}
}

// CreateSnapshot: Creates a snapshot of this disk.
// For details, see https://cloud.google.com/compute/docs/reference/latest/disks/createSnapshot
func (r *DisksService) CreateSnapshot(project string, zone string, disk string, snapshot *Snapshot) *DisksCreateSnapshotCall {
	c := &DisksCreateSnapshotCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.disk = disk
	c.snapshot = snapshot
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DisksCreateSnapshotCall) Fields(s ...googleapi.Field) *DisksCreateSnapshotCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *DisksCreateSnapshotCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.snapshot)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/disks/{disk}/createSnapshot")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
		"disk":    c.disk,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a snapshot of this disk.",
	//   "httpMethod": "POST",
	//   "id": "compute.disks.createSnapshot",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "disk"
	//   ],
	//   "parameters": {
	//     "disk": {
	//       "description": "Name of the persistent disk to snapshot.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/disks/{disk}/createSnapshot",
	//   "request": {
	//     "$ref": "Snapshot"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.disks.delete":

type DisksDeleteCall struct {
	s       *Service
	project string
	zone    string
	disk    string
	opt_    map[string]interface{}
}

// Delete: Deletes the specified persistent disk. Deleting a disk
// removes its data permanently and is irreversible. However, deleting a
// disk does not delete any snapshots previously made from the disk. You
// must separately delete snapshots.
// For details, see https://cloud.google.com/compute/docs/reference/latest/disks/delete
func (r *DisksService) Delete(project string, zone string, disk string) *DisksDeleteCall {
	c := &DisksDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.disk = disk
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DisksDeleteCall) Fields(s ...googleapi.Field) *DisksDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *DisksDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/disks/{disk}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
		"disk":    c.disk,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified persistent disk. Deleting a disk removes its data permanently and is irreversible. However, deleting a disk does not delete any snapshots previously made from the disk. You must separately delete snapshots.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.disks.delete",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "disk"
	//   ],
	//   "parameters": {
	//     "disk": {
	//       "description": "Name of the persistent disk to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/disks/{disk}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.disks.get":

type DisksGetCall struct {
	s       *Service
	project string
	zone    string
	disk    string
	opt_    map[string]interface{}
}

// Get: Returns a specified persistent disk.
// For details, see https://cloud.google.com/compute/docs/reference/latest/disks/get
func (r *DisksService) Get(project string, zone string, disk string) *DisksGetCall {
	c := &DisksGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.disk = disk
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DisksGetCall) Fields(s ...googleapi.Field) *DisksGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *DisksGetCall) Do() (*Disk, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/disks/{disk}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
		"disk":    c.disk,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Disk
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a specified persistent disk.",
	//   "httpMethod": "GET",
	//   "id": "compute.disks.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "disk"
	//   ],
	//   "parameters": {
	//     "disk": {
	//       "description": "Name of the persistent disk to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/disks/{disk}",
	//   "response": {
	//     "$ref": "Disk"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.disks.insert":

type DisksInsertCall struct {
	s       *Service
	project string
	zone    string
	disk    *Disk
	opt_    map[string]interface{}
}

// Insert: Creates a persistent disk in the specified project using the
// data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/disks/insert
func (r *DisksService) Insert(project string, zone string, disk *Disk) *DisksInsertCall {
	c := &DisksInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.disk = disk
	return c
}

// SourceImage sets the optional parameter "sourceImage": Source image
// to restore onto a disk.
func (c *DisksInsertCall) SourceImage(sourceImage string) *DisksInsertCall {
	c.opt_["sourceImage"] = sourceImage
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DisksInsertCall) Fields(s ...googleapi.Field) *DisksInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *DisksInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.disk)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["sourceImage"]; ok {
		params.Set("sourceImage", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/disks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a persistent disk in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.disks.insert",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sourceImage": {
	//       "description": "Optional. Source image to restore onto a disk.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/disks",
	//   "request": {
	//     "$ref": "Disk"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.disks.list":

type DisksListCall struct {
	s       *Service
	project string
	zone    string
	opt_    map[string]interface{}
}

// List: Retrieves the list of persistent disks contained within the
// specified zone.
// For details, see https://cloud.google.com/compute/docs/reference/latest/disks/list
func (r *DisksService) List(project string, zone string) *DisksListCall {
	c := &DisksListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *DisksListCall) Filter(filter string) *DisksListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *DisksListCall) MaxResults(maxResults int64) *DisksListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *DisksListCall) PageToken(pageToken string) *DisksListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DisksListCall) Fields(s ...googleapi.Field) *DisksListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *DisksListCall) Do() (*DiskList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/disks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *DiskList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of persistent disks contained within the specified zone.",
	//   "httpMethod": "GET",
	//   "id": "compute.disks.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/disks",
	//   "response": {
	//     "$ref": "DiskList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.firewalls.delete":

type FirewallsDeleteCall struct {
	s        *Service
	project  string
	firewall string
	opt_     map[string]interface{}
}

// Delete: Deletes the specified firewall resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/firewalls/delete
func (r *FirewallsService) Delete(project string, firewall string) *FirewallsDeleteCall {
	c := &FirewallsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.firewall = firewall
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FirewallsDeleteCall) Fields(s ...googleapi.Field) *FirewallsDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *FirewallsDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/firewalls/{firewall}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"firewall": c.firewall,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified firewall resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.firewalls.delete",
	//   "parameterOrder": [
	//     "project",
	//     "firewall"
	//   ],
	//   "parameters": {
	//     "firewall": {
	//       "description": "Name of the firewall resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/firewalls/{firewall}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.firewalls.get":

type FirewallsGetCall struct {
	s        *Service
	project  string
	firewall string
	opt_     map[string]interface{}
}

// Get: Returns the specified firewall resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/firewalls/get
func (r *FirewallsService) Get(project string, firewall string) *FirewallsGetCall {
	c := &FirewallsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.firewall = firewall
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FirewallsGetCall) Fields(s ...googleapi.Field) *FirewallsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *FirewallsGetCall) Do() (*Firewall, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/firewalls/{firewall}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"firewall": c.firewall,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Firewall
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified firewall resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.firewalls.get",
	//   "parameterOrder": [
	//     "project",
	//     "firewall"
	//   ],
	//   "parameters": {
	//     "firewall": {
	//       "description": "Name of the firewall resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/firewalls/{firewall}",
	//   "response": {
	//     "$ref": "Firewall"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.firewalls.insert":

type FirewallsInsertCall struct {
	s        *Service
	project  string
	firewall *Firewall
	opt_     map[string]interface{}
}

// Insert: Creates a firewall resource in the specified project using
// the data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/firewalls/insert
func (r *FirewallsService) Insert(project string, firewall *Firewall) *FirewallsInsertCall {
	c := &FirewallsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.firewall = firewall
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FirewallsInsertCall) Fields(s ...googleapi.Field) *FirewallsInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *FirewallsInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.firewall)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/firewalls")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a firewall resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.firewalls.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/firewalls",
	//   "request": {
	//     "$ref": "Firewall"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.firewalls.list":

type FirewallsListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of firewall resources available to the
// specified project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/firewalls/list
func (r *FirewallsService) List(project string) *FirewallsListCall {
	c := &FirewallsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *FirewallsListCall) Filter(filter string) *FirewallsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *FirewallsListCall) MaxResults(maxResults int64) *FirewallsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *FirewallsListCall) PageToken(pageToken string) *FirewallsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FirewallsListCall) Fields(s ...googleapi.Field) *FirewallsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *FirewallsListCall) Do() (*FirewallList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/firewalls")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *FirewallList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of firewall resources available to the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.firewalls.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/firewalls",
	//   "response": {
	//     "$ref": "FirewallList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.firewalls.patch":

type FirewallsPatchCall struct {
	s         *Service
	project   string
	firewall  string
	firewall2 *Firewall
	opt_      map[string]interface{}
}

// Patch: Updates the specified firewall resource with the data included
// in the request. This method supports patch semantics.
// For details, see https://cloud.google.com/compute/docs/reference/latest/firewalls/patch
func (r *FirewallsService) Patch(project string, firewall string, firewall2 *Firewall) *FirewallsPatchCall {
	c := &FirewallsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.firewall = firewall
	c.firewall2 = firewall2
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FirewallsPatchCall) Fields(s ...googleapi.Field) *FirewallsPatchCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *FirewallsPatchCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.firewall2)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/firewalls/{firewall}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"firewall": c.firewall,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the specified firewall resource with the data included in the request. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "compute.firewalls.patch",
	//   "parameterOrder": [
	//     "project",
	//     "firewall"
	//   ],
	//   "parameters": {
	//     "firewall": {
	//       "description": "Name of the firewall resource to update.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/firewalls/{firewall}",
	//   "request": {
	//     "$ref": "Firewall"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.firewalls.update":

type FirewallsUpdateCall struct {
	s         *Service
	project   string
	firewall  string
	firewall2 *Firewall
	opt_      map[string]interface{}
}

// Update: Updates the specified firewall resource with the data
// included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/firewalls/update
func (r *FirewallsService) Update(project string, firewall string, firewall2 *Firewall) *FirewallsUpdateCall {
	c := &FirewallsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.firewall = firewall
	c.firewall2 = firewall2
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FirewallsUpdateCall) Fields(s ...googleapi.Field) *FirewallsUpdateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *FirewallsUpdateCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.firewall2)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/firewalls/{firewall}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"firewall": c.firewall,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the specified firewall resource with the data included in the request.",
	//   "httpMethod": "PUT",
	//   "id": "compute.firewalls.update",
	//   "parameterOrder": [
	//     "project",
	//     "firewall"
	//   ],
	//   "parameters": {
	//     "firewall": {
	//       "description": "Name of the firewall resource to update.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/firewalls/{firewall}",
	//   "request": {
	//     "$ref": "Firewall"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.forwardingRules.aggregatedList":

type ForwardingRulesAggregatedListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// AggregatedList: Retrieves the list of forwarding rules grouped by
// scope.
// For details, see https://cloud.google.com/compute/docs/reference/latest/forwardingRules/aggregatedList
func (r *ForwardingRulesService) AggregatedList(project string) *ForwardingRulesAggregatedListCall {
	c := &ForwardingRulesAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *ForwardingRulesAggregatedListCall) Filter(filter string) *ForwardingRulesAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *ForwardingRulesAggregatedListCall) MaxResults(maxResults int64) *ForwardingRulesAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *ForwardingRulesAggregatedListCall) PageToken(pageToken string) *ForwardingRulesAggregatedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ForwardingRulesAggregatedListCall) Fields(s ...googleapi.Field) *ForwardingRulesAggregatedListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ForwardingRulesAggregatedListCall) Do() (*ForwardingRuleAggregatedList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/aggregated/forwardingRules")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ForwardingRuleAggregatedList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of forwarding rules grouped by scope.",
	//   "httpMethod": "GET",
	//   "id": "compute.forwardingRules.aggregatedList",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/aggregated/forwardingRules",
	//   "response": {
	//     "$ref": "ForwardingRuleAggregatedList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.forwardingRules.delete":

type ForwardingRulesDeleteCall struct {
	s              *Service
	project        string
	region         string
	forwardingRule string
	opt_           map[string]interface{}
}

// Delete: Deletes the specified ForwardingRule resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/forwardingRules/delete
func (r *ForwardingRulesService) Delete(project string, region string, forwardingRule string) *ForwardingRulesDeleteCall {
	c := &ForwardingRulesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.forwardingRule = forwardingRule
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ForwardingRulesDeleteCall) Fields(s ...googleapi.Field) *ForwardingRulesDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ForwardingRulesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/forwardingRules/{forwardingRule}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":        c.project,
		"region":         c.region,
		"forwardingRule": c.forwardingRule,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified ForwardingRule resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.forwardingRules.delete",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "forwardingRule"
	//   ],
	//   "parameters": {
	//     "forwardingRule": {
	//       "description": "Name of the ForwardingRule resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/forwardingRules/{forwardingRule}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.forwardingRules.get":

type ForwardingRulesGetCall struct {
	s              *Service
	project        string
	region         string
	forwardingRule string
	opt_           map[string]interface{}
}

// Get: Returns the specified ForwardingRule resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/forwardingRules/get
func (r *ForwardingRulesService) Get(project string, region string, forwardingRule string) *ForwardingRulesGetCall {
	c := &ForwardingRulesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.forwardingRule = forwardingRule
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ForwardingRulesGetCall) Fields(s ...googleapi.Field) *ForwardingRulesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ForwardingRulesGetCall) Do() (*ForwardingRule, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/forwardingRules/{forwardingRule}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":        c.project,
		"region":         c.region,
		"forwardingRule": c.forwardingRule,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ForwardingRule
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified ForwardingRule resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.forwardingRules.get",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "forwardingRule"
	//   ],
	//   "parameters": {
	//     "forwardingRule": {
	//       "description": "Name of the ForwardingRule resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/forwardingRules/{forwardingRule}",
	//   "response": {
	//     "$ref": "ForwardingRule"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.forwardingRules.insert":

type ForwardingRulesInsertCall struct {
	s              *Service
	project        string
	region         string
	forwardingrule *ForwardingRule
	opt_           map[string]interface{}
}

// Insert: Creates a ForwardingRule resource in the specified project
// and region using the data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/forwardingRules/insert
func (r *ForwardingRulesService) Insert(project string, region string, forwardingrule *ForwardingRule) *ForwardingRulesInsertCall {
	c := &ForwardingRulesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.forwardingrule = forwardingrule
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ForwardingRulesInsertCall) Fields(s ...googleapi.Field) *ForwardingRulesInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ForwardingRulesInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.forwardingrule)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/forwardingRules")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"region":  c.region,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a ForwardingRule resource in the specified project and region using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.forwardingRules.insert",
	//   "parameterOrder": [
	//     "project",
	//     "region"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/forwardingRules",
	//   "request": {
	//     "$ref": "ForwardingRule"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.forwardingRules.list":

type ForwardingRulesListCall struct {
	s       *Service
	project string
	region  string
	opt_    map[string]interface{}
}

// List: Retrieves the list of ForwardingRule resources available to the
// specified project and region.
// For details, see https://cloud.google.com/compute/docs/reference/latest/forwardingRules/list
func (r *ForwardingRulesService) List(project string, region string) *ForwardingRulesListCall {
	c := &ForwardingRulesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *ForwardingRulesListCall) Filter(filter string) *ForwardingRulesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *ForwardingRulesListCall) MaxResults(maxResults int64) *ForwardingRulesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *ForwardingRulesListCall) PageToken(pageToken string) *ForwardingRulesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ForwardingRulesListCall) Fields(s ...googleapi.Field) *ForwardingRulesListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ForwardingRulesListCall) Do() (*ForwardingRuleList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/forwardingRules")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"region":  c.region,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ForwardingRuleList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of ForwardingRule resources available to the specified project and region.",
	//   "httpMethod": "GET",
	//   "id": "compute.forwardingRules.list",
	//   "parameterOrder": [
	//     "project",
	//     "region"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/forwardingRules",
	//   "response": {
	//     "$ref": "ForwardingRuleList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.forwardingRules.setTarget":

type ForwardingRulesSetTargetCall struct {
	s               *Service
	project         string
	region          string
	forwardingRule  string
	targetreference *TargetReference
	opt_            map[string]interface{}
}

// SetTarget: Changes target url for forwarding rule.
// For details, see https://cloud.google.com/compute/docs/reference/latest/forwardingRules/setTarget
func (r *ForwardingRulesService) SetTarget(project string, region string, forwardingRule string, targetreference *TargetReference) *ForwardingRulesSetTargetCall {
	c := &ForwardingRulesSetTargetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.forwardingRule = forwardingRule
	c.targetreference = targetreference
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ForwardingRulesSetTargetCall) Fields(s ...googleapi.Field) *ForwardingRulesSetTargetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ForwardingRulesSetTargetCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.targetreference)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/forwardingRules/{forwardingRule}/setTarget")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":        c.project,
		"region":         c.region,
		"forwardingRule": c.forwardingRule,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Changes target url for forwarding rule.",
	//   "httpMethod": "POST",
	//   "id": "compute.forwardingRules.setTarget",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "forwardingRule"
	//   ],
	//   "parameters": {
	//     "forwardingRule": {
	//       "description": "Name of the ForwardingRule resource in which target is to be set.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/forwardingRules/{forwardingRule}/setTarget",
	//   "request": {
	//     "$ref": "TargetReference"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.globalAddresses.delete":

type GlobalAddressesDeleteCall struct {
	s       *Service
	project string
	address string
	opt_    map[string]interface{}
}

// Delete: Deletes the specified address resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/globalAddresses/delete
func (r *GlobalAddressesService) Delete(project string, address string) *GlobalAddressesDeleteCall {
	c := &GlobalAddressesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.address = address
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalAddressesDeleteCall) Fields(s ...googleapi.Field) *GlobalAddressesDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *GlobalAddressesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/addresses/{address}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"address": c.address,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified address resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.globalAddresses.delete",
	//   "parameterOrder": [
	//     "project",
	//     "address"
	//   ],
	//   "parameters": {
	//     "address": {
	//       "description": "Name of the address resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/addresses/{address}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.globalAddresses.get":

type GlobalAddressesGetCall struct {
	s       *Service
	project string
	address string
	opt_    map[string]interface{}
}

// Get: Returns the specified address resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/globalAddresses/get
func (r *GlobalAddressesService) Get(project string, address string) *GlobalAddressesGetCall {
	c := &GlobalAddressesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.address = address
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalAddressesGetCall) Fields(s ...googleapi.Field) *GlobalAddressesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *GlobalAddressesGetCall) Do() (*Address, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/addresses/{address}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"address": c.address,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Address
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified address resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.globalAddresses.get",
	//   "parameterOrder": [
	//     "project",
	//     "address"
	//   ],
	//   "parameters": {
	//     "address": {
	//       "description": "Name of the address resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/addresses/{address}",
	//   "response": {
	//     "$ref": "Address"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.globalAddresses.insert":

type GlobalAddressesInsertCall struct {
	s       *Service
	project string
	address *Address
	opt_    map[string]interface{}
}

// Insert: Creates an address resource in the specified project using
// the data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/globalAddresses/insert
func (r *GlobalAddressesService) Insert(project string, address *Address) *GlobalAddressesInsertCall {
	c := &GlobalAddressesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.address = address
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalAddressesInsertCall) Fields(s ...googleapi.Field) *GlobalAddressesInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *GlobalAddressesInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.address)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/addresses")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates an address resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.globalAddresses.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/addresses",
	//   "request": {
	//     "$ref": "Address"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.globalAddresses.list":

type GlobalAddressesListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of global address resources.
// For details, see https://cloud.google.com/compute/docs/reference/latest/globalAddresses/list
func (r *GlobalAddressesService) List(project string) *GlobalAddressesListCall {
	c := &GlobalAddressesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *GlobalAddressesListCall) Filter(filter string) *GlobalAddressesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *GlobalAddressesListCall) MaxResults(maxResults int64) *GlobalAddressesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *GlobalAddressesListCall) PageToken(pageToken string) *GlobalAddressesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalAddressesListCall) Fields(s ...googleapi.Field) *GlobalAddressesListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *GlobalAddressesListCall) Do() (*AddressList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/addresses")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *AddressList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of global address resources.",
	//   "httpMethod": "GET",
	//   "id": "compute.globalAddresses.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/addresses",
	//   "response": {
	//     "$ref": "AddressList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.globalForwardingRules.delete":

type GlobalForwardingRulesDeleteCall struct {
	s              *Service
	project        string
	forwardingRule string
	opt_           map[string]interface{}
}

// Delete: Deletes the specified ForwardingRule resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/globalForwardingRules/delete
func (r *GlobalForwardingRulesService) Delete(project string, forwardingRule string) *GlobalForwardingRulesDeleteCall {
	c := &GlobalForwardingRulesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.forwardingRule = forwardingRule
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalForwardingRulesDeleteCall) Fields(s ...googleapi.Field) *GlobalForwardingRulesDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *GlobalForwardingRulesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/forwardingRules/{forwardingRule}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":        c.project,
		"forwardingRule": c.forwardingRule,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified ForwardingRule resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.globalForwardingRules.delete",
	//   "parameterOrder": [
	//     "project",
	//     "forwardingRule"
	//   ],
	//   "parameters": {
	//     "forwardingRule": {
	//       "description": "Name of the ForwardingRule resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/forwardingRules/{forwardingRule}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.globalForwardingRules.get":

type GlobalForwardingRulesGetCall struct {
	s              *Service
	project        string
	forwardingRule string
	opt_           map[string]interface{}
}

// Get: Returns the specified ForwardingRule resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/globalForwardingRules/get
func (r *GlobalForwardingRulesService) Get(project string, forwardingRule string) *GlobalForwardingRulesGetCall {
	c := &GlobalForwardingRulesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.forwardingRule = forwardingRule
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalForwardingRulesGetCall) Fields(s ...googleapi.Field) *GlobalForwardingRulesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *GlobalForwardingRulesGetCall) Do() (*ForwardingRule, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/forwardingRules/{forwardingRule}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":        c.project,
		"forwardingRule": c.forwardingRule,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ForwardingRule
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified ForwardingRule resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.globalForwardingRules.get",
	//   "parameterOrder": [
	//     "project",
	//     "forwardingRule"
	//   ],
	//   "parameters": {
	//     "forwardingRule": {
	//       "description": "Name of the ForwardingRule resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/forwardingRules/{forwardingRule}",
	//   "response": {
	//     "$ref": "ForwardingRule"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.globalForwardingRules.insert":

type GlobalForwardingRulesInsertCall struct {
	s              *Service
	project        string
	forwardingrule *ForwardingRule
	opt_           map[string]interface{}
}

// Insert: Creates a ForwardingRule resource in the specified project
// and region using the data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/globalForwardingRules/insert
func (r *GlobalForwardingRulesService) Insert(project string, forwardingrule *ForwardingRule) *GlobalForwardingRulesInsertCall {
	c := &GlobalForwardingRulesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.forwardingrule = forwardingrule
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalForwardingRulesInsertCall) Fields(s ...googleapi.Field) *GlobalForwardingRulesInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *GlobalForwardingRulesInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.forwardingrule)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/forwardingRules")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a ForwardingRule resource in the specified project and region using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.globalForwardingRules.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/forwardingRules",
	//   "request": {
	//     "$ref": "ForwardingRule"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.globalForwardingRules.list":

type GlobalForwardingRulesListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of ForwardingRule resources available to the
// specified project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/globalForwardingRules/list
func (r *GlobalForwardingRulesService) List(project string) *GlobalForwardingRulesListCall {
	c := &GlobalForwardingRulesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *GlobalForwardingRulesListCall) Filter(filter string) *GlobalForwardingRulesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *GlobalForwardingRulesListCall) MaxResults(maxResults int64) *GlobalForwardingRulesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *GlobalForwardingRulesListCall) PageToken(pageToken string) *GlobalForwardingRulesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalForwardingRulesListCall) Fields(s ...googleapi.Field) *GlobalForwardingRulesListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *GlobalForwardingRulesListCall) Do() (*ForwardingRuleList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/forwardingRules")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ForwardingRuleList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of ForwardingRule resources available to the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.globalForwardingRules.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/forwardingRules",
	//   "response": {
	//     "$ref": "ForwardingRuleList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.globalForwardingRules.setTarget":

type GlobalForwardingRulesSetTargetCall struct {
	s               *Service
	project         string
	forwardingRule  string
	targetreference *TargetReference
	opt_            map[string]interface{}
}

// SetTarget: Changes target url for forwarding rule.
// For details, see https://cloud.google.com/compute/docs/reference/latest/globalForwardingRules/setTarget
func (r *GlobalForwardingRulesService) SetTarget(project string, forwardingRule string, targetreference *TargetReference) *GlobalForwardingRulesSetTargetCall {
	c := &GlobalForwardingRulesSetTargetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.forwardingRule = forwardingRule
	c.targetreference = targetreference
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalForwardingRulesSetTargetCall) Fields(s ...googleapi.Field) *GlobalForwardingRulesSetTargetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *GlobalForwardingRulesSetTargetCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.targetreference)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/forwardingRules/{forwardingRule}/setTarget")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":        c.project,
		"forwardingRule": c.forwardingRule,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Changes target url for forwarding rule.",
	//   "httpMethod": "POST",
	//   "id": "compute.globalForwardingRules.setTarget",
	//   "parameterOrder": [
	//     "project",
	//     "forwardingRule"
	//   ],
	//   "parameters": {
	//     "forwardingRule": {
	//       "description": "Name of the ForwardingRule resource in which target is to be set.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/forwardingRules/{forwardingRule}/setTarget",
	//   "request": {
	//     "$ref": "TargetReference"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.globalOperations.aggregatedList":

type GlobalOperationsAggregatedListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// AggregatedList: Retrieves the list of all operations grouped by
// scope.
// For details, see https://cloud.google.com/compute/docs/reference/latest/globalOperations/aggregatedList
func (r *GlobalOperationsService) AggregatedList(project string) *GlobalOperationsAggregatedListCall {
	c := &GlobalOperationsAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *GlobalOperationsAggregatedListCall) Filter(filter string) *GlobalOperationsAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *GlobalOperationsAggregatedListCall) MaxResults(maxResults int64) *GlobalOperationsAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *GlobalOperationsAggregatedListCall) PageToken(pageToken string) *GlobalOperationsAggregatedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalOperationsAggregatedListCall) Fields(s ...googleapi.Field) *GlobalOperationsAggregatedListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *GlobalOperationsAggregatedListCall) Do() (*OperationAggregatedList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/aggregated/operations")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *OperationAggregatedList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of all operations grouped by scope.",
	//   "httpMethod": "GET",
	//   "id": "compute.globalOperations.aggregatedList",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/aggregated/operations",
	//   "response": {
	//     "$ref": "OperationAggregatedList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.globalOperations.delete":

type GlobalOperationsDeleteCall struct {
	s         *Service
	project   string
	operation string
	opt_      map[string]interface{}
}

// Delete: Deletes the specified Operations resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/globalOperations/delete
func (r *GlobalOperationsService) Delete(project string, operation string) *GlobalOperationsDeleteCall {
	c := &GlobalOperationsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.operation = operation
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalOperationsDeleteCall) Fields(s ...googleapi.Field) *GlobalOperationsDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *GlobalOperationsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/operations/{operation}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"operation": c.operation,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the specified Operations resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.globalOperations.delete",
	//   "parameterOrder": [
	//     "project",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the Operations resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/operations/{operation}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.globalOperations.get":

type GlobalOperationsGetCall struct {
	s         *Service
	project   string
	operation string
	opt_      map[string]interface{}
}

// Get: Retrieves the specified Operations resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/globalOperations/get
func (r *GlobalOperationsService) Get(project string, operation string) *GlobalOperationsGetCall {
	c := &GlobalOperationsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.operation = operation
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalOperationsGetCall) Fields(s ...googleapi.Field) *GlobalOperationsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *GlobalOperationsGetCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/operations/{operation}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"operation": c.operation,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the specified Operations resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.globalOperations.get",
	//   "parameterOrder": [
	//     "project",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the Operations resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/operations/{operation}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.globalOperations.list":

type GlobalOperationsListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of Operation resources contained within the
// specified project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/globalOperations/list
func (r *GlobalOperationsService) List(project string) *GlobalOperationsListCall {
	c := &GlobalOperationsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *GlobalOperationsListCall) Filter(filter string) *GlobalOperationsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *GlobalOperationsListCall) MaxResults(maxResults int64) *GlobalOperationsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *GlobalOperationsListCall) PageToken(pageToken string) *GlobalOperationsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalOperationsListCall) Fields(s ...googleapi.Field) *GlobalOperationsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *GlobalOperationsListCall) Do() (*OperationList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/operations")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *OperationList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of Operation resources contained within the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.globalOperations.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/operations",
	//   "response": {
	//     "$ref": "OperationList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.httpHealthChecks.delete":

type HttpHealthChecksDeleteCall struct {
	s               *Service
	project         string
	httpHealthCheck string
	opt_            map[string]interface{}
}

// Delete: Deletes the specified HttpHealthCheck resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/httpHealthChecks/delete
func (r *HttpHealthChecksService) Delete(project string, httpHealthCheck string) *HttpHealthChecksDeleteCall {
	c := &HttpHealthChecksDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.httpHealthCheck = httpHealthCheck
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *HttpHealthChecksDeleteCall) Fields(s ...googleapi.Field) *HttpHealthChecksDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *HttpHealthChecksDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/httpHealthChecks/{httpHealthCheck}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":         c.project,
		"httpHealthCheck": c.httpHealthCheck,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified HttpHealthCheck resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.httpHealthChecks.delete",
	//   "parameterOrder": [
	//     "project",
	//     "httpHealthCheck"
	//   ],
	//   "parameters": {
	//     "httpHealthCheck": {
	//       "description": "Name of the HttpHealthCheck resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/httpHealthChecks/{httpHealthCheck}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.httpHealthChecks.get":

type HttpHealthChecksGetCall struct {
	s               *Service
	project         string
	httpHealthCheck string
	opt_            map[string]interface{}
}

// Get: Returns the specified HttpHealthCheck resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/httpHealthChecks/get
func (r *HttpHealthChecksService) Get(project string, httpHealthCheck string) *HttpHealthChecksGetCall {
	c := &HttpHealthChecksGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.httpHealthCheck = httpHealthCheck
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *HttpHealthChecksGetCall) Fields(s ...googleapi.Field) *HttpHealthChecksGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *HttpHealthChecksGetCall) Do() (*HttpHealthCheck, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/httpHealthChecks/{httpHealthCheck}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":         c.project,
		"httpHealthCheck": c.httpHealthCheck,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *HttpHealthCheck
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified HttpHealthCheck resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.httpHealthChecks.get",
	//   "parameterOrder": [
	//     "project",
	//     "httpHealthCheck"
	//   ],
	//   "parameters": {
	//     "httpHealthCheck": {
	//       "description": "Name of the HttpHealthCheck resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/httpHealthChecks/{httpHealthCheck}",
	//   "response": {
	//     "$ref": "HttpHealthCheck"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.httpHealthChecks.insert":

type HttpHealthChecksInsertCall struct {
	s               *Service
	project         string
	httphealthcheck *HttpHealthCheck
	opt_            map[string]interface{}
}

// Insert: Creates a HttpHealthCheck resource in the specified project
// using the data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/httpHealthChecks/insert
func (r *HttpHealthChecksService) Insert(project string, httphealthcheck *HttpHealthCheck) *HttpHealthChecksInsertCall {
	c := &HttpHealthChecksInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.httphealthcheck = httphealthcheck
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *HttpHealthChecksInsertCall) Fields(s ...googleapi.Field) *HttpHealthChecksInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *HttpHealthChecksInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.httphealthcheck)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/httpHealthChecks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a HttpHealthCheck resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.httpHealthChecks.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/httpHealthChecks",
	//   "request": {
	//     "$ref": "HttpHealthCheck"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.httpHealthChecks.list":

type HttpHealthChecksListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of HttpHealthCheck resources available to
// the specified project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/httpHealthChecks/list
func (r *HttpHealthChecksService) List(project string) *HttpHealthChecksListCall {
	c := &HttpHealthChecksListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *HttpHealthChecksListCall) Filter(filter string) *HttpHealthChecksListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *HttpHealthChecksListCall) MaxResults(maxResults int64) *HttpHealthChecksListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *HttpHealthChecksListCall) PageToken(pageToken string) *HttpHealthChecksListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *HttpHealthChecksListCall) Fields(s ...googleapi.Field) *HttpHealthChecksListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *HttpHealthChecksListCall) Do() (*HttpHealthCheckList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/httpHealthChecks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *HttpHealthCheckList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of HttpHealthCheck resources available to the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.httpHealthChecks.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/httpHealthChecks",
	//   "response": {
	//     "$ref": "HttpHealthCheckList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.httpHealthChecks.patch":

type HttpHealthChecksPatchCall struct {
	s               *Service
	project         string
	httpHealthCheck string
	httphealthcheck *HttpHealthCheck
	opt_            map[string]interface{}
}

// Patch: Updates a HttpHealthCheck resource in the specified project
// using the data included in the request. This method supports patch
// semantics.
// For details, see https://cloud.google.com/compute/docs/reference/latest/httpHealthChecks/patch
func (r *HttpHealthChecksService) Patch(project string, httpHealthCheck string, httphealthcheck *HttpHealthCheck) *HttpHealthChecksPatchCall {
	c := &HttpHealthChecksPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.httpHealthCheck = httpHealthCheck
	c.httphealthcheck = httphealthcheck
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *HttpHealthChecksPatchCall) Fields(s ...googleapi.Field) *HttpHealthChecksPatchCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *HttpHealthChecksPatchCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.httphealthcheck)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/httpHealthChecks/{httpHealthCheck}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":         c.project,
		"httpHealthCheck": c.httpHealthCheck,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a HttpHealthCheck resource in the specified project using the data included in the request. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "compute.httpHealthChecks.patch",
	//   "parameterOrder": [
	//     "project",
	//     "httpHealthCheck"
	//   ],
	//   "parameters": {
	//     "httpHealthCheck": {
	//       "description": "Name of the HttpHealthCheck resource to update.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/httpHealthChecks/{httpHealthCheck}",
	//   "request": {
	//     "$ref": "HttpHealthCheck"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.httpHealthChecks.update":

type HttpHealthChecksUpdateCall struct {
	s               *Service
	project         string
	httpHealthCheck string
	httphealthcheck *HttpHealthCheck
	opt_            map[string]interface{}
}

// Update: Updates a HttpHealthCheck resource in the specified project
// using the data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/httpHealthChecks/update
func (r *HttpHealthChecksService) Update(project string, httpHealthCheck string, httphealthcheck *HttpHealthCheck) *HttpHealthChecksUpdateCall {
	c := &HttpHealthChecksUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.httpHealthCheck = httpHealthCheck
	c.httphealthcheck = httphealthcheck
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *HttpHealthChecksUpdateCall) Fields(s ...googleapi.Field) *HttpHealthChecksUpdateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *HttpHealthChecksUpdateCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.httphealthcheck)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/httpHealthChecks/{httpHealthCheck}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":         c.project,
		"httpHealthCheck": c.httpHealthCheck,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a HttpHealthCheck resource in the specified project using the data included in the request.",
	//   "httpMethod": "PUT",
	//   "id": "compute.httpHealthChecks.update",
	//   "parameterOrder": [
	//     "project",
	//     "httpHealthCheck"
	//   ],
	//   "parameters": {
	//     "httpHealthCheck": {
	//       "description": "Name of the HttpHealthCheck resource to update.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/httpHealthChecks/{httpHealthCheck}",
	//   "request": {
	//     "$ref": "HttpHealthCheck"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.images.delete":

type ImagesDeleteCall struct {
	s       *Service
	project string
	image   string
	opt_    map[string]interface{}
}

// Delete: Deletes the specified image resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/images/delete
func (r *ImagesService) Delete(project string, image string) *ImagesDeleteCall {
	c := &ImagesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.image = image
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ImagesDeleteCall) Fields(s ...googleapi.Field) *ImagesDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ImagesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/images/{image}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"image":   c.image,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified image resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.images.delete",
	//   "parameterOrder": [
	//     "project",
	//     "image"
	//   ],
	//   "parameters": {
	//     "image": {
	//       "description": "Name of the image resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/images/{image}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.images.deprecate":

type ImagesDeprecateCall struct {
	s                 *Service
	project           string
	image             string
	deprecationstatus *DeprecationStatus
	opt_              map[string]interface{}
}

// Deprecate: Sets the deprecation status of an image.
//
// If an empty request body is given, clears the deprecation status
// instead.
// For details, see https://cloud.google.com/compute/docs/reference/latest/images/deprecate
func (r *ImagesService) Deprecate(project string, image string, deprecationstatus *DeprecationStatus) *ImagesDeprecateCall {
	c := &ImagesDeprecateCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.image = image
	c.deprecationstatus = deprecationstatus
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ImagesDeprecateCall) Fields(s ...googleapi.Field) *ImagesDeprecateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ImagesDeprecateCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.deprecationstatus)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/images/{image}/deprecate")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"image":   c.image,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sets the deprecation status of an image.\n\nIf an empty request body is given, clears the deprecation status instead.",
	//   "httpMethod": "POST",
	//   "id": "compute.images.deprecate",
	//   "parameterOrder": [
	//     "project",
	//     "image"
	//   ],
	//   "parameters": {
	//     "image": {
	//       "description": "Image name.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/images/{image}/deprecate",
	//   "request": {
	//     "$ref": "DeprecationStatus"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.images.get":

type ImagesGetCall struct {
	s       *Service
	project string
	image   string
	opt_    map[string]interface{}
}

// Get: Returns the specified image resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/images/get
func (r *ImagesService) Get(project string, image string) *ImagesGetCall {
	c := &ImagesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.image = image
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ImagesGetCall) Fields(s ...googleapi.Field) *ImagesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ImagesGetCall) Do() (*Image, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/images/{image}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"image":   c.image,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Image
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified image resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.images.get",
	//   "parameterOrder": [
	//     "project",
	//     "image"
	//   ],
	//   "parameters": {
	//     "image": {
	//       "description": "Name of the image resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/images/{image}",
	//   "response": {
	//     "$ref": "Image"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.images.insert":

type ImagesInsertCall struct {
	s       *Service
	project string
	image   *Image
	opt_    map[string]interface{}
}

// Insert: Creates an image resource in the specified project using the
// data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/images/insert
func (r *ImagesService) Insert(project string, image *Image) *ImagesInsertCall {
	c := &ImagesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.image = image
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ImagesInsertCall) Fields(s ...googleapi.Field) *ImagesInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ImagesInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.image)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/images")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates an image resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.images.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/images",
	//   "request": {
	//     "$ref": "Image"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_only",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ]
	// }

}

// method id "compute.images.list":

type ImagesListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of image resources available to the
// specified project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/images/list
func (r *ImagesService) List(project string) *ImagesListCall {
	c := &ImagesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *ImagesListCall) Filter(filter string) *ImagesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *ImagesListCall) MaxResults(maxResults int64) *ImagesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *ImagesListCall) PageToken(pageToken string) *ImagesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ImagesListCall) Fields(s ...googleapi.Field) *ImagesListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ImagesListCall) Do() (*ImageList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/images")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ImageList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of image resources available to the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.images.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/images",
	//   "response": {
	//     "$ref": "ImageList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instanceGroupManagers.abandonInstances":

type InstanceGroupManagersAbandonInstancesCall struct {
	s                                            *Service
	project                                      string
	zone                                         string
	instanceGroupManager                         string
	instancegroupmanagersabandoninstancesrequest *InstanceGroupManagersAbandonInstancesRequest
	opt_                                         map[string]interface{}
}

// AbandonInstances: Removes the specified instances from the managed
// instance group, and from any target pools where they are a member.
// The instances are not deleted. The managed instance group
// automatically reduces its targetSize value by the number of instances
// that you abandon from the group.
func (r *InstanceGroupManagersService) AbandonInstances(project string, zone string, instanceGroupManager string, instancegroupmanagersabandoninstancesrequest *InstanceGroupManagersAbandonInstancesRequest) *InstanceGroupManagersAbandonInstancesCall {
	c := &InstanceGroupManagersAbandonInstancesCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	c.instancegroupmanagersabandoninstancesrequest = instancegroupmanagersabandoninstancesrequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersAbandonInstancesCall) Fields(s ...googleapi.Field) *InstanceGroupManagersAbandonInstancesCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupManagersAbandonInstancesCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupmanagersabandoninstancesrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/abandonInstances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Removes the specified instances from the managed instance group, and from any target pools where they are a member. The instances are not deleted. The managed instance group automatically reduces its targetSize value by the number of instances that you abandon from the group.",
	//   "httpMethod": "POST",
	//   "id": "compute.instanceGroupManagers.abandonInstances",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the instance group manager.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the managed instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/abandonInstances",
	//   "request": {
	//     "$ref": "InstanceGroupManagersAbandonInstancesRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceGroupManagers.aggregatedList":

type InstanceGroupManagersAggregatedListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// AggregatedList: Retrieves the list of managed instance groups, and
// groups them by project and zone.
func (r *InstanceGroupManagersService) AggregatedList(project string) *InstanceGroupManagersAggregatedListCall {
	c := &InstanceGroupManagersAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *InstanceGroupManagersAggregatedListCall) Filter(filter string) *InstanceGroupManagersAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *InstanceGroupManagersAggregatedListCall) MaxResults(maxResults int64) *InstanceGroupManagersAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *InstanceGroupManagersAggregatedListCall) PageToken(pageToken string) *InstanceGroupManagersAggregatedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersAggregatedListCall) Fields(s ...googleapi.Field) *InstanceGroupManagersAggregatedListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupManagersAggregatedListCall) Do() (*InstanceGroupManagerAggregatedList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/aggregated/instanceGroupManagers")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *InstanceGroupManagerAggregatedList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of managed instance groups, and groups them by project and zone.",
	//   "httpMethod": "GET",
	//   "id": "compute.instanceGroupManagers.aggregatedList",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/aggregated/instanceGroupManagers",
	//   "response": {
	//     "$ref": "InstanceGroupManagerAggregatedList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instanceGroupManagers.delete":

type InstanceGroupManagersDeleteCall struct {
	s                    *Service
	project              string
	zone                 string
	instanceGroupManager string
	opt_                 map[string]interface{}
}

// Delete: Deletes the specified managed instance group resource.
func (r *InstanceGroupManagersService) Delete(project string, zone string, instanceGroupManager string) *InstanceGroupManagersDeleteCall {
	c := &InstanceGroupManagersDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersDeleteCall) Fields(s ...googleapi.Field) *InstanceGroupManagersDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupManagersDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified managed instance group resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.instanceGroupManagers.delete",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the instance group manager to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the managed instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceGroupManagers.deleteInstances":

type InstanceGroupManagersDeleteInstancesCall struct {
	s                                           *Service
	project                                     string
	zone                                        string
	instanceGroupManager                        string
	instancegroupmanagersdeleteinstancesrequest *InstanceGroupManagersDeleteInstancesRequest
	opt_                                        map[string]interface{}
}

// DeleteInstances: Deletes the specified instances. The instances are
// deleted and removed from the instance group and any target pools
// where they are a member. The managed instance group automatically
// reduces its targetSize value by the number of instances that you
// delete.
func (r *InstanceGroupManagersService) DeleteInstances(project string, zone string, instanceGroupManager string, instancegroupmanagersdeleteinstancesrequest *InstanceGroupManagersDeleteInstancesRequest) *InstanceGroupManagersDeleteInstancesCall {
	c := &InstanceGroupManagersDeleteInstancesCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	c.instancegroupmanagersdeleteinstancesrequest = instancegroupmanagersdeleteinstancesrequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersDeleteInstancesCall) Fields(s ...googleapi.Field) *InstanceGroupManagersDeleteInstancesCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupManagersDeleteInstancesCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupmanagersdeleteinstancesrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/deleteInstances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified instances. The instances are deleted and removed from the instance group and any target pools where they are a member. The managed instance group automatically reduces its targetSize value by the number of instances that you delete.",
	//   "httpMethod": "POST",
	//   "id": "compute.instanceGroupManagers.deleteInstances",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the instance group manager.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the managed instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/deleteInstances",
	//   "request": {
	//     "$ref": "InstanceGroupManagersDeleteInstancesRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceGroupManagers.get":

type InstanceGroupManagersGetCall struct {
	s                    *Service
	project              string
	zone                 string
	instanceGroupManager string
	opt_                 map[string]interface{}
}

// Get: Returns the specified managed instance group resource.
func (r *InstanceGroupManagersService) Get(project string, zone string, instanceGroupManager string) *InstanceGroupManagersGetCall {
	c := &InstanceGroupManagersGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersGetCall) Fields(s ...googleapi.Field) *InstanceGroupManagersGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupManagersGetCall) Do() (*InstanceGroupManager, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *InstanceGroupManager
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified managed instance group resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.instanceGroupManagers.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the instance group manager resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the managed instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}",
	//   "response": {
	//     "$ref": "InstanceGroupManager"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instanceGroupManagers.insert":

type InstanceGroupManagersInsertCall struct {
	s                    *Service
	project              string
	zone                 string
	instancegroupmanager *InstanceGroupManager
	opt_                 map[string]interface{}
}

// Insert: Creates a managed instance group resource in the specified
// project using the data that is included in the request.
func (r *InstanceGroupManagersService) Insert(project string, zone string, instancegroupmanager *InstanceGroupManager) *InstanceGroupManagersInsertCall {
	c := &InstanceGroupManagersInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instancegroupmanager = instancegroupmanager
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersInsertCall) Fields(s ...googleapi.Field) *InstanceGroupManagersInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupManagersInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupmanager)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a managed instance group resource in the specified project using the data that is included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.instanceGroupManagers.insert",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the managed instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers",
	//   "request": {
	//     "$ref": "InstanceGroupManager"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceGroupManagers.list":

type InstanceGroupManagersListCall struct {
	s       *Service
	project string
	zone    string
	opt_    map[string]interface{}
}

// List: Retrieves a list of managed instance groups that are contained
// within the specified project and zone.
func (r *InstanceGroupManagersService) List(project string, zone string) *InstanceGroupManagersListCall {
	c := &InstanceGroupManagersListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *InstanceGroupManagersListCall) Filter(filter string) *InstanceGroupManagersListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *InstanceGroupManagersListCall) MaxResults(maxResults int64) *InstanceGroupManagersListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *InstanceGroupManagersListCall) PageToken(pageToken string) *InstanceGroupManagersListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersListCall) Fields(s ...googleapi.Field) *InstanceGroupManagersListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupManagersListCall) Do() (*InstanceGroupManagerList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *InstanceGroupManagerList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of managed instance groups that are contained within the specified project and zone.",
	//   "httpMethod": "GET",
	//   "id": "compute.instanceGroupManagers.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the managed instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers",
	//   "response": {
	//     "$ref": "InstanceGroupManagerList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instanceGroupManagers.listManagedInstances":

type InstanceGroupManagersListManagedInstancesCall struct {
	s                    *Service
	project              string
	zone                 string
	instanceGroupManager string
	opt_                 map[string]interface{}
}

// ListManagedInstances: Lists managed instances.
func (r *InstanceGroupManagersService) ListManagedInstances(project string, zone string, instanceGroupManager string) *InstanceGroupManagersListManagedInstancesCall {
	c := &InstanceGroupManagersListManagedInstancesCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersListManagedInstancesCall) Fields(s ...googleapi.Field) *InstanceGroupManagersListManagedInstancesCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupManagersListManagedInstancesCall) Do() (*InstanceGroupManagersListManagedInstancesResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/listManagedInstances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *InstanceGroupManagersListManagedInstancesResponse
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists managed instances.",
	//   "httpMethod": "POST",
	//   "id": "compute.instanceGroupManagers.listManagedInstances",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the managed instance group.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the managed instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/listManagedInstances",
	//   "response": {
	//     "$ref": "InstanceGroupManagersListManagedInstancesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instanceGroupManagers.recreateInstances":

type InstanceGroupManagersRecreateInstancesCall struct {
	s                                             *Service
	project                                       string
	zone                                          string
	instanceGroupManager                          string
	instancegroupmanagersrecreateinstancesrequest *InstanceGroupManagersRecreateInstancesRequest
	opt_                                          map[string]interface{}
}

// RecreateInstances: Recreates the specified instances. The instances
// are deleted, then recreated using the managed instance group's
// current instance template.
func (r *InstanceGroupManagersService) RecreateInstances(project string, zone string, instanceGroupManager string, instancegroupmanagersrecreateinstancesrequest *InstanceGroupManagersRecreateInstancesRequest) *InstanceGroupManagersRecreateInstancesCall {
	c := &InstanceGroupManagersRecreateInstancesCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	c.instancegroupmanagersrecreateinstancesrequest = instancegroupmanagersrecreateinstancesrequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersRecreateInstancesCall) Fields(s ...googleapi.Field) *InstanceGroupManagersRecreateInstancesCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupManagersRecreateInstancesCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupmanagersrecreateinstancesrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/recreateInstances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Recreates the specified instances. The instances are deleted, then recreated using the managed instance group's current instance template.",
	//   "httpMethod": "POST",
	//   "id": "compute.instanceGroupManagers.recreateInstances",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the instance group manager.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the managed instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/recreateInstances",
	//   "request": {
	//     "$ref": "InstanceGroupManagersRecreateInstancesRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceGroupManagers.resize":

type InstanceGroupManagersResizeCall struct {
	s                    *Service
	project              string
	zone                 string
	instanceGroupManager string
	size                 int64
	opt_                 map[string]interface{}
}

// Resize: Resizes the managed instance group. If you increase the size,
// the group creates new instances using the current instance template.
// If you decrease the size, the group removes instances in the order
// that is outlined in Resizing a managed instance group.
func (r *InstanceGroupManagersService) Resize(project string, zone string, instanceGroupManager string, size int64) *InstanceGroupManagersResizeCall {
	c := &InstanceGroupManagersResizeCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	c.size = size
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersResizeCall) Fields(s ...googleapi.Field) *InstanceGroupManagersResizeCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupManagersResizeCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("size", fmt.Sprintf("%v", c.size))
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/resize")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Resizes the managed instance group. If you increase the size, the group creates new instances using the current instance template. If you decrease the size, the group removes instances in the order that is outlined in Resizing a managed instance group.",
	//   "httpMethod": "POST",
	//   "id": "compute.instanceGroupManagers.resize",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager",
	//     "size"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the instance group manager.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "size": {
	//       "description": "The number of running instances that the managed instance group should maintain at any given time. The group automatically adds or removes instances to maintain the number of instances specified by this parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the managed instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/resize",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceGroupManagers.setInstanceTemplate":

type InstanceGroupManagersSetInstanceTemplateCall struct {
	s                                               *Service
	project                                         string
	zone                                            string
	instanceGroupManager                            string
	instancegroupmanagerssetinstancetemplaterequest *InstanceGroupManagersSetInstanceTemplateRequest
	opt_                                            map[string]interface{}
}

// SetInstanceTemplate: Specifies the instance template to use when
// creating new instances in this group. The templates for existing
// instances in the group do not change unless you recreate them.
func (r *InstanceGroupManagersService) SetInstanceTemplate(project string, zone string, instanceGroupManager string, instancegroupmanagerssetinstancetemplaterequest *InstanceGroupManagersSetInstanceTemplateRequest) *InstanceGroupManagersSetInstanceTemplateCall {
	c := &InstanceGroupManagersSetInstanceTemplateCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	c.instancegroupmanagerssetinstancetemplaterequest = instancegroupmanagerssetinstancetemplaterequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersSetInstanceTemplateCall) Fields(s ...googleapi.Field) *InstanceGroupManagersSetInstanceTemplateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupManagersSetInstanceTemplateCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupmanagerssetinstancetemplaterequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/setInstanceTemplate")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Specifies the instance template to use when creating new instances in this group. The templates for existing instances in the group do not change unless you recreate them.",
	//   "httpMethod": "POST",
	//   "id": "compute.instanceGroupManagers.setInstanceTemplate",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the instance group manager.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the managed instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/setInstanceTemplate",
	//   "request": {
	//     "$ref": "InstanceGroupManagersSetInstanceTemplateRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceGroupManagers.setTargetPools":

type InstanceGroupManagersSetTargetPoolsCall struct {
	s                                          *Service
	project                                    string
	zone                                       string
	instanceGroupManager                       string
	instancegroupmanagerssettargetpoolsrequest *InstanceGroupManagersSetTargetPoolsRequest
	opt_                                       map[string]interface{}
}

// SetTargetPools: Modifies the target pools to which all new instances
// in this group are assigned. The target pools for existing instances
// in the group do not change unless you recreate them.
func (r *InstanceGroupManagersService) SetTargetPools(project string, zone string, instanceGroupManager string, instancegroupmanagerssettargetpoolsrequest *InstanceGroupManagersSetTargetPoolsRequest) *InstanceGroupManagersSetTargetPoolsCall {
	c := &InstanceGroupManagersSetTargetPoolsCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	c.instancegroupmanagerssettargetpoolsrequest = instancegroupmanagerssettargetpoolsrequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersSetTargetPoolsCall) Fields(s ...googleapi.Field) *InstanceGroupManagersSetTargetPoolsCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupManagersSetTargetPoolsCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupmanagerssettargetpoolsrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/setTargetPools")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Modifies the target pools to which all new instances in this group are assigned. The target pools for existing instances in the group do not change unless you recreate them.",
	//   "httpMethod": "POST",
	//   "id": "compute.instanceGroupManagers.setTargetPools",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the instance group manager.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the managed instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/setTargetPools",
	//   "request": {
	//     "$ref": "InstanceGroupManagersSetTargetPoolsRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceGroups.addInstances":

type InstanceGroupsAddInstancesCall struct {
	s                                 *Service
	project                           string
	zone                              string
	instanceGroup                     string
	instancegroupsaddinstancesrequest *InstanceGroupsAddInstancesRequest
	opt_                              map[string]interface{}
}

// AddInstances: Adds a list of instances to an instance group. All of
// the instances in the instance group must be in the same network.
func (r *InstanceGroupsService) AddInstances(project string, zone string, instanceGroup string, instancegroupsaddinstancesrequest *InstanceGroupsAddInstancesRequest) *InstanceGroupsAddInstancesCall {
	c := &InstanceGroupsAddInstancesCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroup = instanceGroup
	c.instancegroupsaddinstancesrequest = instancegroupsaddinstancesrequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupsAddInstancesCall) Fields(s ...googleapi.Field) *InstanceGroupsAddInstancesCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupsAddInstancesCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupsaddinstancesrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroups/{instanceGroup}/addInstances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":       c.project,
		"zone":          c.zone,
		"instanceGroup": c.instanceGroup,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds a list of instances to an instance group. All of the instances in the instance group must be in the same network.",
	//   "httpMethod": "POST",
	//   "id": "compute.instanceGroups.addInstances",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroup"
	//   ],
	//   "parameters": {
	//     "instanceGroup": {
	//       "description": "The name of the instance group where you are adding instances.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroups/{instanceGroup}/addInstances",
	//   "request": {
	//     "$ref": "InstanceGroupsAddInstancesRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceGroups.aggregatedList":

type InstanceGroupsAggregatedListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// AggregatedList: Retrieves the list of instance groups, and sorts them
// by zone.
func (r *InstanceGroupsService) AggregatedList(project string) *InstanceGroupsAggregatedListCall {
	c := &InstanceGroupsAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *InstanceGroupsAggregatedListCall) Filter(filter string) *InstanceGroupsAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *InstanceGroupsAggregatedListCall) MaxResults(maxResults int64) *InstanceGroupsAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *InstanceGroupsAggregatedListCall) PageToken(pageToken string) *InstanceGroupsAggregatedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupsAggregatedListCall) Fields(s ...googleapi.Field) *InstanceGroupsAggregatedListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupsAggregatedListCall) Do() (*InstanceGroupAggregatedList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/aggregated/instanceGroups")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *InstanceGroupAggregatedList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of instance groups, and sorts them by zone.",
	//   "httpMethod": "GET",
	//   "id": "compute.instanceGroups.aggregatedList",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/aggregated/instanceGroups",
	//   "response": {
	//     "$ref": "InstanceGroupAggregatedList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instanceGroups.delete":

type InstanceGroupsDeleteCall struct {
	s             *Service
	project       string
	zone          string
	instanceGroup string
	opt_          map[string]interface{}
}

// Delete: Deletes the specified instance group.
func (r *InstanceGroupsService) Delete(project string, zone string, instanceGroup string) *InstanceGroupsDeleteCall {
	c := &InstanceGroupsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroup = instanceGroup
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupsDeleteCall) Fields(s ...googleapi.Field) *InstanceGroupsDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupsDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroups/{instanceGroup}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":       c.project,
		"zone":          c.zone,
		"instanceGroup": c.instanceGroup,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified instance group.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.instanceGroups.delete",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroup"
	//   ],
	//   "parameters": {
	//     "instanceGroup": {
	//       "description": "The name of the instance group to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroups/{instanceGroup}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceGroups.get":

type InstanceGroupsGetCall struct {
	s             *Service
	project       string
	zone          string
	instanceGroup string
	opt_          map[string]interface{}
}

// Get: Returns the specified instance group resource.
func (r *InstanceGroupsService) Get(project string, zone string, instanceGroup string) *InstanceGroupsGetCall {
	c := &InstanceGroupsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroup = instanceGroup
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupsGetCall) Fields(s ...googleapi.Field) *InstanceGroupsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupsGetCall) Do() (*InstanceGroup, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroups/{instanceGroup}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":       c.project,
		"zone":          c.zone,
		"instanceGroup": c.instanceGroup,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *InstanceGroup
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified instance group resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.instanceGroups.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroup"
	//   ],
	//   "parameters": {
	//     "instanceGroup": {
	//       "description": "The name of the instance group.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroups/{instanceGroup}",
	//   "response": {
	//     "$ref": "InstanceGroup"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instanceGroups.insert":

type InstanceGroupsInsertCall struct {
	s             *Service
	project       string
	zone          string
	instancegroup *InstanceGroup
	opt_          map[string]interface{}
}

// Insert: Creates an instance group in the specified project using the
// parameters that are included in the request.
func (r *InstanceGroupsService) Insert(project string, zone string, instancegroup *InstanceGroup) *InstanceGroupsInsertCall {
	c := &InstanceGroupsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instancegroup = instancegroup
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupsInsertCall) Fields(s ...googleapi.Field) *InstanceGroupsInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupsInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroup)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroups")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates an instance group in the specified project using the parameters that are included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.instanceGroups.insert",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroups",
	//   "request": {
	//     "$ref": "InstanceGroup"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceGroups.list":

type InstanceGroupsListCall struct {
	s       *Service
	project string
	zone    string
	opt_    map[string]interface{}
}

// List: Retrieves the list of instance groups that are located in the
// specified project and zone.
func (r *InstanceGroupsService) List(project string, zone string) *InstanceGroupsListCall {
	c := &InstanceGroupsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *InstanceGroupsListCall) Filter(filter string) *InstanceGroupsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *InstanceGroupsListCall) MaxResults(maxResults int64) *InstanceGroupsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *InstanceGroupsListCall) PageToken(pageToken string) *InstanceGroupsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupsListCall) Fields(s ...googleapi.Field) *InstanceGroupsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupsListCall) Do() (*InstanceGroupList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroups")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *InstanceGroupList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of instance groups that are located in the specified project and zone.",
	//   "httpMethod": "GET",
	//   "id": "compute.instanceGroups.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroups",
	//   "response": {
	//     "$ref": "InstanceGroupList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instanceGroups.listInstances":

type InstanceGroupsListInstancesCall struct {
	s                                  *Service
	project                            string
	zone                               string
	instanceGroup                      string
	instancegroupslistinstancesrequest *InstanceGroupsListInstancesRequest
	opt_                               map[string]interface{}
}

// ListInstances: Lists instances in an instance group. The parameters
// for this method specify whether the list filters instances by state
// and named ports information.
func (r *InstanceGroupsService) ListInstances(project string, zone string, instanceGroup string, instancegroupslistinstancesrequest *InstanceGroupsListInstancesRequest) *InstanceGroupsListInstancesCall {
	c := &InstanceGroupsListInstancesCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroup = instanceGroup
	c.instancegroupslistinstancesrequest = instancegroupslistinstancesrequest
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *InstanceGroupsListInstancesCall) Filter(filter string) *InstanceGroupsListInstancesCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *InstanceGroupsListInstancesCall) MaxResults(maxResults int64) *InstanceGroupsListInstancesCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *InstanceGroupsListInstancesCall) PageToken(pageToken string) *InstanceGroupsListInstancesCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupsListInstancesCall) Fields(s ...googleapi.Field) *InstanceGroupsListInstancesCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupsListInstancesCall) Do() (*InstanceGroupsListInstances, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupslistinstancesrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroups/{instanceGroup}/listInstances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":       c.project,
		"zone":          c.zone,
		"instanceGroup": c.instanceGroup,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *InstanceGroupsListInstances
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists instances in an instance group. The parameters for this method specify whether the list filters instances by state and named ports information.",
	//   "httpMethod": "POST",
	//   "id": "compute.instanceGroups.listInstances",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroup"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "instanceGroup": {
	//       "description": "The name of the instance group from which you want to generate a list of included instances.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroups/{instanceGroup}/listInstances",
	//   "request": {
	//     "$ref": "InstanceGroupsListInstancesRequest"
	//   },
	//   "response": {
	//     "$ref": "InstanceGroupsListInstances"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instanceGroups.removeInstances":

type InstanceGroupsRemoveInstancesCall struct {
	s                                    *Service
	project                              string
	zone                                 string
	instanceGroup                        string
	instancegroupsremoveinstancesrequest *InstanceGroupsRemoveInstancesRequest
	opt_                                 map[string]interface{}
}

// RemoveInstances: Removes a list of instances from an instance group.
func (r *InstanceGroupsService) RemoveInstances(project string, zone string, instanceGroup string, instancegroupsremoveinstancesrequest *InstanceGroupsRemoveInstancesRequest) *InstanceGroupsRemoveInstancesCall {
	c := &InstanceGroupsRemoveInstancesCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroup = instanceGroup
	c.instancegroupsremoveinstancesrequest = instancegroupsremoveinstancesrequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupsRemoveInstancesCall) Fields(s ...googleapi.Field) *InstanceGroupsRemoveInstancesCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupsRemoveInstancesCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupsremoveinstancesrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroups/{instanceGroup}/removeInstances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":       c.project,
		"zone":          c.zone,
		"instanceGroup": c.instanceGroup,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Removes a list of instances from an instance group.",
	//   "httpMethod": "POST",
	//   "id": "compute.instanceGroups.removeInstances",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroup"
	//   ],
	//   "parameters": {
	//     "instanceGroup": {
	//       "description": "The name of the instance group where the specified instances will be removed.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroups/{instanceGroup}/removeInstances",
	//   "request": {
	//     "$ref": "InstanceGroupsRemoveInstancesRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceGroups.setNamedPorts":

type InstanceGroupsSetNamedPortsCall struct {
	s                                  *Service
	project                            string
	zone                               string
	instanceGroup                      string
	instancegroupssetnamedportsrequest *InstanceGroupsSetNamedPortsRequest
	opt_                               map[string]interface{}
}

// SetNamedPorts: Sets the named ports in an instance group.
func (r *InstanceGroupsService) SetNamedPorts(project string, zone string, instanceGroup string, instancegroupssetnamedportsrequest *InstanceGroupsSetNamedPortsRequest) *InstanceGroupsSetNamedPortsCall {
	c := &InstanceGroupsSetNamedPortsCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instanceGroup = instanceGroup
	c.instancegroupssetnamedportsrequest = instancegroupssetnamedportsrequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupsSetNamedPortsCall) Fields(s ...googleapi.Field) *InstanceGroupsSetNamedPortsCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceGroupsSetNamedPortsCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupssetnamedportsrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroups/{instanceGroup}/setNamedPorts")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":       c.project,
		"zone":          c.zone,
		"instanceGroup": c.instanceGroup,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sets the named ports in an instance group.",
	//   "httpMethod": "POST",
	//   "id": "compute.instanceGroups.setNamedPorts",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroup"
	//   ],
	//   "parameters": {
	//     "instanceGroup": {
	//       "description": "The name of the instance group where the named ports are updated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The URL of the zone where the instance group is located.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroups/{instanceGroup}/setNamedPorts",
	//   "request": {
	//     "$ref": "InstanceGroupsSetNamedPortsRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceTemplates.delete":

type InstanceTemplatesDeleteCall struct {
	s                *Service
	project          string
	instanceTemplate string
	opt_             map[string]interface{}
}

// Delete: Deletes the specified instance template.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instanceTemplates/delete
func (r *InstanceTemplatesService) Delete(project string, instanceTemplate string) *InstanceTemplatesDeleteCall {
	c := &InstanceTemplatesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instanceTemplate = instanceTemplate
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceTemplatesDeleteCall) Fields(s ...googleapi.Field) *InstanceTemplatesDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceTemplatesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/instanceTemplates/{instanceTemplate}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":          c.project,
		"instanceTemplate": c.instanceTemplate,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified instance template.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.instanceTemplates.delete",
	//   "parameterOrder": [
	//     "project",
	//     "instanceTemplate"
	//   ],
	//   "parameters": {
	//     "instanceTemplate": {
	//       "description": "The name of the instance template to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/instanceTemplates/{instanceTemplate}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceTemplates.get":

type InstanceTemplatesGetCall struct {
	s                *Service
	project          string
	instanceTemplate string
	opt_             map[string]interface{}
}

// Get: Returns the specified instance template resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instanceTemplates/get
func (r *InstanceTemplatesService) Get(project string, instanceTemplate string) *InstanceTemplatesGetCall {
	c := &InstanceTemplatesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instanceTemplate = instanceTemplate
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceTemplatesGetCall) Fields(s ...googleapi.Field) *InstanceTemplatesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceTemplatesGetCall) Do() (*InstanceTemplate, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/instanceTemplates/{instanceTemplate}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":          c.project,
		"instanceTemplate": c.instanceTemplate,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *InstanceTemplate
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified instance template resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.instanceTemplates.get",
	//   "parameterOrder": [
	//     "project",
	//     "instanceTemplate"
	//   ],
	//   "parameters": {
	//     "instanceTemplate": {
	//       "description": "The name of the instance template.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/instanceTemplates/{instanceTemplate}",
	//   "response": {
	//     "$ref": "InstanceTemplate"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instanceTemplates.insert":

type InstanceTemplatesInsertCall struct {
	s                *Service
	project          string
	instancetemplate *InstanceTemplate
	opt_             map[string]interface{}
}

// Insert: Creates an instance template in the specified project using
// the data that is included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instanceTemplates/insert
func (r *InstanceTemplatesService) Insert(project string, instancetemplate *InstanceTemplate) *InstanceTemplatesInsertCall {
	c := &InstanceTemplatesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instancetemplate = instancetemplate
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceTemplatesInsertCall) Fields(s ...googleapi.Field) *InstanceTemplatesInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceTemplatesInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancetemplate)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/instanceTemplates")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates an instance template in the specified project using the data that is included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.instanceTemplates.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/instanceTemplates",
	//   "request": {
	//     "$ref": "InstanceTemplate"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instanceTemplates.list":

type InstanceTemplatesListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves a list of instance templates that are contained
// within the specified project and zone.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instanceTemplates/list
func (r *InstanceTemplatesService) List(project string) *InstanceTemplatesListCall {
	c := &InstanceTemplatesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *InstanceTemplatesListCall) Filter(filter string) *InstanceTemplatesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *InstanceTemplatesListCall) MaxResults(maxResults int64) *InstanceTemplatesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *InstanceTemplatesListCall) PageToken(pageToken string) *InstanceTemplatesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceTemplatesListCall) Fields(s ...googleapi.Field) *InstanceTemplatesListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstanceTemplatesListCall) Do() (*InstanceTemplateList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/instanceTemplates")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *InstanceTemplateList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of instance templates that are contained within the specified project and zone.",
	//   "httpMethod": "GET",
	//   "id": "compute.instanceTemplates.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/instanceTemplates",
	//   "response": {
	//     "$ref": "InstanceTemplateList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instances.addAccessConfig":

type InstancesAddAccessConfigCall struct {
	s                *Service
	project          string
	zone             string
	instance         string
	networkInterface string
	accessconfig     *AccessConfig
	opt_             map[string]interface{}
}

// AddAccessConfig: Adds an access config to an instance's network
// interface.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/addAccessConfig
func (r *InstancesService) AddAccessConfig(project string, zone string, instance string, networkInterface string, accessconfig *AccessConfig) *InstancesAddAccessConfigCall {
	c := &InstancesAddAccessConfigCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.networkInterface = networkInterface
	c.accessconfig = accessconfig
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesAddAccessConfigCall) Fields(s ...googleapi.Field) *InstancesAddAccessConfigCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesAddAccessConfigCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.accessconfig)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("networkInterface", fmt.Sprintf("%v", c.networkInterface))
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances/{instance}/addAccessConfig")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"instance": c.instance,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds an access config to an instance's network interface.",
	//   "httpMethod": "POST",
	//   "id": "compute.instances.addAccessConfig",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance",
	//     "networkInterface"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "The instance name for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "networkInterface": {
	//       "description": "The name of the network interface to add to this instance.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances/{instance}/addAccessConfig",
	//   "request": {
	//     "$ref": "AccessConfig"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instances.aggregatedList":

type InstancesAggregatedListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// AggregatedList:
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/aggregatedList
func (r *InstancesService) AggregatedList(project string) *InstancesAggregatedListCall {
	c := &InstancesAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *InstancesAggregatedListCall) Filter(filter string) *InstancesAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *InstancesAggregatedListCall) MaxResults(maxResults int64) *InstancesAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *InstancesAggregatedListCall) PageToken(pageToken string) *InstancesAggregatedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesAggregatedListCall) Fields(s ...googleapi.Field) *InstancesAggregatedListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesAggregatedListCall) Do() (*InstanceAggregatedList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/aggregated/instances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *InstanceAggregatedList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "GET",
	//   "id": "compute.instances.aggregatedList",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/aggregated/instances",
	//   "response": {
	//     "$ref": "InstanceAggregatedList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instances.attachDisk":

type InstancesAttachDiskCall struct {
	s            *Service
	project      string
	zone         string
	instance     string
	attacheddisk *AttachedDisk
	opt_         map[string]interface{}
}

// AttachDisk: Attaches a Disk resource to an instance.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/attachDisk
func (r *InstancesService) AttachDisk(project string, zone string, instance string, attacheddisk *AttachedDisk) *InstancesAttachDiskCall {
	c := &InstancesAttachDiskCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.attacheddisk = attacheddisk
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesAttachDiskCall) Fields(s ...googleapi.Field) *InstancesAttachDiskCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesAttachDiskCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.attacheddisk)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances/{instance}/attachDisk")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"instance": c.instance,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Attaches a Disk resource to an instance.",
	//   "httpMethod": "POST",
	//   "id": "compute.instances.attachDisk",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Instance name.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances/{instance}/attachDisk",
	//   "request": {
	//     "$ref": "AttachedDisk"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instances.delete":

type InstancesDeleteCall struct {
	s        *Service
	project  string
	zone     string
	instance string
	opt_     map[string]interface{}
}

// Delete: Deletes the specified Instance resource. For more
// information, see Shutting down an instance.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/delete
func (r *InstancesService) Delete(project string, zone string, instance string) *InstancesDeleteCall {
	c := &InstancesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesDeleteCall) Fields(s ...googleapi.Field) *InstancesDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances/{instance}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"instance": c.instance,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified Instance resource. For more information, see Shutting down an instance.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.instances.delete",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Name of the instance resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances/{instance}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instances.deleteAccessConfig":

type InstancesDeleteAccessConfigCall struct {
	s                *Service
	project          string
	zone             string
	instance         string
	accessConfig     string
	networkInterface string
	opt_             map[string]interface{}
}

// DeleteAccessConfig: Deletes an access config from an instance's
// network interface.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/deleteAccessConfig
func (r *InstancesService) DeleteAccessConfig(project string, zone string, instance string, accessConfig string, networkInterface string) *InstancesDeleteAccessConfigCall {
	c := &InstancesDeleteAccessConfigCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.accessConfig = accessConfig
	c.networkInterface = networkInterface
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesDeleteAccessConfigCall) Fields(s ...googleapi.Field) *InstancesDeleteAccessConfigCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesDeleteAccessConfigCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("accessConfig", fmt.Sprintf("%v", c.accessConfig))
	params.Set("networkInterface", fmt.Sprintf("%v", c.networkInterface))
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances/{instance}/deleteAccessConfig")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"instance": c.instance,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes an access config from an instance's network interface.",
	//   "httpMethod": "POST",
	//   "id": "compute.instances.deleteAccessConfig",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance",
	//     "accessConfig",
	//     "networkInterface"
	//   ],
	//   "parameters": {
	//     "accessConfig": {
	//       "description": "The name of the access config to delete.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "instance": {
	//       "description": "The instance name for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "networkInterface": {
	//       "description": "The name of the network interface.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances/{instance}/deleteAccessConfig",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instances.detachDisk":

type InstancesDetachDiskCall struct {
	s          *Service
	project    string
	zone       string
	instance   string
	deviceName string
	opt_       map[string]interface{}
}

// DetachDisk: Detaches a disk from an instance.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/detachDisk
func (r *InstancesService) DetachDisk(project string, zone string, instance string, deviceName string) *InstancesDetachDiskCall {
	c := &InstancesDetachDiskCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.deviceName = deviceName
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesDetachDiskCall) Fields(s ...googleapi.Field) *InstancesDetachDiskCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesDetachDiskCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("deviceName", fmt.Sprintf("%v", c.deviceName))
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances/{instance}/detachDisk")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"instance": c.instance,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Detaches a disk from an instance.",
	//   "httpMethod": "POST",
	//   "id": "compute.instances.detachDisk",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance",
	//     "deviceName"
	//   ],
	//   "parameters": {
	//     "deviceName": {
	//       "description": "Disk device name to detach.",
	//       "location": "query",
	//       "pattern": "\\w[\\w.-]{0,254}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "instance": {
	//       "description": "Instance name.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances/{instance}/detachDisk",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instances.get":

type InstancesGetCall struct {
	s        *Service
	project  string
	zone     string
	instance string
	opt_     map[string]interface{}
}

// Get: Returns the specified instance resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/get
func (r *InstancesService) Get(project string, zone string, instance string) *InstancesGetCall {
	c := &InstancesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesGetCall) Fields(s ...googleapi.Field) *InstancesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesGetCall) Do() (*Instance, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances/{instance}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"instance": c.instance,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Instance
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified instance resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.instances.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Name of the instance resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the The name of the zone for this request..",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances/{instance}",
	//   "response": {
	//     "$ref": "Instance"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instances.getSerialPortOutput":

type InstancesGetSerialPortOutputCall struct {
	s        *Service
	project  string
	zone     string
	instance string
	opt_     map[string]interface{}
}

// GetSerialPortOutput: Returns the specified instance's serial port
// output.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/getSerialPortOutput
func (r *InstancesService) GetSerialPortOutput(project string, zone string, instance string) *InstancesGetSerialPortOutputCall {
	c := &InstancesGetSerialPortOutputCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	return c
}

// Port sets the optional parameter "port": Specifies which COM or
// serial port to retrieve data from.
func (c *InstancesGetSerialPortOutputCall) Port(port int64) *InstancesGetSerialPortOutputCall {
	c.opt_["port"] = port
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesGetSerialPortOutputCall) Fields(s ...googleapi.Field) *InstancesGetSerialPortOutputCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesGetSerialPortOutputCall) Do() (*SerialPortOutput, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["port"]; ok {
		params.Set("port", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances/{instance}/serialPort")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"instance": c.instance,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *SerialPortOutput
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified instance's serial port output.",
	//   "httpMethod": "GET",
	//   "id": "compute.instances.getSerialPortOutput",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Name of the instance scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "port": {
	//       "default": "1",
	//       "description": "Specifies which COM or serial port to retrieve data from.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "4",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances/{instance}/serialPort",
	//   "response": {
	//     "$ref": "SerialPortOutput"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instances.insert":

type InstancesInsertCall struct {
	s        *Service
	project  string
	zone     string
	instance *Instance
	opt_     map[string]interface{}
}

// Insert: Creates an instance resource in the specified project using
// the data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/insert
func (r *InstancesService) Insert(project string, zone string, instance *Instance) *InstancesInsertCall {
	c := &InstancesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesInsertCall) Fields(s ...googleapi.Field) *InstancesInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instance)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates an instance resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.instances.insert",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances",
	//   "request": {
	//     "$ref": "Instance"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instances.list":

type InstancesListCall struct {
	s       *Service
	project string
	zone    string
	opt_    map[string]interface{}
}

// List: Retrieves the list of instance resources contained within the
// specified zone.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/list
func (r *InstancesService) List(project string, zone string) *InstancesListCall {
	c := &InstancesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *InstancesListCall) Filter(filter string) *InstancesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *InstancesListCall) MaxResults(maxResults int64) *InstancesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *InstancesListCall) PageToken(pageToken string) *InstancesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesListCall) Fields(s ...googleapi.Field) *InstancesListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesListCall) Do() (*InstanceList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *InstanceList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of instance resources contained within the specified zone.",
	//   "httpMethod": "GET",
	//   "id": "compute.instances.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances",
	//   "response": {
	//     "$ref": "InstanceList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.instances.reset":

type InstancesResetCall struct {
	s        *Service
	project  string
	zone     string
	instance string
	opt_     map[string]interface{}
}

// Reset: Performs a hard reset on the instance.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/reset
func (r *InstancesService) Reset(project string, zone string, instance string) *InstancesResetCall {
	c := &InstancesResetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesResetCall) Fields(s ...googleapi.Field) *InstancesResetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesResetCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances/{instance}/reset")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"instance": c.instance,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Performs a hard reset on the instance.",
	//   "httpMethod": "POST",
	//   "id": "compute.instances.reset",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Name of the instance scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances/{instance}/reset",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instances.setDiskAutoDelete":

type InstancesSetDiskAutoDeleteCall struct {
	s          *Service
	project    string
	zone       string
	instance   string
	autoDelete bool
	deviceName string
	opt_       map[string]interface{}
}

// SetDiskAutoDelete: Sets the auto-delete flag for a disk attached to
// an instance.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/setDiskAutoDelete
func (r *InstancesService) SetDiskAutoDelete(project string, zone string, instance string, autoDelete bool, deviceName string) *InstancesSetDiskAutoDeleteCall {
	c := &InstancesSetDiskAutoDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.autoDelete = autoDelete
	c.deviceName = deviceName
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesSetDiskAutoDeleteCall) Fields(s ...googleapi.Field) *InstancesSetDiskAutoDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesSetDiskAutoDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("autoDelete", fmt.Sprintf("%v", c.autoDelete))
	params.Set("deviceName", fmt.Sprintf("%v", c.deviceName))
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances/{instance}/setDiskAutoDelete")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"instance": c.instance,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sets the auto-delete flag for a disk attached to an instance.",
	//   "httpMethod": "POST",
	//   "id": "compute.instances.setDiskAutoDelete",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance",
	//     "autoDelete",
	//     "deviceName"
	//   ],
	//   "parameters": {
	//     "autoDelete": {
	//       "description": "Whether to auto-delete the disk when the instance is deleted.",
	//       "location": "query",
	//       "required": true,
	//       "type": "boolean"
	//     },
	//     "deviceName": {
	//       "description": "The device name of the disk to modify.",
	//       "location": "query",
	//       "pattern": "\\w[\\w.-]{0,254}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "instance": {
	//       "description": "The instance name.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances/{instance}/setDiskAutoDelete",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instances.setMetadata":

type InstancesSetMetadataCall struct {
	s        *Service
	project  string
	zone     string
	instance string
	metadata *Metadata
	opt_     map[string]interface{}
}

// SetMetadata: Sets metadata for the specified instance to the data
// included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/setMetadata
func (r *InstancesService) SetMetadata(project string, zone string, instance string, metadata *Metadata) *InstancesSetMetadataCall {
	c := &InstancesSetMetadataCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.metadata = metadata
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesSetMetadataCall) Fields(s ...googleapi.Field) *InstancesSetMetadataCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesSetMetadataCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.metadata)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances/{instance}/setMetadata")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"instance": c.instance,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sets metadata for the specified instance to the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.instances.setMetadata",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Name of the instance scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances/{instance}/setMetadata",
	//   "request": {
	//     "$ref": "Metadata"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instances.setScheduling":

type InstancesSetSchedulingCall struct {
	s          *Service
	project    string
	zone       string
	instance   string
	scheduling *Scheduling
	opt_       map[string]interface{}
}

// SetScheduling: Sets an instance's scheduling options.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/setScheduling
func (r *InstancesService) SetScheduling(project string, zone string, instance string, scheduling *Scheduling) *InstancesSetSchedulingCall {
	c := &InstancesSetSchedulingCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.scheduling = scheduling
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesSetSchedulingCall) Fields(s ...googleapi.Field) *InstancesSetSchedulingCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesSetSchedulingCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.scheduling)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances/{instance}/setScheduling")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"instance": c.instance,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sets an instance's scheduling options.",
	//   "httpMethod": "POST",
	//   "id": "compute.instances.setScheduling",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Instance name.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances/{instance}/setScheduling",
	//   "request": {
	//     "$ref": "Scheduling"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instances.setTags":

type InstancesSetTagsCall struct {
	s        *Service
	project  string
	zone     string
	instance string
	tags     *Tags
	opt_     map[string]interface{}
}

// SetTags: Sets tags for the specified instance to the data included in
// the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/setTags
func (r *InstancesService) SetTags(project string, zone string, instance string, tags *Tags) *InstancesSetTagsCall {
	c := &InstancesSetTagsCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.tags = tags
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesSetTagsCall) Fields(s ...googleapi.Field) *InstancesSetTagsCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesSetTagsCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.tags)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances/{instance}/setTags")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"instance": c.instance,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sets tags for the specified instance to the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.instances.setTags",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Name of the instance scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances/{instance}/setTags",
	//   "request": {
	//     "$ref": "Tags"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instances.start":

type InstancesStartCall struct {
	s        *Service
	project  string
	zone     string
	instance string
	opt_     map[string]interface{}
}

// Start: This method starts an instance that was stopped using the
// using the instances().stop method. For more information, see Restart
// an instance.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/start
func (r *InstancesService) Start(project string, zone string, instance string) *InstancesStartCall {
	c := &InstancesStartCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesStartCall) Fields(s ...googleapi.Field) *InstancesStartCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesStartCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances/{instance}/start")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"instance": c.instance,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "This method starts an instance that was stopped using the using the instances().stop method. For more information, see Restart an instance.",
	//   "httpMethod": "POST",
	//   "id": "compute.instances.start",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Name of the instance resource to start.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances/{instance}/start",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.instances.stop":

type InstancesStopCall struct {
	s        *Service
	project  string
	zone     string
	instance string
	opt_     map[string]interface{}
}

// Stop: This method stops a running instance, shutting it down cleanly,
// and allows you to restart the instance at a later time. Stopped
// instances do not incur per-minute, virtual machine usage charges
// while they are stopped, but any resources that the virtual machine is
// using, such as persistent disks and static IP addresses,will continue
// to be charged until they are deleted. For more information, see
// Stopping an instance.
// For details, see https://cloud.google.com/compute/docs/reference/latest/instances/stop
func (r *InstancesService) Stop(project string, zone string, instance string) *InstancesStopCall {
	c := &InstancesStopCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesStopCall) Fields(s ...googleapi.Field) *InstancesStopCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *InstancesStopCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instances/{instance}/stop")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"zone":     c.zone,
		"instance": c.instance,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "This method stops a running instance, shutting it down cleanly, and allows you to restart the instance at a later time. Stopped instances do not incur per-minute, virtual machine usage charges while they are stopped, but any resources that the virtual machine is using, such as persistent disks and static IP addresses,will continue to be charged until they are deleted. For more information, see Stopping an instance.",
	//   "httpMethod": "POST",
	//   "id": "compute.instances.stop",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Name of the instance resource to stop.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instances/{instance}/stop",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.licenses.get":

type LicensesGetCall struct {
	s       *Service
	project string
	license string
	opt_    map[string]interface{}
}

// Get: Returns the specified license resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/licenses/get
func (r *LicensesService) Get(project string, license string) *LicensesGetCall {
	c := &LicensesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.license = license
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *LicensesGetCall) Fields(s ...googleapi.Field) *LicensesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *LicensesGetCall) Do() (*License, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/licenses/{license}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"license": c.license,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *License
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified license resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.licenses.get",
	//   "parameterOrder": [
	//     "project",
	//     "license"
	//   ],
	//   "parameters": {
	//     "license": {
	//       "description": "Name of the license resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/licenses/{license}",
	//   "response": {
	//     "$ref": "License"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.machineTypes.aggregatedList":

type MachineTypesAggregatedListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// AggregatedList: Retrieves the list of machine type resources grouped
// by scope.
// For details, see https://cloud.google.com/compute/docs/reference/latest/machineTypes/aggregatedList
func (r *MachineTypesService) AggregatedList(project string) *MachineTypesAggregatedListCall {
	c := &MachineTypesAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *MachineTypesAggregatedListCall) Filter(filter string) *MachineTypesAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *MachineTypesAggregatedListCall) MaxResults(maxResults int64) *MachineTypesAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *MachineTypesAggregatedListCall) PageToken(pageToken string) *MachineTypesAggregatedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MachineTypesAggregatedListCall) Fields(s ...googleapi.Field) *MachineTypesAggregatedListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *MachineTypesAggregatedListCall) Do() (*MachineTypeAggregatedList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/aggregated/machineTypes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *MachineTypeAggregatedList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of machine type resources grouped by scope.",
	//   "httpMethod": "GET",
	//   "id": "compute.machineTypes.aggregatedList",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/aggregated/machineTypes",
	//   "response": {
	//     "$ref": "MachineTypeAggregatedList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.machineTypes.get":

type MachineTypesGetCall struct {
	s           *Service
	project     string
	zone        string
	machineType string
	opt_        map[string]interface{}
}

// Get: Returns the specified machine type resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/machineTypes/get
func (r *MachineTypesService) Get(project string, zone string, machineType string) *MachineTypesGetCall {
	c := &MachineTypesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.machineType = machineType
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MachineTypesGetCall) Fields(s ...googleapi.Field) *MachineTypesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *MachineTypesGetCall) Do() (*MachineType, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/machineTypes/{machineType}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":     c.project,
		"zone":        c.zone,
		"machineType": c.machineType,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *MachineType
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified machine type resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.machineTypes.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "machineType"
	//   ],
	//   "parameters": {
	//     "machineType": {
	//       "description": "Name of the machine type resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/machineTypes/{machineType}",
	//   "response": {
	//     "$ref": "MachineType"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.machineTypes.list":

type MachineTypesListCall struct {
	s       *Service
	project string
	zone    string
	opt_    map[string]interface{}
}

// List: Retrieves the list of machine type resources available to the
// specified project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/machineTypes/list
func (r *MachineTypesService) List(project string, zone string) *MachineTypesListCall {
	c := &MachineTypesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *MachineTypesListCall) Filter(filter string) *MachineTypesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *MachineTypesListCall) MaxResults(maxResults int64) *MachineTypesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *MachineTypesListCall) PageToken(pageToken string) *MachineTypesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MachineTypesListCall) Fields(s ...googleapi.Field) *MachineTypesListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *MachineTypesListCall) Do() (*MachineTypeList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/machineTypes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *MachineTypeList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of machine type resources available to the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.machineTypes.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/machineTypes",
	//   "response": {
	//     "$ref": "MachineTypeList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.networks.delete":

type NetworksDeleteCall struct {
	s       *Service
	project string
	network string
	opt_    map[string]interface{}
}

// Delete: Deletes the specified network resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/networks/delete
func (r *NetworksService) Delete(project string, network string) *NetworksDeleteCall {
	c := &NetworksDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.network = network
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *NetworksDeleteCall) Fields(s ...googleapi.Field) *NetworksDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *NetworksDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/networks/{network}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"network": c.network,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified network resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.networks.delete",
	//   "parameterOrder": [
	//     "project",
	//     "network"
	//   ],
	//   "parameters": {
	//     "network": {
	//       "description": "Name of the network resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/networks/{network}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.networks.get":

type NetworksGetCall struct {
	s       *Service
	project string
	network string
	opt_    map[string]interface{}
}

// Get: Returns the specified network resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/networks/get
func (r *NetworksService) Get(project string, network string) *NetworksGetCall {
	c := &NetworksGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.network = network
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *NetworksGetCall) Fields(s ...googleapi.Field) *NetworksGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *NetworksGetCall) Do() (*Network, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/networks/{network}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"network": c.network,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Network
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified network resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.networks.get",
	//   "parameterOrder": [
	//     "project",
	//     "network"
	//   ],
	//   "parameters": {
	//     "network": {
	//       "description": "Name of the network resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/networks/{network}",
	//   "response": {
	//     "$ref": "Network"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.networks.insert":

type NetworksInsertCall struct {
	s       *Service
	project string
	network *Network
	opt_    map[string]interface{}
}

// Insert: Creates a network resource in the specified project using the
// data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/networks/insert
func (r *NetworksService) Insert(project string, network *Network) *NetworksInsertCall {
	c := &NetworksInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.network = network
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *NetworksInsertCall) Fields(s ...googleapi.Field) *NetworksInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *NetworksInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.network)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/networks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a network resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.networks.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/networks",
	//   "request": {
	//     "$ref": "Network"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.networks.list":

type NetworksListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of network resources available to the
// specified project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/networks/list
func (r *NetworksService) List(project string) *NetworksListCall {
	c := &NetworksListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *NetworksListCall) Filter(filter string) *NetworksListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *NetworksListCall) MaxResults(maxResults int64) *NetworksListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *NetworksListCall) PageToken(pageToken string) *NetworksListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *NetworksListCall) Fields(s ...googleapi.Field) *NetworksListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *NetworksListCall) Do() (*NetworkList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/networks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *NetworkList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of network resources available to the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.networks.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/networks",
	//   "response": {
	//     "$ref": "NetworkList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.projects.get":

type ProjectsGetCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// Get: Returns the specified project resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/projects/get
func (r *ProjectsService) Get(project string) *ProjectsGetCall {
	c := &ProjectsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsGetCall) Fields(s ...googleapi.Field) *ProjectsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ProjectsGetCall) Do() (*Project, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Project
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified project resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.projects.get",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}",
	//   "response": {
	//     "$ref": "Project"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.projects.moveDisk":

type ProjectsMoveDiskCall struct {
	s               *Service
	project         string
	diskmoverequest *DiskMoveRequest
	opt_            map[string]interface{}
}

// MoveDisk: Moves a persistent disk from one zone to another.
func (r *ProjectsService) MoveDisk(project string, diskmoverequest *DiskMoveRequest) *ProjectsMoveDiskCall {
	c := &ProjectsMoveDiskCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.diskmoverequest = diskmoverequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsMoveDiskCall) Fields(s ...googleapi.Field) *ProjectsMoveDiskCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ProjectsMoveDiskCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.diskmoverequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/moveDisk")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Moves a persistent disk from one zone to another.",
	//   "httpMethod": "POST",
	//   "id": "compute.projects.moveDisk",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/moveDisk",
	//   "request": {
	//     "$ref": "DiskMoveRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.projects.moveInstance":

type ProjectsMoveInstanceCall struct {
	s                   *Service
	project             string
	instancemoverequest *InstanceMoveRequest
	opt_                map[string]interface{}
}

// MoveInstance: Moves an instance and its attached persistent disks
// from one zone to another.
func (r *ProjectsService) MoveInstance(project string, instancemoverequest *InstanceMoveRequest) *ProjectsMoveInstanceCall {
	c := &ProjectsMoveInstanceCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instancemoverequest = instancemoverequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsMoveInstanceCall) Fields(s ...googleapi.Field) *ProjectsMoveInstanceCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ProjectsMoveInstanceCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancemoverequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/moveInstance")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Moves an instance and its attached persistent disks from one zone to another.",
	//   "httpMethod": "POST",
	//   "id": "compute.projects.moveInstance",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/moveInstance",
	//   "request": {
	//     "$ref": "InstanceMoveRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.projects.setCommonInstanceMetadata":

type ProjectsSetCommonInstanceMetadataCall struct {
	s        *Service
	project  string
	metadata *Metadata
	opt_     map[string]interface{}
}

// SetCommonInstanceMetadata: Sets metadata common to all instances
// within the specified project using the data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/projects/setCommonInstanceMetadata
func (r *ProjectsService) SetCommonInstanceMetadata(project string, metadata *Metadata) *ProjectsSetCommonInstanceMetadataCall {
	c := &ProjectsSetCommonInstanceMetadataCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.metadata = metadata
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSetCommonInstanceMetadataCall) Fields(s ...googleapi.Field) *ProjectsSetCommonInstanceMetadataCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ProjectsSetCommonInstanceMetadataCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.metadata)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/setCommonInstanceMetadata")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sets metadata common to all instances within the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.projects.setCommonInstanceMetadata",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/setCommonInstanceMetadata",
	//   "request": {
	//     "$ref": "Metadata"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.projects.setUsageExportBucket":

type ProjectsSetUsageExportBucketCall struct {
	s                   *Service
	project             string
	usageexportlocation *UsageExportLocation
	opt_                map[string]interface{}
}

// SetUsageExportBucket: Enables the usage export feature and sets the
// usage export bucket where reports are stored. If you provide an empty
// request body using this method, the usage export feature will be
// disabled.
// For details, see https://cloud.google.com/compute/docs/reference/latest/projects/setUsageExportBucket
func (r *ProjectsService) SetUsageExportBucket(project string, usageexportlocation *UsageExportLocation) *ProjectsSetUsageExportBucketCall {
	c := &ProjectsSetUsageExportBucketCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.usageexportlocation = usageexportlocation
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSetUsageExportBucketCall) Fields(s ...googleapi.Field) *ProjectsSetUsageExportBucketCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ProjectsSetUsageExportBucketCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.usageexportlocation)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/setUsageExportBucket")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Enables the usage export feature and sets the usage export bucket where reports are stored. If you provide an empty request body using this method, the usage export feature will be disabled.",
	//   "httpMethod": "POST",
	//   "id": "compute.projects.setUsageExportBucket",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/setUsageExportBucket",
	//   "request": {
	//     "$ref": "UsageExportLocation"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_only",
	//     "https://www.googleapis.com/auth/devstorage.read_write"
	//   ]
	// }

}

// method id "compute.regionOperations.delete":

type RegionOperationsDeleteCall struct {
	s         *Service
	project   string
	region    string
	operation string
	opt_      map[string]interface{}
}

// Delete: Deletes the specified region-specific Operations resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/regionOperations/delete
func (r *RegionOperationsService) Delete(project string, region string, operation string) *RegionOperationsDeleteCall {
	c := &RegionOperationsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.operation = operation
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RegionOperationsDeleteCall) Fields(s ...googleapi.Field) *RegionOperationsDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *RegionOperationsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/operations/{operation}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"region":    c.region,
		"operation": c.operation,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the specified region-specific Operations resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.regionOperations.delete",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the Operations resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/operations/{operation}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.regionOperations.get":

type RegionOperationsGetCall struct {
	s         *Service
	project   string
	region    string
	operation string
	opt_      map[string]interface{}
}

// Get: Retrieves the specified region-specific Operations resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/regionOperations/get
func (r *RegionOperationsService) Get(project string, region string, operation string) *RegionOperationsGetCall {
	c := &RegionOperationsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.operation = operation
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RegionOperationsGetCall) Fields(s ...googleapi.Field) *RegionOperationsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *RegionOperationsGetCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/operations/{operation}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"region":    c.region,
		"operation": c.operation,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the specified region-specific Operations resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.regionOperations.get",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the Operations resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/operations/{operation}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.regionOperations.list":

type RegionOperationsListCall struct {
	s       *Service
	project string
	region  string
	opt_    map[string]interface{}
}

// List: Retrieves the list of Operation resources contained within the
// specified region.
// For details, see https://cloud.google.com/compute/docs/reference/latest/regionOperations/list
func (r *RegionOperationsService) List(project string, region string) *RegionOperationsListCall {
	c := &RegionOperationsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *RegionOperationsListCall) Filter(filter string) *RegionOperationsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *RegionOperationsListCall) MaxResults(maxResults int64) *RegionOperationsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *RegionOperationsListCall) PageToken(pageToken string) *RegionOperationsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RegionOperationsListCall) Fields(s ...googleapi.Field) *RegionOperationsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *RegionOperationsListCall) Do() (*OperationList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/operations")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"region":  c.region,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *OperationList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of Operation resources contained within the specified region.",
	//   "httpMethod": "GET",
	//   "id": "compute.regionOperations.list",
	//   "parameterOrder": [
	//     "project",
	//     "region"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/operations",
	//   "response": {
	//     "$ref": "OperationList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.regions.get":

type RegionsGetCall struct {
	s       *Service
	project string
	region  string
	opt_    map[string]interface{}
}

// Get: Returns the specified region resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/regions/get
func (r *RegionsService) Get(project string, region string) *RegionsGetCall {
	c := &RegionsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RegionsGetCall) Fields(s ...googleapi.Field) *RegionsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *RegionsGetCall) Do() (*Region, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"region":  c.region,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Region
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified region resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.regions.get",
	//   "parameterOrder": [
	//     "project",
	//     "region"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}",
	//   "response": {
	//     "$ref": "Region"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.regions.list":

type RegionsListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of region resources available to the
// specified project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/regions/list
func (r *RegionsService) List(project string) *RegionsListCall {
	c := &RegionsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *RegionsListCall) Filter(filter string) *RegionsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *RegionsListCall) MaxResults(maxResults int64) *RegionsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *RegionsListCall) PageToken(pageToken string) *RegionsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RegionsListCall) Fields(s ...googleapi.Field) *RegionsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *RegionsListCall) Do() (*RegionList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *RegionList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of region resources available to the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.regions.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions",
	//   "response": {
	//     "$ref": "RegionList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.routes.delete":

type RoutesDeleteCall struct {
	s       *Service
	project string
	route   string
	opt_    map[string]interface{}
}

// Delete: Deletes the specified route resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/routes/delete
func (r *RoutesService) Delete(project string, route string) *RoutesDeleteCall {
	c := &RoutesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.route = route
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RoutesDeleteCall) Fields(s ...googleapi.Field) *RoutesDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *RoutesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/routes/{route}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"route":   c.route,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified route resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.routes.delete",
	//   "parameterOrder": [
	//     "project",
	//     "route"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "route": {
	//       "description": "Name of the route resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/routes/{route}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.routes.get":

type RoutesGetCall struct {
	s       *Service
	project string
	route   string
	opt_    map[string]interface{}
}

// Get: Returns the specified route resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/routes/get
func (r *RoutesService) Get(project string, route string) *RoutesGetCall {
	c := &RoutesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.route = route
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RoutesGetCall) Fields(s ...googleapi.Field) *RoutesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *RoutesGetCall) Do() (*Route, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/routes/{route}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"route":   c.route,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Route
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified route resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.routes.get",
	//   "parameterOrder": [
	//     "project",
	//     "route"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "route": {
	//       "description": "Name of the route resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/routes/{route}",
	//   "response": {
	//     "$ref": "Route"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.routes.insert":

type RoutesInsertCall struct {
	s       *Service
	project string
	route   *Route
	opt_    map[string]interface{}
}

// Insert: Creates a route resource in the specified project using the
// data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/routes/insert
func (r *RoutesService) Insert(project string, route *Route) *RoutesInsertCall {
	c := &RoutesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.route = route
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RoutesInsertCall) Fields(s ...googleapi.Field) *RoutesInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *RoutesInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.route)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/routes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a route resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.routes.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/routes",
	//   "request": {
	//     "$ref": "Route"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.routes.list":

type RoutesListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of route resources available to the
// specified project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/routes/list
func (r *RoutesService) List(project string) *RoutesListCall {
	c := &RoutesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *RoutesListCall) Filter(filter string) *RoutesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *RoutesListCall) MaxResults(maxResults int64) *RoutesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *RoutesListCall) PageToken(pageToken string) *RoutesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RoutesListCall) Fields(s ...googleapi.Field) *RoutesListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *RoutesListCall) Do() (*RouteList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/routes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *RouteList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of route resources available to the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.routes.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/routes",
	//   "response": {
	//     "$ref": "RouteList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.snapshots.delete":

type SnapshotsDeleteCall struct {
	s        *Service
	project  string
	snapshot string
	opt_     map[string]interface{}
}

// Delete: Deletes the specified Snapshot resource. Keep in mind that
// deleting a single snapshot might not necessarily delete all the data
// on that snapshot. If any data on the snapshot that is marked for
// deletion is needed for subsequent snapshots, the data will be moved
// to the next corresponding snapshot.
//
// For more information, see Deleting snaphots.
// For details, see https://cloud.google.com/compute/docs/reference/latest/snapshots/delete
func (r *SnapshotsService) Delete(project string, snapshot string) *SnapshotsDeleteCall {
	c := &SnapshotsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.snapshot = snapshot
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SnapshotsDeleteCall) Fields(s ...googleapi.Field) *SnapshotsDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *SnapshotsDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/snapshots/{snapshot}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"snapshot": c.snapshot,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified Snapshot resource. Keep in mind that deleting a single snapshot might not necessarily delete all the data on that snapshot. If any data on the snapshot that is marked for deletion is needed for subsequent snapshots, the data will be moved to the next corresponding snapshot.\n\nFor more information, see Deleting snaphots.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.snapshots.delete",
	//   "parameterOrder": [
	//     "project",
	//     "snapshot"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "snapshot": {
	//       "description": "Name of the Snapshot resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/snapshots/{snapshot}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.snapshots.get":

type SnapshotsGetCall struct {
	s        *Service
	project  string
	snapshot string
	opt_     map[string]interface{}
}

// Get: Returns the specified Snapshot resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/snapshots/get
func (r *SnapshotsService) Get(project string, snapshot string) *SnapshotsGetCall {
	c := &SnapshotsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.snapshot = snapshot
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SnapshotsGetCall) Fields(s ...googleapi.Field) *SnapshotsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *SnapshotsGetCall) Do() (*Snapshot, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/snapshots/{snapshot}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"snapshot": c.snapshot,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Snapshot
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified Snapshot resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.snapshots.get",
	//   "parameterOrder": [
	//     "project",
	//     "snapshot"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "snapshot": {
	//       "description": "Name of the Snapshot resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/snapshots/{snapshot}",
	//   "response": {
	//     "$ref": "Snapshot"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.snapshots.list":

type SnapshotsListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of Snapshot resources contained within the
// specified project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/snapshots/list
func (r *SnapshotsService) List(project string) *SnapshotsListCall {
	c := &SnapshotsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *SnapshotsListCall) Filter(filter string) *SnapshotsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *SnapshotsListCall) MaxResults(maxResults int64) *SnapshotsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *SnapshotsListCall) PageToken(pageToken string) *SnapshotsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SnapshotsListCall) Fields(s ...googleapi.Field) *SnapshotsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *SnapshotsListCall) Do() (*SnapshotList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/snapshots")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *SnapshotList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of Snapshot resources contained within the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.snapshots.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/snapshots",
	//   "response": {
	//     "$ref": "SnapshotList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.targetHttpProxies.delete":

type TargetHttpProxiesDeleteCall struct {
	s               *Service
	project         string
	targetHttpProxy string
	opt_            map[string]interface{}
}

// Delete: Deletes the specified TargetHttpProxy resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetHttpProxies/delete
func (r *TargetHttpProxiesService) Delete(project string, targetHttpProxy string) *TargetHttpProxiesDeleteCall {
	c := &TargetHttpProxiesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.targetHttpProxy = targetHttpProxy
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetHttpProxiesDeleteCall) Fields(s ...googleapi.Field) *TargetHttpProxiesDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetHttpProxiesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/targetHttpProxies/{targetHttpProxy}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":         c.project,
		"targetHttpProxy": c.targetHttpProxy,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified TargetHttpProxy resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.targetHttpProxies.delete",
	//   "parameterOrder": [
	//     "project",
	//     "targetHttpProxy"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetHttpProxy": {
	//       "description": "Name of the TargetHttpProxy resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/targetHttpProxies/{targetHttpProxy}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetHttpProxies.get":

type TargetHttpProxiesGetCall struct {
	s               *Service
	project         string
	targetHttpProxy string
	opt_            map[string]interface{}
}

// Get: Returns the specified TargetHttpProxy resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetHttpProxies/get
func (r *TargetHttpProxiesService) Get(project string, targetHttpProxy string) *TargetHttpProxiesGetCall {
	c := &TargetHttpProxiesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.targetHttpProxy = targetHttpProxy
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetHttpProxiesGetCall) Fields(s ...googleapi.Field) *TargetHttpProxiesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetHttpProxiesGetCall) Do() (*TargetHttpProxy, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/targetHttpProxies/{targetHttpProxy}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":         c.project,
		"targetHttpProxy": c.targetHttpProxy,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *TargetHttpProxy
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified TargetHttpProxy resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.targetHttpProxies.get",
	//   "parameterOrder": [
	//     "project",
	//     "targetHttpProxy"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetHttpProxy": {
	//       "description": "Name of the TargetHttpProxy resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/targetHttpProxies/{targetHttpProxy}",
	//   "response": {
	//     "$ref": "TargetHttpProxy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.targetHttpProxies.insert":

type TargetHttpProxiesInsertCall struct {
	s               *Service
	project         string
	targethttpproxy *TargetHttpProxy
	opt_            map[string]interface{}
}

// Insert: Creates a TargetHttpProxy resource in the specified project
// using the data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetHttpProxies/insert
func (r *TargetHttpProxiesService) Insert(project string, targethttpproxy *TargetHttpProxy) *TargetHttpProxiesInsertCall {
	c := &TargetHttpProxiesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.targethttpproxy = targethttpproxy
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetHttpProxiesInsertCall) Fields(s ...googleapi.Field) *TargetHttpProxiesInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetHttpProxiesInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.targethttpproxy)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/targetHttpProxies")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a TargetHttpProxy resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.targetHttpProxies.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/targetHttpProxies",
	//   "request": {
	//     "$ref": "TargetHttpProxy"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetHttpProxies.list":

type TargetHttpProxiesListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of TargetHttpProxy resources available to
// the specified project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetHttpProxies/list
func (r *TargetHttpProxiesService) List(project string) *TargetHttpProxiesListCall {
	c := &TargetHttpProxiesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *TargetHttpProxiesListCall) Filter(filter string) *TargetHttpProxiesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *TargetHttpProxiesListCall) MaxResults(maxResults int64) *TargetHttpProxiesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *TargetHttpProxiesListCall) PageToken(pageToken string) *TargetHttpProxiesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetHttpProxiesListCall) Fields(s ...googleapi.Field) *TargetHttpProxiesListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetHttpProxiesListCall) Do() (*TargetHttpProxyList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/targetHttpProxies")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *TargetHttpProxyList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of TargetHttpProxy resources available to the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.targetHttpProxies.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/targetHttpProxies",
	//   "response": {
	//     "$ref": "TargetHttpProxyList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.targetHttpProxies.setUrlMap":

type TargetHttpProxiesSetUrlMapCall struct {
	s               *Service
	project         string
	targetHttpProxy string
	urlmapreference *UrlMapReference
	opt_            map[string]interface{}
}

// SetUrlMap: Changes the URL map for TargetHttpProxy.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetHttpProxies/setUrlMap
func (r *TargetHttpProxiesService) SetUrlMap(project string, targetHttpProxy string, urlmapreference *UrlMapReference) *TargetHttpProxiesSetUrlMapCall {
	c := &TargetHttpProxiesSetUrlMapCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.targetHttpProxy = targetHttpProxy
	c.urlmapreference = urlmapreference
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetHttpProxiesSetUrlMapCall) Fields(s ...googleapi.Field) *TargetHttpProxiesSetUrlMapCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetHttpProxiesSetUrlMapCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.urlmapreference)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/targetHttpProxies/{targetHttpProxy}/setUrlMap")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":         c.project,
		"targetHttpProxy": c.targetHttpProxy,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Changes the URL map for TargetHttpProxy.",
	//   "httpMethod": "POST",
	//   "id": "compute.targetHttpProxies.setUrlMap",
	//   "parameterOrder": [
	//     "project",
	//     "targetHttpProxy"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetHttpProxy": {
	//       "description": "Name of the TargetHttpProxy resource whose URL map is to be set.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/targetHttpProxies/{targetHttpProxy}/setUrlMap",
	//   "request": {
	//     "$ref": "UrlMapReference"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetInstances.aggregatedList":

type TargetInstancesAggregatedListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// AggregatedList: Retrieves the list of target instances grouped by
// scope.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetInstances/aggregatedList
func (r *TargetInstancesService) AggregatedList(project string) *TargetInstancesAggregatedListCall {
	c := &TargetInstancesAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *TargetInstancesAggregatedListCall) Filter(filter string) *TargetInstancesAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *TargetInstancesAggregatedListCall) MaxResults(maxResults int64) *TargetInstancesAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *TargetInstancesAggregatedListCall) PageToken(pageToken string) *TargetInstancesAggregatedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetInstancesAggregatedListCall) Fields(s ...googleapi.Field) *TargetInstancesAggregatedListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetInstancesAggregatedListCall) Do() (*TargetInstanceAggregatedList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/aggregated/targetInstances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *TargetInstanceAggregatedList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of target instances grouped by scope.",
	//   "httpMethod": "GET",
	//   "id": "compute.targetInstances.aggregatedList",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/aggregated/targetInstances",
	//   "response": {
	//     "$ref": "TargetInstanceAggregatedList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.targetInstances.delete":

type TargetInstancesDeleteCall struct {
	s              *Service
	project        string
	zone           string
	targetInstance string
	opt_           map[string]interface{}
}

// Delete: Deletes the specified TargetInstance resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetInstances/delete
func (r *TargetInstancesService) Delete(project string, zone string, targetInstance string) *TargetInstancesDeleteCall {
	c := &TargetInstancesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.targetInstance = targetInstance
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetInstancesDeleteCall) Fields(s ...googleapi.Field) *TargetInstancesDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetInstancesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/targetInstances/{targetInstance}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":        c.project,
		"zone":           c.zone,
		"targetInstance": c.targetInstance,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified TargetInstance resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.targetInstances.delete",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "targetInstance"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetInstance": {
	//       "description": "Name of the TargetInstance resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/targetInstances/{targetInstance}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetInstances.get":

type TargetInstancesGetCall struct {
	s              *Service
	project        string
	zone           string
	targetInstance string
	opt_           map[string]interface{}
}

// Get: Returns the specified TargetInstance resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetInstances/get
func (r *TargetInstancesService) Get(project string, zone string, targetInstance string) *TargetInstancesGetCall {
	c := &TargetInstancesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.targetInstance = targetInstance
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetInstancesGetCall) Fields(s ...googleapi.Field) *TargetInstancesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetInstancesGetCall) Do() (*TargetInstance, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/targetInstances/{targetInstance}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":        c.project,
		"zone":           c.zone,
		"targetInstance": c.targetInstance,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *TargetInstance
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified TargetInstance resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.targetInstances.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "targetInstance"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetInstance": {
	//       "description": "Name of the TargetInstance resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/targetInstances/{targetInstance}",
	//   "response": {
	//     "$ref": "TargetInstance"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.targetInstances.insert":

type TargetInstancesInsertCall struct {
	s              *Service
	project        string
	zone           string
	targetinstance *TargetInstance
	opt_           map[string]interface{}
}

// Insert: Creates a TargetInstance resource in the specified project
// and zone using the data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetInstances/insert
func (r *TargetInstancesService) Insert(project string, zone string, targetinstance *TargetInstance) *TargetInstancesInsertCall {
	c := &TargetInstancesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.targetinstance = targetinstance
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetInstancesInsertCall) Fields(s ...googleapi.Field) *TargetInstancesInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetInstancesInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.targetinstance)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/targetInstances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a TargetInstance resource in the specified project and zone using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.targetInstances.insert",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/targetInstances",
	//   "request": {
	//     "$ref": "TargetInstance"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetInstances.list":

type TargetInstancesListCall struct {
	s       *Service
	project string
	zone    string
	opt_    map[string]interface{}
}

// List: Retrieves the list of TargetInstance resources available to the
// specified project and zone.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetInstances/list
func (r *TargetInstancesService) List(project string, zone string) *TargetInstancesListCall {
	c := &TargetInstancesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *TargetInstancesListCall) Filter(filter string) *TargetInstancesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *TargetInstancesListCall) MaxResults(maxResults int64) *TargetInstancesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *TargetInstancesListCall) PageToken(pageToken string) *TargetInstancesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetInstancesListCall) Fields(s ...googleapi.Field) *TargetInstancesListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetInstancesListCall) Do() (*TargetInstanceList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/targetInstances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *TargetInstanceList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of TargetInstance resources available to the specified project and zone.",
	//   "httpMethod": "GET",
	//   "id": "compute.targetInstances.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/targetInstances",
	//   "response": {
	//     "$ref": "TargetInstanceList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.targetPools.addHealthCheck":

type TargetPoolsAddHealthCheckCall struct {
	s                                *Service
	project                          string
	region                           string
	targetPool                       string
	targetpoolsaddhealthcheckrequest *TargetPoolsAddHealthCheckRequest
	opt_                             map[string]interface{}
}

// AddHealthCheck: Adds health check URL to targetPool.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetPools/addHealthCheck
func (r *TargetPoolsService) AddHealthCheck(project string, region string, targetPool string, targetpoolsaddhealthcheckrequest *TargetPoolsAddHealthCheckRequest) *TargetPoolsAddHealthCheckCall {
	c := &TargetPoolsAddHealthCheckCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	c.targetpoolsaddhealthcheckrequest = targetpoolsaddhealthcheckrequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetPoolsAddHealthCheckCall) Fields(s ...googleapi.Field) *TargetPoolsAddHealthCheckCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetPoolsAddHealthCheckCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.targetpoolsaddhealthcheckrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/targetPools/{targetPool}/addHealthCheck")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"region":     c.region,
		"targetPool": c.targetPool,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds health check URL to targetPool.",
	//   "httpMethod": "POST",
	//   "id": "compute.targetPools.addHealthCheck",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "targetPool"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetPool": {
	//       "description": "Name of the TargetPool resource to which health_check_url is to be added.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetPools/{targetPool}/addHealthCheck",
	//   "request": {
	//     "$ref": "TargetPoolsAddHealthCheckRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetPools.addInstance":

type TargetPoolsAddInstanceCall struct {
	s                             *Service
	project                       string
	region                        string
	targetPool                    string
	targetpoolsaddinstancerequest *TargetPoolsAddInstanceRequest
	opt_                          map[string]interface{}
}

// AddInstance: Adds instance url to targetPool.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetPools/addInstance
func (r *TargetPoolsService) AddInstance(project string, region string, targetPool string, targetpoolsaddinstancerequest *TargetPoolsAddInstanceRequest) *TargetPoolsAddInstanceCall {
	c := &TargetPoolsAddInstanceCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	c.targetpoolsaddinstancerequest = targetpoolsaddinstancerequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetPoolsAddInstanceCall) Fields(s ...googleapi.Field) *TargetPoolsAddInstanceCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetPoolsAddInstanceCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.targetpoolsaddinstancerequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/targetPools/{targetPool}/addInstance")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"region":     c.region,
		"targetPool": c.targetPool,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds instance url to targetPool.",
	//   "httpMethod": "POST",
	//   "id": "compute.targetPools.addInstance",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "targetPool"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetPool": {
	//       "description": "Name of the TargetPool resource to which instance_url is to be added.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetPools/{targetPool}/addInstance",
	//   "request": {
	//     "$ref": "TargetPoolsAddInstanceRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetPools.aggregatedList":

type TargetPoolsAggregatedListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// AggregatedList: Retrieves the list of target pools grouped by scope.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetPools/aggregatedList
func (r *TargetPoolsService) AggregatedList(project string) *TargetPoolsAggregatedListCall {
	c := &TargetPoolsAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *TargetPoolsAggregatedListCall) Filter(filter string) *TargetPoolsAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *TargetPoolsAggregatedListCall) MaxResults(maxResults int64) *TargetPoolsAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *TargetPoolsAggregatedListCall) PageToken(pageToken string) *TargetPoolsAggregatedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetPoolsAggregatedListCall) Fields(s ...googleapi.Field) *TargetPoolsAggregatedListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetPoolsAggregatedListCall) Do() (*TargetPoolAggregatedList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/aggregated/targetPools")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *TargetPoolAggregatedList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of target pools grouped by scope.",
	//   "httpMethod": "GET",
	//   "id": "compute.targetPools.aggregatedList",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/aggregated/targetPools",
	//   "response": {
	//     "$ref": "TargetPoolAggregatedList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.targetPools.delete":

type TargetPoolsDeleteCall struct {
	s          *Service
	project    string
	region     string
	targetPool string
	opt_       map[string]interface{}
}

// Delete: Deletes the specified TargetPool resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetPools/delete
func (r *TargetPoolsService) Delete(project string, region string, targetPool string) *TargetPoolsDeleteCall {
	c := &TargetPoolsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetPoolsDeleteCall) Fields(s ...googleapi.Field) *TargetPoolsDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetPoolsDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/targetPools/{targetPool}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"region":     c.region,
		"targetPool": c.targetPool,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified TargetPool resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.targetPools.delete",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "targetPool"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetPool": {
	//       "description": "Name of the TargetPool resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetPools/{targetPool}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetPools.get":

type TargetPoolsGetCall struct {
	s          *Service
	project    string
	region     string
	targetPool string
	opt_       map[string]interface{}
}

// Get: Returns the specified TargetPool resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetPools/get
func (r *TargetPoolsService) Get(project string, region string, targetPool string) *TargetPoolsGetCall {
	c := &TargetPoolsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetPoolsGetCall) Fields(s ...googleapi.Field) *TargetPoolsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetPoolsGetCall) Do() (*TargetPool, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/targetPools/{targetPool}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"region":     c.region,
		"targetPool": c.targetPool,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *TargetPool
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified TargetPool resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.targetPools.get",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "targetPool"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetPool": {
	//       "description": "Name of the TargetPool resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetPools/{targetPool}",
	//   "response": {
	//     "$ref": "TargetPool"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.targetPools.getHealth":

type TargetPoolsGetHealthCall struct {
	s                 *Service
	project           string
	region            string
	targetPool        string
	instancereference *InstanceReference
	opt_              map[string]interface{}
}

// GetHealth: Gets the most recent health check results for each IP for
// the given instance that is referenced by given TargetPool.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetPools/getHealth
func (r *TargetPoolsService) GetHealth(project string, region string, targetPool string, instancereference *InstanceReference) *TargetPoolsGetHealthCall {
	c := &TargetPoolsGetHealthCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	c.instancereference = instancereference
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetPoolsGetHealthCall) Fields(s ...googleapi.Field) *TargetPoolsGetHealthCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetPoolsGetHealthCall) Do() (*TargetPoolInstanceHealth, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancereference)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/targetPools/{targetPool}/getHealth")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"region":     c.region,
		"targetPool": c.targetPool,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *TargetPoolInstanceHealth
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the most recent health check results for each IP for the given instance that is referenced by given TargetPool.",
	//   "httpMethod": "POST",
	//   "id": "compute.targetPools.getHealth",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "targetPool"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetPool": {
	//       "description": "Name of the TargetPool resource to which the queried instance belongs.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetPools/{targetPool}/getHealth",
	//   "request": {
	//     "$ref": "InstanceReference"
	//   },
	//   "response": {
	//     "$ref": "TargetPoolInstanceHealth"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.targetPools.insert":

type TargetPoolsInsertCall struct {
	s          *Service
	project    string
	region     string
	targetpool *TargetPool
	opt_       map[string]interface{}
}

// Insert: Creates a TargetPool resource in the specified project and
// region using the data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetPools/insert
func (r *TargetPoolsService) Insert(project string, region string, targetpool *TargetPool) *TargetPoolsInsertCall {
	c := &TargetPoolsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetpool = targetpool
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetPoolsInsertCall) Fields(s ...googleapi.Field) *TargetPoolsInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetPoolsInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.targetpool)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/targetPools")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"region":  c.region,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a TargetPool resource in the specified project and region using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.targetPools.insert",
	//   "parameterOrder": [
	//     "project",
	//     "region"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetPools",
	//   "request": {
	//     "$ref": "TargetPool"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetPools.list":

type TargetPoolsListCall struct {
	s       *Service
	project string
	region  string
	opt_    map[string]interface{}
}

// List: Retrieves the list of TargetPool resources available to the
// specified project and region.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetPools/list
func (r *TargetPoolsService) List(project string, region string) *TargetPoolsListCall {
	c := &TargetPoolsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *TargetPoolsListCall) Filter(filter string) *TargetPoolsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *TargetPoolsListCall) MaxResults(maxResults int64) *TargetPoolsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *TargetPoolsListCall) PageToken(pageToken string) *TargetPoolsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetPoolsListCall) Fields(s ...googleapi.Field) *TargetPoolsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetPoolsListCall) Do() (*TargetPoolList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/targetPools")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"region":  c.region,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *TargetPoolList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of TargetPool resources available to the specified project and region.",
	//   "httpMethod": "GET",
	//   "id": "compute.targetPools.list",
	//   "parameterOrder": [
	//     "project",
	//     "region"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetPools",
	//   "response": {
	//     "$ref": "TargetPoolList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.targetPools.removeHealthCheck":

type TargetPoolsRemoveHealthCheckCall struct {
	s                                   *Service
	project                             string
	region                              string
	targetPool                          string
	targetpoolsremovehealthcheckrequest *TargetPoolsRemoveHealthCheckRequest
	opt_                                map[string]interface{}
}

// RemoveHealthCheck: Removes health check URL from targetPool.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetPools/removeHealthCheck
func (r *TargetPoolsService) RemoveHealthCheck(project string, region string, targetPool string, targetpoolsremovehealthcheckrequest *TargetPoolsRemoveHealthCheckRequest) *TargetPoolsRemoveHealthCheckCall {
	c := &TargetPoolsRemoveHealthCheckCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	c.targetpoolsremovehealthcheckrequest = targetpoolsremovehealthcheckrequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetPoolsRemoveHealthCheckCall) Fields(s ...googleapi.Field) *TargetPoolsRemoveHealthCheckCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetPoolsRemoveHealthCheckCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.targetpoolsremovehealthcheckrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/targetPools/{targetPool}/removeHealthCheck")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"region":     c.region,
		"targetPool": c.targetPool,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Removes health check URL from targetPool.",
	//   "httpMethod": "POST",
	//   "id": "compute.targetPools.removeHealthCheck",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "targetPool"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetPool": {
	//       "description": "Name of the TargetPool resource to which health_check_url is to be removed.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetPools/{targetPool}/removeHealthCheck",
	//   "request": {
	//     "$ref": "TargetPoolsRemoveHealthCheckRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetPools.removeInstance":

type TargetPoolsRemoveInstanceCall struct {
	s                                *Service
	project                          string
	region                           string
	targetPool                       string
	targetpoolsremoveinstancerequest *TargetPoolsRemoveInstanceRequest
	opt_                             map[string]interface{}
}

// RemoveInstance: Removes instance URL from targetPool.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetPools/removeInstance
func (r *TargetPoolsService) RemoveInstance(project string, region string, targetPool string, targetpoolsremoveinstancerequest *TargetPoolsRemoveInstanceRequest) *TargetPoolsRemoveInstanceCall {
	c := &TargetPoolsRemoveInstanceCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	c.targetpoolsremoveinstancerequest = targetpoolsremoveinstancerequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetPoolsRemoveInstanceCall) Fields(s ...googleapi.Field) *TargetPoolsRemoveInstanceCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetPoolsRemoveInstanceCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.targetpoolsremoveinstancerequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/targetPools/{targetPool}/removeInstance")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"region":     c.region,
		"targetPool": c.targetPool,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Removes instance URL from targetPool.",
	//   "httpMethod": "POST",
	//   "id": "compute.targetPools.removeInstance",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "targetPool"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetPool": {
	//       "description": "Name of the TargetPool resource to which instance_url is to be removed.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetPools/{targetPool}/removeInstance",
	//   "request": {
	//     "$ref": "TargetPoolsRemoveInstanceRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetPools.setBackup":

type TargetPoolsSetBackupCall struct {
	s               *Service
	project         string
	region          string
	targetPool      string
	targetreference *TargetReference
	opt_            map[string]interface{}
}

// SetBackup: Changes backup pool configurations.
// For details, see https://cloud.google.com/compute/docs/reference/latest/targetPools/setBackup
func (r *TargetPoolsService) SetBackup(project string, region string, targetPool string, targetreference *TargetReference) *TargetPoolsSetBackupCall {
	c := &TargetPoolsSetBackupCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	c.targetreference = targetreference
	return c
}

// FailoverRatio sets the optional parameter "failoverRatio": New
// failoverRatio value for the containing target pool.
func (c *TargetPoolsSetBackupCall) FailoverRatio(failoverRatio float64) *TargetPoolsSetBackupCall {
	c.opt_["failoverRatio"] = failoverRatio
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetPoolsSetBackupCall) Fields(s ...googleapi.Field) *TargetPoolsSetBackupCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetPoolsSetBackupCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.targetreference)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["failoverRatio"]; ok {
		params.Set("failoverRatio", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/targetPools/{targetPool}/setBackup")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"region":     c.region,
		"targetPool": c.targetPool,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Changes backup pool configurations.",
	//   "httpMethod": "POST",
	//   "id": "compute.targetPools.setBackup",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "targetPool"
	//   ],
	//   "parameters": {
	//     "failoverRatio": {
	//       "description": "New failoverRatio value for the containing target pool.",
	//       "format": "float",
	//       "location": "query",
	//       "type": "number"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Name of the region scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetPool": {
	//       "description": "Name of the TargetPool resource for which the backup is to be set.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetPools/{targetPool}/setBackup",
	//   "request": {
	//     "$ref": "TargetReference"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetVpnGateways.aggregatedList":

type TargetVpnGatewaysAggregatedListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// AggregatedList: Retrieves the list of target VPN gateways grouped by
// scope.
func (r *TargetVpnGatewaysService) AggregatedList(project string) *TargetVpnGatewaysAggregatedListCall {
	c := &TargetVpnGatewaysAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *TargetVpnGatewaysAggregatedListCall) Filter(filter string) *TargetVpnGatewaysAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *TargetVpnGatewaysAggregatedListCall) MaxResults(maxResults int64) *TargetVpnGatewaysAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *TargetVpnGatewaysAggregatedListCall) PageToken(pageToken string) *TargetVpnGatewaysAggregatedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetVpnGatewaysAggregatedListCall) Fields(s ...googleapi.Field) *TargetVpnGatewaysAggregatedListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetVpnGatewaysAggregatedListCall) Do() (*TargetVpnGatewayAggregatedList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/aggregated/targetVpnGateways")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *TargetVpnGatewayAggregatedList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of target VPN gateways grouped by scope.",
	//   "httpMethod": "GET",
	//   "id": "compute.targetVpnGateways.aggregatedList",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/aggregated/targetVpnGateways",
	//   "response": {
	//     "$ref": "TargetVpnGatewayAggregatedList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.targetVpnGateways.delete":

type TargetVpnGatewaysDeleteCall struct {
	s                *Service
	project          string
	region           string
	targetVpnGateway string
	opt_             map[string]interface{}
}

// Delete: Deletes the specified TargetVpnGateway resource.
func (r *TargetVpnGatewaysService) Delete(project string, region string, targetVpnGateway string) *TargetVpnGatewaysDeleteCall {
	c := &TargetVpnGatewaysDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetVpnGateway = targetVpnGateway
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetVpnGatewaysDeleteCall) Fields(s ...googleapi.Field) *TargetVpnGatewaysDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetVpnGatewaysDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/targetVpnGateways/{targetVpnGateway}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":          c.project,
		"region":           c.region,
		"targetVpnGateway": c.targetVpnGateway,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified TargetVpnGateway resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.targetVpnGateways.delete",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "targetVpnGateway"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The name of the region for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetVpnGateway": {
	//       "description": "Name of the TargetVpnGateway resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetVpnGateways/{targetVpnGateway}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetVpnGateways.get":

type TargetVpnGatewaysGetCall struct {
	s                *Service
	project          string
	region           string
	targetVpnGateway string
	opt_             map[string]interface{}
}

// Get: Returns the specified TargetVpnGateway resource.
func (r *TargetVpnGatewaysService) Get(project string, region string, targetVpnGateway string) *TargetVpnGatewaysGetCall {
	c := &TargetVpnGatewaysGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetVpnGateway = targetVpnGateway
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetVpnGatewaysGetCall) Fields(s ...googleapi.Field) *TargetVpnGatewaysGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetVpnGatewaysGetCall) Do() (*TargetVpnGateway, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/targetVpnGateways/{targetVpnGateway}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":          c.project,
		"region":           c.region,
		"targetVpnGateway": c.targetVpnGateway,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *TargetVpnGateway
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified TargetVpnGateway resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.targetVpnGateways.get",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "targetVpnGateway"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The name of the region for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "targetVpnGateway": {
	//       "description": "Name of the TargetVpnGateway resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetVpnGateways/{targetVpnGateway}",
	//   "response": {
	//     "$ref": "TargetVpnGateway"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.targetVpnGateways.insert":

type TargetVpnGatewaysInsertCall struct {
	s                *Service
	project          string
	region           string
	targetvpngateway *TargetVpnGateway
	opt_             map[string]interface{}
}

// Insert: Creates a TargetVpnGateway resource in the specified project
// and region using the data included in the request.
func (r *TargetVpnGatewaysService) Insert(project string, region string, targetvpngateway *TargetVpnGateway) *TargetVpnGatewaysInsertCall {
	c := &TargetVpnGatewaysInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetvpngateway = targetvpngateway
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetVpnGatewaysInsertCall) Fields(s ...googleapi.Field) *TargetVpnGatewaysInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetVpnGatewaysInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.targetvpngateway)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/targetVpnGateways")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"region":  c.region,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a TargetVpnGateway resource in the specified project and region using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.targetVpnGateways.insert",
	//   "parameterOrder": [
	//     "project",
	//     "region"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The name of the region for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetVpnGateways",
	//   "request": {
	//     "$ref": "TargetVpnGateway"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetVpnGateways.list":

type TargetVpnGatewaysListCall struct {
	s       *Service
	project string
	region  string
	opt_    map[string]interface{}
}

// List: Retrieves the list of TargetVpnGateway resources available to
// the specified project and region.
func (r *TargetVpnGatewaysService) List(project string, region string) *TargetVpnGatewaysListCall {
	c := &TargetVpnGatewaysListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *TargetVpnGatewaysListCall) Filter(filter string) *TargetVpnGatewaysListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *TargetVpnGatewaysListCall) MaxResults(maxResults int64) *TargetVpnGatewaysListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *TargetVpnGatewaysListCall) PageToken(pageToken string) *TargetVpnGatewaysListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TargetVpnGatewaysListCall) Fields(s ...googleapi.Field) *TargetVpnGatewaysListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *TargetVpnGatewaysListCall) Do() (*TargetVpnGatewayList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/targetVpnGateways")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"region":  c.region,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *TargetVpnGatewayList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of TargetVpnGateway resources available to the specified project and region.",
	//   "httpMethod": "GET",
	//   "id": "compute.targetVpnGateways.list",
	//   "parameterOrder": [
	//     "project",
	//     "region"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The name of the region for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetVpnGateways",
	//   "response": {
	//     "$ref": "TargetVpnGatewayList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.urlMaps.delete":

type UrlMapsDeleteCall struct {
	s       *Service
	project string
	urlMap  string
	opt_    map[string]interface{}
}

// Delete: Deletes the specified UrlMap resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/urlMaps/delete
func (r *UrlMapsService) Delete(project string, urlMap string) *UrlMapsDeleteCall {
	c := &UrlMapsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.urlMap = urlMap
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UrlMapsDeleteCall) Fields(s ...googleapi.Field) *UrlMapsDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *UrlMapsDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/urlMaps/{urlMap}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"urlMap":  c.urlMap,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified UrlMap resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.urlMaps.delete",
	//   "parameterOrder": [
	//     "project",
	//     "urlMap"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "urlMap": {
	//       "description": "Name of the UrlMap resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/urlMaps/{urlMap}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.urlMaps.get":

type UrlMapsGetCall struct {
	s       *Service
	project string
	urlMap  string
	opt_    map[string]interface{}
}

// Get: Returns the specified UrlMap resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/urlMaps/get
func (r *UrlMapsService) Get(project string, urlMap string) *UrlMapsGetCall {
	c := &UrlMapsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.urlMap = urlMap
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UrlMapsGetCall) Fields(s ...googleapi.Field) *UrlMapsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *UrlMapsGetCall) Do() (*UrlMap, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/urlMaps/{urlMap}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"urlMap":  c.urlMap,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *UrlMap
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified UrlMap resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.urlMaps.get",
	//   "parameterOrder": [
	//     "project",
	//     "urlMap"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "urlMap": {
	//       "description": "Name of the UrlMap resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/urlMaps/{urlMap}",
	//   "response": {
	//     "$ref": "UrlMap"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.urlMaps.insert":

type UrlMapsInsertCall struct {
	s       *Service
	project string
	urlmap  *UrlMap
	opt_    map[string]interface{}
}

// Insert: Creates a UrlMap resource in the specified project using the
// data included in the request.
// For details, see https://cloud.google.com/compute/docs/reference/latest/urlMaps/insert
func (r *UrlMapsService) Insert(project string, urlmap *UrlMap) *UrlMapsInsertCall {
	c := &UrlMapsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.urlmap = urlmap
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UrlMapsInsertCall) Fields(s ...googleapi.Field) *UrlMapsInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *UrlMapsInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.urlmap)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/urlMaps")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a UrlMap resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.urlMaps.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/urlMaps",
	//   "request": {
	//     "$ref": "UrlMap"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.urlMaps.list":

type UrlMapsListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of UrlMap resources available to the
// specified project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/urlMaps/list
func (r *UrlMapsService) List(project string) *UrlMapsListCall {
	c := &UrlMapsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *UrlMapsListCall) Filter(filter string) *UrlMapsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *UrlMapsListCall) MaxResults(maxResults int64) *UrlMapsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *UrlMapsListCall) PageToken(pageToken string) *UrlMapsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UrlMapsListCall) Fields(s ...googleapi.Field) *UrlMapsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *UrlMapsListCall) Do() (*UrlMapList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/urlMaps")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *UrlMapList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of UrlMap resources available to the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.urlMaps.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/urlMaps",
	//   "response": {
	//     "$ref": "UrlMapList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.urlMaps.patch":

type UrlMapsPatchCall struct {
	s       *Service
	project string
	urlMap  string
	urlmap  *UrlMap
	opt_    map[string]interface{}
}

// Patch: Update the entire content of the UrlMap resource. This method
// supports patch semantics.
// For details, see https://cloud.google.com/compute/docs/reference/latest/urlMaps/patch
func (r *UrlMapsService) Patch(project string, urlMap string, urlmap *UrlMap) *UrlMapsPatchCall {
	c := &UrlMapsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.urlMap = urlMap
	c.urlmap = urlmap
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UrlMapsPatchCall) Fields(s ...googleapi.Field) *UrlMapsPatchCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *UrlMapsPatchCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.urlmap)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/urlMaps/{urlMap}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"urlMap":  c.urlMap,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update the entire content of the UrlMap resource. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "compute.urlMaps.patch",
	//   "parameterOrder": [
	//     "project",
	//     "urlMap"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "urlMap": {
	//       "description": "Name of the UrlMap resource to update.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/urlMaps/{urlMap}",
	//   "request": {
	//     "$ref": "UrlMap"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.urlMaps.update":

type UrlMapsUpdateCall struct {
	s       *Service
	project string
	urlMap  string
	urlmap  *UrlMap
	opt_    map[string]interface{}
}

// Update: Update the entire content of the UrlMap resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/urlMaps/update
func (r *UrlMapsService) Update(project string, urlMap string, urlmap *UrlMap) *UrlMapsUpdateCall {
	c := &UrlMapsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.urlMap = urlMap
	c.urlmap = urlmap
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UrlMapsUpdateCall) Fields(s ...googleapi.Field) *UrlMapsUpdateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *UrlMapsUpdateCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.urlmap)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/urlMaps/{urlMap}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"urlMap":  c.urlMap,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update the entire content of the UrlMap resource.",
	//   "httpMethod": "PUT",
	//   "id": "compute.urlMaps.update",
	//   "parameterOrder": [
	//     "project",
	//     "urlMap"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "urlMap": {
	//       "description": "Name of the UrlMap resource to update.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/urlMaps/{urlMap}",
	//   "request": {
	//     "$ref": "UrlMap"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.urlMaps.validate":

type UrlMapsValidateCall struct {
	s                      *Service
	project                string
	urlMap                 string
	urlmapsvalidaterequest *UrlMapsValidateRequest
	opt_                   map[string]interface{}
}

// Validate: Run static validation for the UrlMap. In particular, the
// tests of the provided UrlMap will be run. Calling this method does
// NOT create the UrlMap.
// For details, see https://cloud.google.com/compute/docs/reference/latest/urlMaps/validate
func (r *UrlMapsService) Validate(project string, urlMap string, urlmapsvalidaterequest *UrlMapsValidateRequest) *UrlMapsValidateCall {
	c := &UrlMapsValidateCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.urlMap = urlMap
	c.urlmapsvalidaterequest = urlmapsvalidaterequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UrlMapsValidateCall) Fields(s ...googleapi.Field) *UrlMapsValidateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *UrlMapsValidateCall) Do() (*UrlMapsValidateResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.urlmapsvalidaterequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/urlMaps/{urlMap}/validate")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"urlMap":  c.urlMap,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *UrlMapsValidateResponse
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Run static validation for the UrlMap. In particular, the tests of the provided UrlMap will be run. Calling this method does NOT create the UrlMap.",
	//   "httpMethod": "POST",
	//   "id": "compute.urlMaps.validate",
	//   "parameterOrder": [
	//     "project",
	//     "urlMap"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "urlMap": {
	//       "description": "Name of the UrlMap resource to be validated as.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/urlMaps/{urlMap}/validate",
	//   "request": {
	//     "$ref": "UrlMapsValidateRequest"
	//   },
	//   "response": {
	//     "$ref": "UrlMapsValidateResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.vpnTunnels.aggregatedList":

type VpnTunnelsAggregatedListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// AggregatedList: Retrieves the list of VPN tunnels grouped by scope.
func (r *VpnTunnelsService) AggregatedList(project string) *VpnTunnelsAggregatedListCall {
	c := &VpnTunnelsAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *VpnTunnelsAggregatedListCall) Filter(filter string) *VpnTunnelsAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *VpnTunnelsAggregatedListCall) MaxResults(maxResults int64) *VpnTunnelsAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *VpnTunnelsAggregatedListCall) PageToken(pageToken string) *VpnTunnelsAggregatedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VpnTunnelsAggregatedListCall) Fields(s ...googleapi.Field) *VpnTunnelsAggregatedListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *VpnTunnelsAggregatedListCall) Do() (*VpnTunnelAggregatedList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/aggregated/vpnTunnels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *VpnTunnelAggregatedList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of VPN tunnels grouped by scope.",
	//   "httpMethod": "GET",
	//   "id": "compute.vpnTunnels.aggregatedList",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/aggregated/vpnTunnels",
	//   "response": {
	//     "$ref": "VpnTunnelAggregatedList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.vpnTunnels.delete":

type VpnTunnelsDeleteCall struct {
	s         *Service
	project   string
	region    string
	vpnTunnel string
	opt_      map[string]interface{}
}

// Delete: Deletes the specified VpnTunnel resource.
func (r *VpnTunnelsService) Delete(project string, region string, vpnTunnel string) *VpnTunnelsDeleteCall {
	c := &VpnTunnelsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.vpnTunnel = vpnTunnel
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VpnTunnelsDeleteCall) Fields(s ...googleapi.Field) *VpnTunnelsDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *VpnTunnelsDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/vpnTunnels/{vpnTunnel}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"region":    c.region,
		"vpnTunnel": c.vpnTunnel,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified VpnTunnel resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.vpnTunnels.delete",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "vpnTunnel"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The name of the region for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "vpnTunnel": {
	//       "description": "Name of the VpnTunnel resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/vpnTunnels/{vpnTunnel}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.vpnTunnels.get":

type VpnTunnelsGetCall struct {
	s         *Service
	project   string
	region    string
	vpnTunnel string
	opt_      map[string]interface{}
}

// Get: Returns the specified VpnTunnel resource.
func (r *VpnTunnelsService) Get(project string, region string, vpnTunnel string) *VpnTunnelsGetCall {
	c := &VpnTunnelsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.vpnTunnel = vpnTunnel
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VpnTunnelsGetCall) Fields(s ...googleapi.Field) *VpnTunnelsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *VpnTunnelsGetCall) Do() (*VpnTunnel, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/vpnTunnels/{vpnTunnel}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"region":    c.region,
		"vpnTunnel": c.vpnTunnel,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *VpnTunnel
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified VpnTunnel resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.vpnTunnels.get",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "vpnTunnel"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The name of the region for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "vpnTunnel": {
	//       "description": "Name of the VpnTunnel resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/vpnTunnels/{vpnTunnel}",
	//   "response": {
	//     "$ref": "VpnTunnel"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.vpnTunnels.insert":

type VpnTunnelsInsertCall struct {
	s         *Service
	project   string
	region    string
	vpntunnel *VpnTunnel
	opt_      map[string]interface{}
}

// Insert: Creates a VpnTunnel resource in the specified project and
// region using the data included in the request.
func (r *VpnTunnelsService) Insert(project string, region string, vpntunnel *VpnTunnel) *VpnTunnelsInsertCall {
	c := &VpnTunnelsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.vpntunnel = vpntunnel
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VpnTunnelsInsertCall) Fields(s ...googleapi.Field) *VpnTunnelsInsertCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *VpnTunnelsInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.vpntunnel)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/vpnTunnels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"region":  c.region,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a VpnTunnel resource in the specified project and region using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.vpnTunnels.insert",
	//   "parameterOrder": [
	//     "project",
	//     "region"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The name of the region for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/vpnTunnels",
	//   "request": {
	//     "$ref": "VpnTunnel"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.vpnTunnels.list":

type VpnTunnelsListCall struct {
	s       *Service
	project string
	region  string
	opt_    map[string]interface{}
}

// List: Retrieves the list of VpnTunnel resources contained in the
// specified project and region.
func (r *VpnTunnelsService) List(project string, region string) *VpnTunnelsListCall {
	c := &VpnTunnelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *VpnTunnelsListCall) Filter(filter string) *VpnTunnelsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *VpnTunnelsListCall) MaxResults(maxResults int64) *VpnTunnelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *VpnTunnelsListCall) PageToken(pageToken string) *VpnTunnelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VpnTunnelsListCall) Fields(s ...googleapi.Field) *VpnTunnelsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *VpnTunnelsListCall) Do() (*VpnTunnelList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/regions/{region}/vpnTunnels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"region":  c.region,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *VpnTunnelList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of VpnTunnel resources contained in the specified project and region.",
	//   "httpMethod": "GET",
	//   "id": "compute.vpnTunnels.list",
	//   "parameterOrder": [
	//     "project",
	//     "region"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "The name of the region for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/vpnTunnels",
	//   "response": {
	//     "$ref": "VpnTunnelList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.zoneOperations.delete":

type ZoneOperationsDeleteCall struct {
	s         *Service
	project   string
	zone      string
	operation string
	opt_      map[string]interface{}
}

// Delete: Deletes the specified zone-specific Operations resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/zoneOperations/delete
func (r *ZoneOperationsService) Delete(project string, zone string, operation string) *ZoneOperationsDeleteCall {
	c := &ZoneOperationsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.operation = operation
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZoneOperationsDeleteCall) Fields(s ...googleapi.Field) *ZoneOperationsDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ZoneOperationsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/operations/{operation}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"zone":      c.zone,
		"operation": c.operation,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the specified zone-specific Operations resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.zoneOperations.delete",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the Operations resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/operations/{operation}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.zoneOperations.get":

type ZoneOperationsGetCall struct {
	s         *Service
	project   string
	zone      string
	operation string
	opt_      map[string]interface{}
}

// Get: Retrieves the specified zone-specific Operations resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/zoneOperations/get
func (r *ZoneOperationsService) Get(project string, zone string, operation string) *ZoneOperationsGetCall {
	c := &ZoneOperationsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.operation = operation
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZoneOperationsGetCall) Fields(s ...googleapi.Field) *ZoneOperationsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ZoneOperationsGetCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/operations/{operation}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"zone":      c.zone,
		"operation": c.operation,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the specified zone-specific Operations resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.zoneOperations.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the Operations resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/operations/{operation}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.zoneOperations.list":

type ZoneOperationsListCall struct {
	s       *Service
	project string
	zone    string
	opt_    map[string]interface{}
}

// List: Retrieves the list of Operation resources contained within the
// specified zone.
// For details, see https://cloud.google.com/compute/docs/reference/latest/zoneOperations/list
func (r *ZoneOperationsService) List(project string, zone string) *ZoneOperationsListCall {
	c := &ZoneOperationsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *ZoneOperationsListCall) Filter(filter string) *ZoneOperationsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *ZoneOperationsListCall) MaxResults(maxResults int64) *ZoneOperationsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *ZoneOperationsListCall) PageToken(pageToken string) *ZoneOperationsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZoneOperationsListCall) Fields(s ...googleapi.Field) *ZoneOperationsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ZoneOperationsListCall) Do() (*OperationList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/operations")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *OperationList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of Operation resources contained within the specified zone.",
	//   "httpMethod": "GET",
	//   "id": "compute.zoneOperations.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/operations",
	//   "response": {
	//     "$ref": "OperationList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.zones.get":

type ZonesGetCall struct {
	s       *Service
	project string
	zone    string
	opt_    map[string]interface{}
}

// Get: Returns the specified zone resource.
// For details, see https://cloud.google.com/compute/docs/reference/latest/zones/get
func (r *ZonesService) Get(project string, zone string) *ZonesGetCall {
	c := &ZonesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZonesGetCall) Fields(s ...googleapi.Field) *ZonesGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ZonesGetCall) Do() (*Zone, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Zone
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified zone resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.zones.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}",
	//   "response": {
	//     "$ref": "Zone"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.zones.list":

type ZonesListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of zone resources available to the specified
// project.
// For details, see https://cloud.google.com/compute/docs/reference/latest/zones/list
func (r *ZonesService) List(project string) *ZonesListCall {
	c := &ZonesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must contain the following:
// FIELD_NAME COMPARISON_STRING LITERAL_STRING
//
// - FIELD_NAME: The name of the field you want to compare. The field
// name must be valid for the type of resource being filtered. Only
// atomic field types are supported (string, number, boolean). Array and
// object fields are not currently supported.
// - COMPARISON_STRING: The comparison string, either eq (equals) or ne
// (not equals).
// - LITERAL_STRING: The literal string value to filter to. The literal
// value must be valid for the type of field (string, number, boolean).
// For string fields, the literal value is interpreted as a regular
// expression using RE2 syntax. The literal value must match the entire
// field.  For example, you can filter by the name of a
// resource:
// filter=name ne example-instance
// The above filter returns only results whose name field does not equal
// example-instance. You can also enclose your literal string in single,
// double, or no quotes.
func (c *ZonesListCall) Filter(filter string) *ZonesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned.
func (c *ZonesListCall) MaxResults(maxResults int64) *ZonesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Use this parameter if you want to list the next page of
// results. Set pageToken to the nextPageToken returned by a previous
// list request.
func (c *ZonesListCall) PageToken(pageToken string) *ZonesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZonesListCall) Fields(s ...googleapi.Field) *ZonesListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ZonesListCall) Do() (*ZoneList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ZoneList
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of zone resources available to the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.zones.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must contain the following:\nFIELD_NAME COMPARISON_STRING LITERAL_STRING\n \n- FIELD_NAME: The name of the field you want to compare. The field name must be valid for the type of resource being filtered. Only atomic field types are supported (string, number, boolean). Array and object fields are not currently supported. \n- COMPARISON_STRING: The comparison string, either eq (equals) or ne (not equals). \n- LITERAL_STRING: The literal string value to filter to. The literal value must be valid for the type of field (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.  For example, you can filter by the name of a resource:\nfilter=name ne example-instance\nThe above filter returns only results whose name field does not equal example-instance. You can also enclose your literal string in single, double, or no quotes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Maximum count of results to be returned.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Use this parameter if you want to list the next page of results. Set pageToken to the nextPageToken returned by a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones",
	//   "response": {
	//     "$ref": "ZoneList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}
