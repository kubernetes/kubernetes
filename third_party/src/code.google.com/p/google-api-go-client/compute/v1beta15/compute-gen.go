// Package compute provides access to the Compute Engine API.
//
// See https://developers.google.com/compute/docs/reference/v1beta15
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/compute/v1beta15"
//   ...
//   computeService, err := compute.New(oauthHttpClient)
package compute

import (
	"bytes"
	"code.google.com/p/google-api-go-client/googleapi"
	"encoding/json"
	"errors"
	"fmt"
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

const apiId = "compute:v1beta15"
const apiName = "compute"
const apiVersion = "v1beta15"
const basePath = "https://www.googleapis.com/compute/v1beta15/projects/"

// OAuth2 scopes used by this API.
const (
	// View and manage your Google Compute Engine resources
	ComputeScope = "https://www.googleapis.com/auth/compute"

	// View your Google Compute Engine resources
	ComputeReadonlyScope = "https://www.googleapis.com/auth/compute.readonly"

	// Manage your data and permissions in Google Cloud Storage
	DevstorageFull_controlScope = "https://www.googleapis.com/auth/devstorage.full_control"

	// View your data in Google Cloud Storage
	DevstorageRead_onlyScope = "https://www.googleapis.com/auth/devstorage.read_only"

	// Manage your data in Google Cloud Storage
	DevstorageRead_writeScope = "https://www.googleapis.com/auth/devstorage.read_write"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client}
	s.Addresses = NewAddressesService(s)
	s.Disks = NewDisksService(s)
	s.Firewalls = NewFirewallsService(s)
	s.ForwardingRules = NewForwardingRulesService(s)
	s.GlobalOperations = NewGlobalOperationsService(s)
	s.HttpHealthChecks = NewHttpHealthChecksService(s)
	s.Images = NewImagesService(s)
	s.Instances = NewInstancesService(s)
	s.Kernels = NewKernelsService(s)
	s.MachineTypes = NewMachineTypesService(s)
	s.Networks = NewNetworksService(s)
	s.Projects = NewProjectsService(s)
	s.RegionOperations = NewRegionOperationsService(s)
	s.Regions = NewRegionsService(s)
	s.Routes = NewRoutesService(s)
	s.Snapshots = NewSnapshotsService(s)
	s.TargetPools = NewTargetPoolsService(s)
	s.ZoneOperations = NewZoneOperationsService(s)
	s.Zones = NewZonesService(s)
	return s, nil
}

type Service struct {
	client *http.Client

	Addresses *AddressesService

	Disks *DisksService

	Firewalls *FirewallsService

	ForwardingRules *ForwardingRulesService

	GlobalOperations *GlobalOperationsService

	HttpHealthChecks *HttpHealthChecksService

	Images *ImagesService

	Instances *InstancesService

	Kernels *KernelsService

	MachineTypes *MachineTypesService

	Networks *NetworksService

	Projects *ProjectsService

	RegionOperations *RegionOperationsService

	Regions *RegionsService

	Routes *RoutesService

	Snapshots *SnapshotsService

	TargetPools *TargetPoolsService

	ZoneOperations *ZoneOperationsService

	Zones *ZonesService
}

func NewAddressesService(s *Service) *AddressesService {
	rs := &AddressesService{s: s}
	return rs
}

type AddressesService struct {
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

func NewInstancesService(s *Service) *InstancesService {
	rs := &InstancesService{s: s}
	return rs
}

type InstancesService struct {
	s *Service
}

func NewKernelsService(s *Service) *KernelsService {
	rs := &KernelsService{s: s}
	return rs
}

type KernelsService struct {
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

func NewTargetPoolsService(s *Service) *TargetPoolsService {
	rs := &TargetPoolsService{s: s}
	return rs
}

type TargetPoolsService struct {
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

type AccessConfig struct {
	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Name: Name of this access configuration.
	Name string `json:"name,omitempty"`

	// NatIP: An external IP address associated with this instance. Specify
	// an unused static IP address available to the project. If not
	// specified, the external IP will be drawn from a shared ephemeral
	// pool.
	NatIP string `json:"natIP,omitempty"`

	// Type: Type of configuration. Must be set to "ONE_TO_ONE_NAT". This
	// configures port-for-port NAT to the internet.
	Type string `json:"type,omitempty"`
}

type Address struct {
	// Address: The IP address represented by this resource.
	Address string `json:"address,omitempty"`

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

	// Region: URL of the region where the address resides (output only).
	Region string `json:"region,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// Status: The status of the address (output only).
	Status string `json:"status,omitempty"`

	User string `json:"user,omitempty"`
}

type AddressAggregatedList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A map of scoped address lists.
	Items *AddressAggregatedListItems `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type AddressAggregatedListItems struct {
}

type AddressList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The address resources.
	Items []*Address `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type AddressesScopedList struct {
	// Addresses: List of addresses contained in this scope.
	Addresses []*Address `json:"addresses,omitempty"`

	// Warning: Informational warning which replaces the list of addresses
	// when the list is empty.
	Warning *AddressesScopedListWarning `json:"warning,omitempty"`
}

type AddressesScopedListWarning struct {
	// Code: The warning type identifier for this warning.
	Code string `json:"code,omitempty"`

	// Data: Metadata for this warning in 'key: value' format.
	Data []*AddressesScopedListWarningData `json:"data,omitempty"`

	// Message: Optional human-readable details for this warning.
	Message string `json:"message,omitempty"`
}

type AddressesScopedListWarningData struct {
	// Key: A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type AttachedDisk struct {
	// Boot: Indicates that this is a boot disk. VM will use the first
	// partition of the disk for its root filesystem.
	Boot bool `json:"boot,omitempty"`

	// DeviceName: Persistent disk only; must be unique within the instance
	// when specified. This represents a unique device name that is
	// reflected into the /dev/ tree of a Linux operating system running
	// within the instance. If not specified, a default will be chosen by
	// the system.
	DeviceName string `json:"deviceName,omitempty"`

	// Index: A zero-based index to assign to this disk, where 0 is reserved
	// for the boot disk. If not specified, the server will choose an
	// appropriate value (output only).
	Index int64 `json:"index,omitempty"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Mode: The mode in which to attach this disk, either "READ_WRITE" or
	// "READ_ONLY".
	Mode string `json:"mode,omitempty"`

	// Source: Persistent disk only; the URL of the persistent disk
	// resource.
	Source string `json:"source,omitempty"`

	// Type: Type of the disk, either "SCRATCH" or "PERSISTENT". Note that
	// persistent disks must be created before you can specify them here.
	Type string `json:"type,omitempty"`
}

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

	// Replacement: A URL of the suggested replacement for the deprecated
	// resource. The deprecated resource and its replacement must be
	// resources of the same kind.
	Replacement string `json:"replacement,omitempty"`

	// State: The deprecation state. Can be "DEPRECATED", "OBSOLETE", or
	// "DELETED". Operations which create a new resource using a
	// "DEPRECATED" resource will return successfully, but with a warning
	// indicating the deprecated resource and recommending its replacement.
	// New uses of "OBSOLETE" or "DELETED" resources will result in an
	// error.
	State string `json:"state,omitempty"`
}

type Disk struct {
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

	// Options: Internal use only.
	Options string `json:"options,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// SizeGb: Size of the persistent disk, specified in GB. This parameter
	// is optional when creating a disk from a disk image or a snapshot,
	// otherwise it is required.
	SizeGb int64 `json:"sizeGb,omitempty,string"`

	// SourceImage: The source image used to create this disk. Once the
	// source image has been deleted from the system, this field will not be
	// set, even if an image with the same name has been re-created.
	SourceImage string `json:"sourceImage,omitempty"`

	// SourceImageId: The 'id' value of the image used to create this disk.
	// This value may be used to determine whether the disk was created from
	// the current or a previous instance of a given image.
	SourceImageId string `json:"sourceImageId,omitempty"`

	// SourceSnapshot: The source snapshot used to create this disk. Once
	// the source snapshot has been deleted from the system, this field will
	// be cleared, and will not be set even if a snapshot with the same name
	// has been re-created.
	SourceSnapshot string `json:"sourceSnapshot,omitempty"`

	// SourceSnapshotId: The 'id' value of the snapshot used to create this
	// disk. This value may be used to determine whether the disk was
	// created from the current or a previous instance of a given disk
	// snapshot.
	SourceSnapshotId string `json:"sourceSnapshotId,omitempty"`

	// Status: The status of disk creation (output only).
	Status string `json:"status,omitempty"`

	// Zone: URL of the zone where the disk resides (output only).
	Zone string `json:"zone,omitempty"`
}

type DiskAggregatedList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A map of scoped disk lists.
	Items *DiskAggregatedListItems `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type DiskAggregatedListItems struct {
}

type DiskList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The persistent disk resources.
	Items []*Disk `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type DisksScopedList struct {
	// Disks: List of disks contained in this scope.
	Disks []*Disk `json:"disks,omitempty"`

	// Warning: Informational warning which replaces the list of disks when
	// the list is empty.
	Warning *DisksScopedListWarning `json:"warning,omitempty"`
}

type DisksScopedListWarning struct {
	// Code: The warning type identifier for this warning.
	Code string `json:"code,omitempty"`

	// Data: Metadata for this warning in 'key: value' format.
	Data []*DisksScopedListWarningData `json:"data,omitempty"`

	// Message: Optional human-readable details for this warning.
	Message string `json:"message,omitempty"`
}

type DisksScopedListWarningData struct {
	// Key: A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type Firewall struct {
	// Allowed: The list of rules specified by this firewall. Each rule
	// specifies a protocol and port-range tuple that describes a permitted
	// connection.
	Allowed []*FirewallAllowed `json:"allowed,omitempty"`

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

	// Network: URL of the network to which this firewall is applied;
	// provided by the client when the firewall is created.
	Network string `json:"network,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// SourceRanges: A list of IP address blocks expressed in CIDR format
	// which this rule applies to. One or both of sourceRanges and
	// sourceTags may be set; an inbound connection is allowed if either the
	// range or the tag of the source matches.
	SourceRanges []string `json:"sourceRanges,omitempty"`

	// SourceTags: A list of instance tags which this rule applies to. One
	// or both of sourceRanges and sourceTags may be set; an inbound
	// connection is allowed if either the range or the tag of the source
	// matches.
	SourceTags []string `json:"sourceTags,omitempty"`

	// TargetTags: A list of instance tags indicating sets of instances
	// located on network which may make network connections as specified in
	// allowed. If no targetTags are specified, the firewall rule applies to
	// all instances on the specified network.
	TargetTags []string `json:"targetTags,omitempty"`
}

type FirewallAllowed struct {
	// IPProtocol: Required; this is the IP protocol that is allowed for
	// this rule. This can either be a well known protocol string (tcp, udp
	// or icmp) or the IP protocol number.
	IPProtocol string `json:"IPProtocol,omitempty"`

	// Ports: An optional list of ports which are allowed. It is an error to
	// specify this for any protocol that isn't UDP or TCP. Each entry must
	// be either an integer or a range. If not specified, connections
	// through any port are allowed.
	//
	// Example inputs include: ["22"],
	// ["80","443"] and ["12345-12349"].
	Ports []string `json:"ports,omitempty"`
}

type FirewallList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The firewall resources.
	Items []*Firewall `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type ForwardingRule struct {
	// IPAddress: Value of the reserved IP address that this forwarding rule
	// is serving on behalf of. The address resource must live in the same
	// region as the forwarding rule. If left empty (default value), an
	// ephemeral IP will be assigned.
	IPAddress string `json:"IPAddress,omitempty"`

	// IPProtocol: The IP protocol to which this rule applies, can be either
	// 'TCP' or 'UDP' (If left empty, will use TCP by default).
	IPProtocol string `json:"IPProtocol,omitempty"`

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

	// PortRange: If 'IPProtocol' is 'TCP' or 'UDP', only packets addressed
	// to ports in the specified range will be forwarded to 'target'. If
	// left empty (default value), all ports are forwarded. Forwarding rules
	// with the same [IPAddress, IPProtocol] pair must have disjoint port
	// ranges.
	PortRange string `json:"portRange,omitempty"`

	// Region: URL of the region where the forwarding rule resides (output
	// only).
	Region string `json:"region,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// Target: The URL of the target resource to receive the matched
	// traffic. It must live in the same region as this forwarding rule.
	Target string `json:"target,omitempty"`
}

type ForwardingRuleAggregatedList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A map of scoped forwarding rule lists.
	Items *ForwardingRuleAggregatedListItems `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type ForwardingRuleAggregatedListItems struct {
}

type ForwardingRuleList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The ForwardingRule resources.
	Items []*ForwardingRule `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type ForwardingRulesScopedList struct {
	// ForwardingRules: List of forwarding rules contained in this scope.
	ForwardingRules []*ForwardingRule `json:"forwardingRules,omitempty"`

	// Warning: Informational warning which replaces the list of forwarding
	// rules when the list is empty.
	Warning *ForwardingRulesScopedListWarning `json:"warning,omitempty"`
}

type ForwardingRulesScopedListWarning struct {
	// Code: The warning type identifier for this warning.
	Code string `json:"code,omitempty"`

	// Data: Metadata for this warning in 'key: value' format.
	Data []*ForwardingRulesScopedListWarningData `json:"data,omitempty"`

	// Message: Optional human-readable details for this warning.
	Message string `json:"message,omitempty"`
}

type ForwardingRulesScopedListWarningData struct {
	// Key: A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type HealthCheckReference struct {
	HealthCheck string `json:"healthCheck,omitempty"`
}

type HealthStatus struct {
	// HealthState: Health state of the instance.
	HealthState string `json:"healthState,omitempty"`

	// Instance: URL of the instance resource.
	Instance string `json:"instance,omitempty"`

	// IpAddress: The IP address represented by this resource.
	IpAddress string `json:"ipAddress,omitempty"`
}

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
	// The default value is 5 seconds.
	TimeoutSec int64 `json:"timeoutSec,omitempty"`

	// UnhealthyThreshold: A so-far healthy VM will be marked unhealthy
	// after this many consecutive failures. The default value is 2.
	UnhealthyThreshold int64 `json:"unhealthyThreshold,omitempty"`
}

type HttpHealthCheckList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The HttpHealthCheck resources.
	Items []*HttpHealthCheck `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type Image struct {
	// ArchiveSizeBytes: Size of the image tar.gz archive stored in Google
	// Cloud Storage (in bytes).
	ArchiveSizeBytes int64 `json:"archiveSizeBytes,omitempty,string"`

	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Deprecated: The deprecation status associated with this image.
	Deprecated *DeprecationStatus `json:"deprecated,omitempty"`

	// Description: Textual description of the resource; provided by the
	// client when the resource is created.
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

	// PreferredKernel: An optional URL of the preferred kernel for use with
	// this disk image. If not specified, a server defined default kernel
	// will be used.
	PreferredKernel string `json:"preferredKernel,omitempty"`

	// RawDisk: The raw disk image parameters.
	RawDisk *ImageRawDisk `json:"rawDisk,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// SourceType: Must be "RAW"; provided by the client when the disk image
	// is created.
	SourceType string `json:"sourceType,omitempty"`

	// Status: Status of the image (output only). It will be one of the
	// following READY - after image has been successfully created and is
	// ready for use FAILED - if creating the image fails for some reason
	// PENDING - the image creation is in progress An image can be used to
	// create other resources suck as instances only after the image has
	// been successfully created and the status is set to READY.
	Status string `json:"status,omitempty"`
}

type ImageRawDisk struct {
	// ContainerType: The format used to encode and transmit the block
	// device. Should be TAR. This is just a container and transmission
	// format and not a runtime format. Provided by the client when the disk
	// image is created.
	ContainerType string `json:"containerType,omitempty"`

	// Sha1Checksum: An optional SHA1 checksum of the disk image before
	// unpackaging; provided by the client when the disk image is created.
	Sha1Checksum string `json:"sha1Checksum,omitempty"`

	// Source: The full Google Cloud Storage URL where the disk image is
	// stored; provided by the client when the disk image is created.
	Source string `json:"source,omitempty"`
}

type ImageList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The disk image resources.
	Items []*Image `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type Instance struct {
	// CanIpForward: Allows this instance to send packets with source IP
	// addresses other than its own and receive packets with destination IP
	// addresses other than its own. If this instance will be used as an IP
	// gateway or it will be set as the next-hop in a Route resource, say
	// true. If unsure, leave this set to false.
	CanIpForward bool `json:"canIpForward,omitempty"`

	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// Disks: Array of disks associated with this instance. Persistent disks
	// must be created before you can assign them.
	Disks []*AttachedDisk `json:"disks,omitempty"`

	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id uint64 `json:"id,omitempty,string"`

	// Image: An optional URL of the disk image resource to be installed on
	// this instance; provided by the client when the instance is created.
	// Alternatively to passing the image, the client may choose to boot
	// from a persistent disk, by setting boot=true flag on one of the
	// entries in disks[] collection.
	Image string `json:"image,omitempty"`

	// Kernel: URL of the kernel resource to use when booting. In case of
	// booting from persistent disk, this parameter is required. When
	// booting from a disk image, it is optional, but may be provided to use
	// a different kernel than the one associated with the image.
	Kernel string `json:"kernel,omitempty"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// MachineType: URL of the machine type resource describing which
	// machine type to use to host the instance; provided by the client when
	// the instance is created.
	MachineType string `json:"machineType,omitempty"`

	// Metadata: Metadata key/value pairs assigned to this instance.
	// Consists of custom metadata or predefined keys; see Instance
	// documentation for more information.
	Metadata *Metadata `json:"metadata,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035.
	Name string `json:"name,omitempty"`

	// NetworkInterfaces: Array of configurations for this interface. This
	// specifies how this interface is configured to interact with other
	// network services, such as connecting to the internet. Currently,
	// ONE_TO_ONE_NAT is the only access config supported. If there are no
	// accessConfigs specified, then this instance will have no external
	// internet access.
	NetworkInterfaces []*NetworkInterface `json:"networkInterfaces,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// ServiceAccounts: A list of service accounts each with specified
	// scopes, for which access tokens are to be made available to the
	// instance through metadata queries.
	ServiceAccounts []*ServiceAccount `json:"serviceAccounts,omitempty"`

	// Status: Instance status. One of the following values: "PROVISIONING",
	// "STAGING", "RUNNING", "STOPPING", "STOPPED", "TERMINATED" (output
	// only).
	Status string `json:"status,omitempty"`

	// StatusMessage: An optional, human-readable explanation of the status
	// (output only).
	StatusMessage string `json:"statusMessage,omitempty"`

	// Tags: A list of tags to be applied to this instance. Used to identify
	// valid sources or targets for network firewalls. Provided by the
	// client on instance creation. The tags can be later modified by the
	// setTags method. Each tag within the list must comply with RFC1035.
	Tags *Tags `json:"tags,omitempty"`

	// Zone: URL of the zone where the instance resides (output only).
	Zone string `json:"zone,omitempty"`
}

type InstanceAggregatedList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A map of scoped instance lists.
	Items *InstanceAggregatedListItems `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type InstanceAggregatedListItems struct {
}

type InstanceList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A list of instance resources.
	Items []*Instance `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type InstanceReference struct {
	Instance string `json:"instance,omitempty"`
}

type InstancesScopedList struct {
	// Instances: List of instances contained in this scope.
	Instances []*Instance `json:"instances,omitempty"`

	// Warning: Informational warning which replaces the list of instances
	// when the list is empty.
	Warning *InstancesScopedListWarning `json:"warning,omitempty"`
}

type InstancesScopedListWarning struct {
	// Code: The warning type identifier for this warning.
	Code string `json:"code,omitempty"`

	// Data: Metadata for this warning in 'key: value' format.
	Data []*InstancesScopedListWarningData `json:"data,omitempty"`

	// Message: Optional human-readable details for this warning.
	Message string `json:"message,omitempty"`
}

type InstancesScopedListWarningData struct {
	// Key: A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type Kernel struct {
	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Deprecated: The deprecation status associated with this kernel.
	Deprecated *DeprecationStatus `json:"deprecated,omitempty"`

	// Description: An optional textual description of the resource.
	Description string `json:"description,omitempty"`

	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id uint64 `json:"id,omitempty,string"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource.
	Name string `json:"name,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type KernelList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The kernel resources.
	Items []*Kernel `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type MachineType struct {
	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Deprecated: The deprecation status associated with this machine type.
	Deprecated *DeprecationStatus `json:"deprecated,omitempty"`

	// Description: An optional textual description of the resource.
	Description string `json:"description,omitempty"`

	// GuestCpus: Count of CPUs exposed to the instance.
	GuestCpus int64 `json:"guestCpus,omitempty"`

	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id uint64 `json:"id,omitempty,string"`

	// ImageSpaceGb: Space allotted for the image, defined in GB.
	ImageSpaceGb int64 `json:"imageSpaceGb,omitempty"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// MaximumPersistentDisks: Maximum persistent disks allowed.
	MaximumPersistentDisks int64 `json:"maximumPersistentDisks,omitempty"`

	// MaximumPersistentDisksSizeGb: Maximum total persistent disks size
	// (GB) allowed.
	MaximumPersistentDisksSizeGb int64 `json:"maximumPersistentDisksSizeGb,omitempty,string"`

	// MemoryMb: Physical memory assigned to the instance, defined in MB.
	MemoryMb int64 `json:"memoryMb,omitempty"`

	// Name: Name of the resource.
	Name string `json:"name,omitempty"`

	// ScratchDisks: List of extended scratch disks assigned to the
	// instance.
	ScratchDisks []*MachineTypeScratchDisks `json:"scratchDisks,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// Zone: Url of the zone where the machine type resides (output only).
	Zone string `json:"zone,omitempty"`
}

type MachineTypeScratchDisks struct {
	// DiskGb: Size of the scratch disk, defined in GB.
	DiskGb int64 `json:"diskGb,omitempty"`
}

type MachineTypeAggregatedList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A map of scoped machine type lists.
	Items *MachineTypeAggregatedListItems `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type MachineTypeAggregatedListItems struct {
}

type MachineTypeList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The machine type resources.
	Items []*MachineType `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type MachineTypesScopedList struct {
	// MachineTypes: List of machine types contained in this scope.
	MachineTypes []*MachineType `json:"machineTypes,omitempty"`

	// Warning: Informational warning which replaces the list of machine
	// types when the list is empty.
	Warning *MachineTypesScopedListWarning `json:"warning,omitempty"`
}

type MachineTypesScopedListWarning struct {
	// Code: The warning type identifier for this warning.
	Code string `json:"code,omitempty"`

	// Data: Metadata for this warning in 'key: value' format.
	Data []*MachineTypesScopedListWarningData `json:"data,omitempty"`

	// Message: Optional human-readable details for this warning.
	Message string `json:"message,omitempty"`
}

type MachineTypesScopedListWarningData struct {
	// Key: A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type Metadata struct {
	// Fingerprint: Fingerprint of this resource. A hash of the metadata's
	// contents. This field is used for optimistic locking. An up-to-date
	// metadata fingerprint must be provided in order to modify metadata.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Items: Array of key/value pairs. The total size of all keys and
	// values must be less than 512 KB.
	Items []*MetadataItems `json:"items,omitempty"`

	// Kind: Type of the resource.
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

type Network struct {
	// IPv4Range: Required; The range of internal addresses that are legal
	// on this network. This range is a CIDR specification, for example:
	// 192.168.0.0/16. Provided by the client when the network is created.
	IPv4Range string `json:"IPv4Range,omitempty"`

	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// GatewayIPv4: An optional address that is used for default routing to
	// other networks. This must be within the range specified by IPv4Range,
	// and is typically the first usable address in that range. If not
	// specified, the default value is the first usable address in
	// IPv4Range.
	GatewayIPv4 string `json:"gatewayIPv4,omitempty"`

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
}

type NetworkInterface struct {
	// AccessConfigs: Array of configurations for this interface. This
	// specifies how this interface is configured to interact with other
	// network services, such as connecting to the internet. Currently,
	// ONE_TO_ONE_NAT is the only access config supported. If there are no
	// accessConfigs specified, then this instance will have no external
	// internet access.
	AccessConfigs []*AccessConfig `json:"accessConfigs,omitempty"`

	// Name: Name of the network interface, determined by the server; for
	// network devices, these are e.g. eth0, eth1, etc. (output only).
	Name string `json:"name,omitempty"`

	// Network: URL of the network resource attached to this interface.
	Network string `json:"network,omitempty"`

	// NetworkIP: An optional IPV4 internal network address assigned to the
	// instance for this network interface (output only).
	NetworkIP string `json:"networkIP,omitempty"`
}

type NetworkList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The network resources.
	Items []*Network `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type Operation struct {
	// ClientOperationId: An optional identifier specified by the client
	// when the mutation was initiated. Must be unique for all operation
	// resources in the project (output only).
	ClientOperationId string `json:"clientOperationId,omitempty"`

	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// EndTime: The time that this operation was completed. This is in RFC
	// 3339 format (output only).
	EndTime string `json:"endTime,omitempty"`

	// Error: If errors occurred during processing of this operation, this
	// field will be populated (output only).
	Error *OperationError `json:"error,omitempty"`

	// HttpErrorMessage: If operation fails, the HTTP error message
	// returned, e.g. NOT FOUND. (output only).
	HttpErrorMessage string `json:"httpErrorMessage,omitempty"`

	// HttpErrorStatusCode: If operation fails, the HTTP error status code
	// returned, e.g. 404. (output only).
	HttpErrorStatusCode int64 `json:"httpErrorStatusCode,omitempty"`

	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id uint64 `json:"id,omitempty,string"`

	// InsertTime: The time that this operation was requested. This is in
	// RFC 3339 format (output only).
	InsertTime string `json:"insertTime,omitempty"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource (output only).
	Name string `json:"name,omitempty"`

	// OperationType: Type of the operation. Examples include "insert",
	// "update", and "delete" (output only).
	OperationType string `json:"operationType,omitempty"`

	// Progress: An optional progress indicator that ranges from 0 to 100.
	// There is no requirement that this be linear or support any
	// granularity of operations. This should not be used to guess at when
	// the operation will be complete. This number should be monotonically
	// increasing as the operation progresses (output only).
	Progress int64 `json:"progress,omitempty"`

	// Region: URL of the region where the operation resides (output only).
	Region string `json:"region,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// StartTime: The time that this operation was started by the server.
	// This is in RFC 3339 format (output only).
	StartTime string `json:"startTime,omitempty"`

	// Status: Status of the operation. Can be one of the following:
	// "PENDING", "RUNNING", or "DONE" (output only).
	Status string `json:"status,omitempty"`

	// StatusMessage: An optional textual description of the current status
	// of the operation (output only).
	StatusMessage string `json:"statusMessage,omitempty"`

	// TargetId: Unique target id which identifies a particular incarnation
	// of the target (output only).
	TargetId uint64 `json:"targetId,omitempty,string"`

	// TargetLink: URL of the resource the operation is mutating (output
	// only).
	TargetLink string `json:"targetLink,omitempty"`

	// User: User who requested the operation, for example
	// "user@example.com" (output only).
	User string `json:"user,omitempty"`

	// Warnings: If warning messages generated during processing of this
	// operation, this field will be populated (output only).
	Warnings []*OperationWarnings `json:"warnings,omitempty"`

	// Zone: URL of the zone where the operation resides (output only).
	Zone string `json:"zone,omitempty"`
}

type OperationError struct {
	// Errors: The array of errors encountered while processing this
	// operation.
	Errors []*OperationErrorErrors `json:"errors,omitempty"`
}

type OperationErrorErrors struct {
	// Code: The error type identifier for this error.
	Code string `json:"code,omitempty"`

	// Location: Indicates the field in the request which caused the error.
	// This property is optional.
	Location string `json:"location,omitempty"`

	// Message: An optional, human-readable error message.
	Message string `json:"message,omitempty"`
}

type OperationWarnings struct {
	// Code: The warning type identifier for this warning.
	Code string `json:"code,omitempty"`

	// Data: Metadata for this warning in 'key: value' format.
	Data []*OperationWarningsData `json:"data,omitempty"`

	// Message: Optional human-readable details for this warning.
	Message string `json:"message,omitempty"`
}

type OperationWarningsData struct {
	// Key: A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type OperationAggregatedList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A map of scoped operation lists.
	Items *OperationAggregatedListItems `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type OperationAggregatedListItems struct {
}

type OperationList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The operation resources.
	Items []*Operation `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type OperationsScopedList struct {
	// Operations: List of operations contained in this scope.
	Operations []*Operation `json:"operations,omitempty"`

	// Warning: Informational warning which replaces the list of operations
	// when the list is empty.
	Warning *OperationsScopedListWarning `json:"warning,omitempty"`
}

type OperationsScopedListWarning struct {
	// Code: The warning type identifier for this warning.
	Code string `json:"code,omitempty"`

	// Data: Metadata for this warning in 'key: value' format.
	Data []*OperationsScopedListWarningData `json:"data,omitempty"`

	// Message: Optional human-readable details for this warning.
	Message string `json:"message,omitempty"`
}

type OperationsScopedListWarningData struct {
	// Key: A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type Project struct {
	// CommonInstanceMetadata: Metadata key/value pairs available to all
	// instances contained in this project.
	CommonInstanceMetadata *Metadata `json:"commonInstanceMetadata,omitempty"`

	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource.
	Description string `json:"description,omitempty"`

	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id uint64 `json:"id,omitempty,string"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource.
	Name string `json:"name,omitempty"`

	// Quotas: Quotas assigned to this project.
	Quotas []*Quota `json:"quotas,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type Quota struct {
	// Limit: Quota limit for this metric.
	Limit float64 `json:"limit,omitempty"`

	// Metric: Name of the quota metric.
	Metric string `json:"metric,omitempty"`

	// Usage: Current usage of this metric.
	Usage float64 `json:"usage,omitempty"`
}

type Region struct {
	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Deprecated: The deprecation status associated with this region.
	Deprecated *DeprecationStatus `json:"deprecated,omitempty"`

	// Description: Textual description of the resource.
	Description string `json:"description,omitempty"`

	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id uint64 `json:"id,omitempty,string"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource.
	Name string `json:"name,omitempty"`

	// Quotas: Quotas assigned to this region.
	Quotas []*Quota `json:"quotas,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// Status: Status of the region, "UP" or "DOWN".
	Status string `json:"status,omitempty"`

	// Zones: A list of zones homed in this region, in the form of resource
	// URLs.
	Zones []string `json:"zones,omitempty"`
}

type RegionList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The region resources.
	Items []*Region `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type Route struct {
	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// DestRange: Which packets does this route apply to?
	DestRange string `json:"destRange,omitempty"`

	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id uint64 `json:"id,omitempty,string"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created. The name must be 1-63 characters long, and comply with
	// RFC1035.
	Name string `json:"name,omitempty"`

	// Network: URL of the network to which this route is applied; provided
	// by the client when the route is created.
	Network string `json:"network,omitempty"`

	// NextHopGateway: The URL to a gateway that should handle matching
	// packets.
	NextHopGateway string `json:"nextHopGateway,omitempty"`

	// NextHopInstance: The URL to an instance that should handle matching
	// packets.
	NextHopInstance string `json:"nextHopInstance,omitempty"`

	// NextHopIp: The network IP address of an instance that should handle
	// matching packets.
	NextHopIp string `json:"nextHopIp,omitempty"`

	// NextHopNetwork: The URL of the local network if it should handle
	// matching packets.
	NextHopNetwork string `json:"nextHopNetwork,omitempty"`

	// Priority: Breaks ties between Routes of equal specificity. Routes
	// with smaller values win when tied with routes with larger values.
	Priority int64 `json:"priority,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// Tags: A list of instance tags to which this route applies.
	Tags []string `json:"tags,omitempty"`

	// Warnings: If potential misconfigurations are detected for this route,
	// this field will be populated with warning messages.
	Warnings []*RouteWarnings `json:"warnings,omitempty"`
}

type RouteWarnings struct {
	// Code: The warning type identifier for this warning.
	Code string `json:"code,omitempty"`

	// Data: Metadata for this warning in 'key: value' format.
	Data []*RouteWarningsData `json:"data,omitempty"`

	// Message: Optional human-readable details for this warning.
	Message string `json:"message,omitempty"`
}

type RouteWarningsData struct {
	// Key: A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type RouteList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The route resources.
	Items []*Route `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type SerialPortOutput struct {
	// Contents: The contents of the console output.
	Contents string `json:"contents,omitempty"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type ServiceAccount struct {
	// Email: Email address of the service account.
	Email string `json:"email,omitempty"`

	// Scopes: The list of scopes to be made available for this service
	// account.
	Scopes []string `json:"scopes,omitempty"`
}

type Snapshot struct {
	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// DiskSizeGb: Size of the persistent disk snapshot, specified in GB
	// (output only).
	DiskSizeGb int64 `json:"diskSizeGb,omitempty,string"`

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

	// SourceDisk: The source disk used to create this snapshot. Once the
	// source disk has been deleted from the system, this field will be
	// cleared, and will not be set even if a disk with the same name has
	// been re-created (output only).
	SourceDisk string `json:"sourceDisk,omitempty"`

	// SourceDiskId: The 'id' value of the disk used to create this
	// snapshot. This value may be used to determine whether the snapshot
	// was taken from the current or a previous instance of a given disk
	// name.
	SourceDiskId string `json:"sourceDiskId,omitempty"`

	// Status: The status of the persistent disk snapshot (output only).
	Status string `json:"status,omitempty"`
}

type SnapshotList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The persistent snapshot resources.
	Items []*Snapshot `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type Tags struct {
	// Fingerprint: Fingerprint of this resource. A hash of the tags stored
	// in this object. This field is used optimistic locking. An up-to-date
	// tags fingerprint must be provided in order to modify tags.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Items: An array of tags. Each tag must be 1-63 characters long, and
	// comply with RFC1035.
	Items []string `json:"items,omitempty"`
}

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
	// In case
	// where 'failoverRatio' and 'backupPool' are not set, or all the VMs in
	// the backup pool are unhealthy, the traffic will be directed back to
	// the primary pool in the "force" mode, where traffic will be spread to
	// the healthy VMs with the best effort, or to all VMs when no VM is
	// healthy.
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
	// If set, 'backupPool' must also be set. They
	// together define the fallback behavior of the primary target pool: if
	// the ratio of the healthy VMs in the primary pool is at or below this
	// number, traffic arriving at the load-balanced IP will be directed to
	// the backup pool.
	//
	// In case where 'failoverRatio' is not set or all the
	// VMs in the backup pool are unhealthy, the traffic will be directed
	// back to the primary pool in the "force" mode, where traffic will be
	// spread to the healthy VMs with the best effort, or to all VMs when no
	// VM is healthy.
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
	SessionAffinity string `json:"sessionAffinity,omitempty"`
}

type TargetPoolAggregatedList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A map of scoped target pool lists.
	Items *TargetPoolAggregatedListItems `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type TargetPoolAggregatedListItems struct {
}

type TargetPoolInstanceHealth struct {
	HealthStatus []*HealthStatus `json:"healthStatus,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`
}

type TargetPoolList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The TargetPool resources.
	Items []*TargetPool `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type TargetPoolsScopedList struct {
	// TargetPools: List of target pools contained in this scope.
	TargetPools []*TargetPool `json:"targetPools,omitempty"`

	// Warning: Informational warning which replaces the list of addresses
	// when the list is empty.
	Warning *TargetPoolsScopedListWarning `json:"warning,omitempty"`
}

type TargetPoolsScopedListWarning struct {
	// Code: The warning type identifier for this warning.
	Code string `json:"code,omitempty"`

	// Data: Metadata for this warning in 'key: value' format.
	Data []*TargetPoolsScopedListWarningData `json:"data,omitempty"`

	// Message: Optional human-readable details for this warning.
	Message string `json:"message,omitempty"`
}

type TargetPoolsScopedListWarningData struct {
	// Key: A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`
}

type TargetReference struct {
	Target string `json:"target,omitempty"`
}

type Zone struct {
	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Deprecated: The deprecation status associated with this zone.
	Deprecated *DeprecationStatus `json:"deprecated,omitempty"`

	// Description: Textual description of the resource.
	Description string `json:"description,omitempty"`

	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id uint64 `json:"id,omitempty,string"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// MaintenanceWindows: Scheduled maintenance windows for the zone. When
	// the zone is in a maintenance window, all resources which reside in
	// the zone will be unavailable.
	MaintenanceWindows []*ZoneMaintenanceWindows `json:"maintenanceWindows,omitempty"`

	// Name: Name of the resource.
	Name string `json:"name,omitempty"`

	// Quotas: Quotas assigned to this zone.
	Quotas []*Quota `json:"quotas,omitempty"`

	// Region: Full URL reference to the region which hosts the zone (output
	// only).
	Region string `json:"region,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`

	// Status: Status of the zone. "UP" or "DOWN".
	Status string `json:"status,omitempty"`
}

type ZoneMaintenanceWindows struct {
	// BeginTime: Begin time of the maintenance window, in RFC 3339 format.
	BeginTime string `json:"beginTime,omitempty"`

	// Description: Textual description of the maintenance window.
	Description string `json:"description,omitempty"`

	// EndTime: End time of the maintenance window, in RFC 3339 format.
	EndTime string `json:"endTime,omitempty"`

	// Name: Name of the maintenance window.
	Name string `json:"name,omitempty"`
}

type ZoneList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The zone resources.
	Items []*Zone `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
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
func (r *AddressesService) AggregatedList(project string) *AddressesAggregatedListCall {
	c := &AddressesAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *AddressesAggregatedListCall) Filter(filter string) *AddressesAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *AddressesAggregatedListCall) MaxResults(maxResults int64) *AddressesAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *AddressesAggregatedListCall) PageToken(pageToken string) *AddressesAggregatedListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/aggregated/addresses")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AddressAggregatedList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/aggregated/addresses",
	//   "response": {
	//     "$ref": "AddressAggregatedList"
	//   },
	//   "scopes": [
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
func (r *AddressesService) Delete(project string, region string, address string) *AddressesDeleteCall {
	c := &AddressesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.address = address
	return c
}

func (c *AddressesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/addresses/{address}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{address}", url.QueryEscape(c.address), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//   "path": "{project}/regions/{region}/addresses/{address}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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
func (r *AddressesService) Get(project string, region string, address string) *AddressesGetCall {
	c := &AddressesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.address = address
	return c
}

func (c *AddressesGetCall) Do() (*Address, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/addresses/{address}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{address}", url.QueryEscape(c.address), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Address)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//   "path": "{project}/regions/{region}/addresses/{address}",
	//   "response": {
	//     "$ref": "Address"
	//   },
	//   "scopes": [
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
func (r *AddressesService) Insert(project string, region string, address *Address) *AddressesInsertCall {
	c := &AddressesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.address = address
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/addresses")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//   "path": "{project}/regions/{region}/addresses",
	//   "request": {
	//     "$ref": "Address"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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
func (r *AddressesService) List(project string, region string) *AddressesListCall {
	c := &AddressesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *AddressesListCall) Filter(filter string) *AddressesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *AddressesListCall) MaxResults(maxResults int64) *AddressesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *AddressesListCall) PageToken(pageToken string) *AddressesListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/addresses")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AddressList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/regions/{region}/addresses",
	//   "response": {
	//     "$ref": "AddressList"
	//   },
	//   "scopes": [
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
func (r *DisksService) AggregatedList(project string) *DisksAggregatedListCall {
	c := &DisksAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *DisksAggregatedListCall) Filter(filter string) *DisksAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *DisksAggregatedListCall) MaxResults(maxResults int64) *DisksAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *DisksAggregatedListCall) PageToken(pageToken string) *DisksAggregatedListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/aggregated/disks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(DiskAggregatedList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/aggregated/disks",
	//   "response": {
	//     "$ref": "DiskAggregatedList"
	//   },
	//   "scopes": [
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

// CreateSnapshot:
func (r *DisksService) CreateSnapshot(project string, zone string, disk string, snapshot *Snapshot) *DisksCreateSnapshotCall {
	c := &DisksCreateSnapshotCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.disk = disk
	c.snapshot = snapshot
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/disks/{disk}/createSnapshot")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{disk}", url.QueryEscape(c.disk), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "POST",
	//   "id": "compute.disks.createSnapshot",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "disk"
	//   ],
	//   "parameters": {
	//     "disk": {
	//       "description": "Name of the persistent disk resource to delete.",
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
	//   "path": "{project}/zones/{zone}/disks/{disk}/createSnapshot",
	//   "request": {
	//     "$ref": "Snapshot"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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

// Delete: Deletes the specified persistent disk resource.
func (r *DisksService) Delete(project string, zone string, disk string) *DisksDeleteCall {
	c := &DisksDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.disk = disk
	return c
}

func (c *DisksDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/disks/{disk}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{disk}", url.QueryEscape(c.disk), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified persistent disk resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.disks.delete",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "disk"
	//   ],
	//   "parameters": {
	//     "disk": {
	//       "description": "Name of the persistent disk resource to delete.",
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
	//   "path": "{project}/zones/{zone}/disks/{disk}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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

// Get: Returns the specified persistent disk resource.
func (r *DisksService) Get(project string, zone string, disk string) *DisksGetCall {
	c := &DisksGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.disk = disk
	return c
}

func (c *DisksGetCall) Do() (*Disk, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/disks/{disk}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{disk}", url.QueryEscape(c.disk), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Disk)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified persistent disk resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.disks.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "disk"
	//   ],
	//   "parameters": {
	//     "disk": {
	//       "description": "Name of the persistent disk resource to return.",
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
	//   "path": "{project}/zones/{zone}/disks/{disk}",
	//   "response": {
	//     "$ref": "Disk"
	//   },
	//   "scopes": [
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

// Insert: Creates a persistent disk resource in the specified project
// using the data included in the request.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/disks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a persistent disk resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.disks.insert",
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
	//     "sourceImage": {
	//       "description": "Optional. Source image to restore onto a disk.",
	//       "location": "query",
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
	//   "path": "{project}/zones/{zone}/disks",
	//   "request": {
	//     "$ref": "Disk"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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

// List: Retrieves the list of persistent disk resources contained
// within the specified zone.
func (r *DisksService) List(project string, zone string) *DisksListCall {
	c := &DisksListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *DisksListCall) Filter(filter string) *DisksListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *DisksListCall) MaxResults(maxResults int64) *DisksListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *DisksListCall) PageToken(pageToken string) *DisksListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/disks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(DiskList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of persistent disk resources contained within the specified zone.",
	//   "httpMethod": "GET",
	//   "id": "compute.disks.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/zones/{zone}/disks",
	//   "response": {
	//     "$ref": "DiskList"
	//   },
	//   "scopes": [
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
func (r *FirewallsService) Delete(project string, firewall string) *FirewallsDeleteCall {
	c := &FirewallsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.firewall = firewall
	return c
}

func (c *FirewallsDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/firewalls/{firewall}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{firewall}", url.QueryEscape(c.firewall), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project scoping this request.",
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
func (r *FirewallsService) Get(project string, firewall string) *FirewallsGetCall {
	c := &FirewallsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.firewall = firewall
	return c
}

func (c *FirewallsGetCall) Do() (*Firewall, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/firewalls/{firewall}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{firewall}", url.QueryEscape(c.firewall), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Firewall)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project scoping this request.",
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
func (r *FirewallsService) Insert(project string, firewall *Firewall) *FirewallsInsertCall {
	c := &FirewallsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.firewall = firewall
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/firewalls")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project scoping this request.",
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
func (r *FirewallsService) List(project string) *FirewallsListCall {
	c := &FirewallsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *FirewallsListCall) Filter(filter string) *FirewallsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *FirewallsListCall) MaxResults(maxResults int64) *FirewallsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *FirewallsListCall) PageToken(pageToken string) *FirewallsListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/firewalls")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(FirewallList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/global/firewalls",
	//   "response": {
	//     "$ref": "FirewallList"
	//   },
	//   "scopes": [
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
func (r *FirewallsService) Patch(project string, firewall string, firewall2 *Firewall) *FirewallsPatchCall {
	c := &FirewallsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.firewall = firewall
	c.firewall2 = firewall2
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/firewalls/{firewall}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{firewall}", url.QueryEscape(c.firewall), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project scoping this request.",
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
func (r *FirewallsService) Update(project string, firewall string, firewall2 *Firewall) *FirewallsUpdateCall {
	c := &FirewallsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.firewall = firewall
	c.firewall2 = firewall2
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/firewalls/{firewall}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{firewall}", url.QueryEscape(c.firewall), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project scoping this request.",
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
func (r *ForwardingRulesService) AggregatedList(project string) *ForwardingRulesAggregatedListCall {
	c := &ForwardingRulesAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *ForwardingRulesAggregatedListCall) Filter(filter string) *ForwardingRulesAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *ForwardingRulesAggregatedListCall) MaxResults(maxResults int64) *ForwardingRulesAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *ForwardingRulesAggregatedListCall) PageToken(pageToken string) *ForwardingRulesAggregatedListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/aggregated/forwardingRules")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(ForwardingRuleAggregatedList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
func (r *ForwardingRulesService) Delete(project string, region string, forwardingRule string) *ForwardingRulesDeleteCall {
	c := &ForwardingRulesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.forwardingRule = forwardingRule
	return c
}

func (c *ForwardingRulesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/forwardingRules/{forwardingRule}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{forwardingRule}", url.QueryEscape(c.forwardingRule), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *ForwardingRulesService) Get(project string, region string, forwardingRule string) *ForwardingRulesGetCall {
	c := &ForwardingRulesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.forwardingRule = forwardingRule
	return c
}

func (c *ForwardingRulesGetCall) Do() (*ForwardingRule, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/forwardingRules/{forwardingRule}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{forwardingRule}", url.QueryEscape(c.forwardingRule), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(ForwardingRule)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *ForwardingRulesService) Insert(project string, region string, forwardingrule *ForwardingRule) *ForwardingRulesInsertCall {
	c := &ForwardingRulesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.forwardingrule = forwardingrule
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/forwardingRules")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *ForwardingRulesService) List(project string, region string) *ForwardingRulesListCall {
	c := &ForwardingRulesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *ForwardingRulesListCall) Filter(filter string) *ForwardingRulesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *ForwardingRulesListCall) MaxResults(maxResults int64) *ForwardingRulesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *ForwardingRulesListCall) PageToken(pageToken string) *ForwardingRulesListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/forwardingRules")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(ForwardingRuleList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
func (r *ForwardingRulesService) SetTarget(project string, region string, forwardingRule string, targetreference *TargetReference) *ForwardingRulesSetTargetCall {
	c := &ForwardingRulesSetTargetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.forwardingRule = forwardingRule
	c.targetreference = targetreference
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/forwardingRules/{forwardingRule}/setTarget")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{forwardingRule}", url.QueryEscape(c.forwardingRule), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *GlobalOperationsService) AggregatedList(project string) *GlobalOperationsAggregatedListCall {
	c := &GlobalOperationsAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *GlobalOperationsAggregatedListCall) Filter(filter string) *GlobalOperationsAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *GlobalOperationsAggregatedListCall) MaxResults(maxResults int64) *GlobalOperationsAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *GlobalOperationsAggregatedListCall) PageToken(pageToken string) *GlobalOperationsAggregatedListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/aggregated/operations")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(OperationAggregatedList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/aggregated/operations",
	//   "response": {
	//     "$ref": "OperationAggregatedList"
	//   },
	//   "scopes": [
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

// Delete: Deletes the specified operation resource.
func (r *GlobalOperationsService) Delete(project string, operation string) *GlobalOperationsDeleteCall {
	c := &GlobalOperationsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.operation = operation
	return c
}

func (c *GlobalOperationsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/operations/{operation}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{operation}", url.QueryEscape(c.operation), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the specified operation resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.globalOperations.delete",
	//   "parameterOrder": [
	//     "project",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the operation resource to delete.",
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
	//   "path": "{project}/global/operations/{operation}",
	//   "scopes": [
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

// Get: Retrieves the specified operation resource.
func (r *GlobalOperationsService) Get(project string, operation string) *GlobalOperationsGetCall {
	c := &GlobalOperationsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.operation = operation
	return c
}

func (c *GlobalOperationsGetCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/operations/{operation}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{operation}", url.QueryEscape(c.operation), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the specified operation resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.globalOperations.get",
	//   "parameterOrder": [
	//     "project",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the operation resource to return.",
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
	//   "path": "{project}/global/operations/{operation}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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

// List: Retrieves the list of operation resources contained within the
// specified project.
func (r *GlobalOperationsService) List(project string) *GlobalOperationsListCall {
	c := &GlobalOperationsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *GlobalOperationsListCall) Filter(filter string) *GlobalOperationsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *GlobalOperationsListCall) MaxResults(maxResults int64) *GlobalOperationsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *GlobalOperationsListCall) PageToken(pageToken string) *GlobalOperationsListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/operations")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(OperationList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of operation resources contained within the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.globalOperations.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/global/operations",
	//   "response": {
	//     "$ref": "OperationList"
	//   },
	//   "scopes": [
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
func (r *HttpHealthChecksService) Delete(project string, httpHealthCheck string) *HttpHealthChecksDeleteCall {
	c := &HttpHealthChecksDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.httpHealthCheck = httpHealthCheck
	return c
}

func (c *HttpHealthChecksDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/httpHealthChecks/{httpHealthCheck}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{httpHealthCheck}", url.QueryEscape(c.httpHealthCheck), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *HttpHealthChecksService) Get(project string, httpHealthCheck string) *HttpHealthChecksGetCall {
	c := &HttpHealthChecksGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.httpHealthCheck = httpHealthCheck
	return c
}

func (c *HttpHealthChecksGetCall) Do() (*HttpHealthCheck, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/httpHealthChecks/{httpHealthCheck}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{httpHealthCheck}", url.QueryEscape(c.httpHealthCheck), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(HttpHealthCheck)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *HttpHealthChecksService) Insert(project string, httphealthcheck *HttpHealthCheck) *HttpHealthChecksInsertCall {
	c := &HttpHealthChecksInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.httphealthcheck = httphealthcheck
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/httpHealthChecks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *HttpHealthChecksService) List(project string) *HttpHealthChecksListCall {
	c := &HttpHealthChecksListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *HttpHealthChecksListCall) Filter(filter string) *HttpHealthChecksListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *HttpHealthChecksListCall) MaxResults(maxResults int64) *HttpHealthChecksListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *HttpHealthChecksListCall) PageToken(pageToken string) *HttpHealthChecksListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/httpHealthChecks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(HttpHealthCheckList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
func (r *HttpHealthChecksService) Patch(project string, httpHealthCheck string, httphealthcheck *HttpHealthCheck) *HttpHealthChecksPatchCall {
	c := &HttpHealthChecksPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.httpHealthCheck = httpHealthCheck
	c.httphealthcheck = httphealthcheck
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/httpHealthChecks/{httpHealthCheck}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{httpHealthCheck}", url.QueryEscape(c.httpHealthCheck), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *HttpHealthChecksService) Update(project string, httpHealthCheck string, httphealthcheck *HttpHealthCheck) *HttpHealthChecksUpdateCall {
	c := &HttpHealthChecksUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.httpHealthCheck = httpHealthCheck
	c.httphealthcheck = httphealthcheck
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/httpHealthChecks/{httpHealthCheck}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{httpHealthCheck}", url.QueryEscape(c.httpHealthCheck), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *ImagesService) Delete(project string, image string) *ImagesDeleteCall {
	c := &ImagesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.image = image
	return c
}

func (c *ImagesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/images/{image}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{image}", url.QueryEscape(c.image), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project scoping this request.",
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

// Deprecate: Sets the deprecation status of an image. If no message
// body is given, clears the deprecation status instead.
func (r *ImagesService) Deprecate(project string, image string, deprecationstatus *DeprecationStatus) *ImagesDeprecateCall {
	c := &ImagesDeprecateCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.image = image
	c.deprecationstatus = deprecationstatus
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/images/{image}/deprecate")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{image}", url.QueryEscape(c.image), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sets the deprecation status of an image. If no message body is given, clears the deprecation status instead.",
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
	//       "description": "Name of the project scoping this request.",
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
func (r *ImagesService) Get(project string, image string) *ImagesGetCall {
	c := &ImagesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.image = image
	return c
}

func (c *ImagesGetCall) Do() (*Image, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/images/{image}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{image}", url.QueryEscape(c.image), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Image)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project scoping this request.",
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
func (r *ImagesService) Insert(project string, image *Image) *ImagesInsertCall {
	c := &ImagesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.image = image
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/images")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project scoping this request.",
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
func (r *ImagesService) List(project string) *ImagesListCall {
	c := &ImagesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *ImagesListCall) Filter(filter string) *ImagesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *ImagesListCall) MaxResults(maxResults int64) *ImagesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *ImagesListCall) PageToken(pageToken string) *ImagesListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/images")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(ImageList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/global/images",
	//   "response": {
	//     "$ref": "ImageList"
	//   },
	//   "scopes": [
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
func (r *InstancesService) AddAccessConfig(project string, zone string, instance string, networkInterface string, accessconfig *AccessConfig) *InstancesAddAccessConfigCall {
	c := &InstancesAddAccessConfigCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.networkInterface = networkInterface
	c.accessconfig = accessconfig
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/instances/{instance}/addAccessConfig")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Instance name.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "networkInterface": {
	//       "description": "Network interface name.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project name.",
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
	//   "path": "{project}/zones/{zone}/instances/{instance}/addAccessConfig",
	//   "request": {
	//     "$ref": "AccessConfig"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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
func (r *InstancesService) AggregatedList(project string) *InstancesAggregatedListCall {
	c := &InstancesAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *InstancesAggregatedListCall) Filter(filter string) *InstancesAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *InstancesAggregatedListCall) MaxResults(maxResults int64) *InstancesAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *InstancesAggregatedListCall) PageToken(pageToken string) *InstancesAggregatedListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/aggregated/instances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(InstanceAggregatedList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/aggregated/instances",
	//   "response": {
	//     "$ref": "InstanceAggregatedList"
	//   },
	//   "scopes": [
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

// AttachDisk: Attaches a disk resource to an instance.
func (r *InstancesService) AttachDisk(project string, zone string, instance string, attacheddisk *AttachedDisk) *InstancesAttachDiskCall {
	c := &InstancesAttachDiskCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.attacheddisk = attacheddisk
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/instances/{instance}/attachDisk")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Attaches a disk resource to an instance.",
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
	//       "description": "Project name.",
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
	//   "path": "{project}/zones/{zone}/instances/{instance}/attachDisk",
	//   "request": {
	//     "$ref": "AttachedDisk"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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

// Delete: Deletes the specified instance resource.
func (r *InstancesService) Delete(project string, zone string, instance string) *InstancesDeleteCall {
	c := &InstancesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	return c
}

func (c *InstancesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/instances/{instance}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified instance resource.",
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
	//   "path": "{project}/zones/{zone}/instances/{instance}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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
func (r *InstancesService) DeleteAccessConfig(project string, zone string, instance string, accessConfig string, networkInterface string) *InstancesDeleteAccessConfigCall {
	c := &InstancesDeleteAccessConfigCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.accessConfig = accessConfig
	c.networkInterface = networkInterface
	return c
}

func (c *InstancesDeleteAccessConfigCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("accessConfig", fmt.Sprintf("%v", c.accessConfig))
	params.Set("networkInterface", fmt.Sprintf("%v", c.networkInterface))
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/instances/{instance}/deleteAccessConfig")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Access config name.",
	//       "location": "query",
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
	//     "networkInterface": {
	//       "description": "Network interface name.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project name.",
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
	//   "path": "{project}/zones/{zone}/instances/{instance}/deleteAccessConfig",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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
func (r *InstancesService) DetachDisk(project string, zone string, instance string, deviceName string) *InstancesDetachDiskCall {
	c := &InstancesDetachDiskCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.deviceName = deviceName
	return c
}

func (c *InstancesDetachDiskCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("deviceName", fmt.Sprintf("%v", c.deviceName))
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/instances/{instance}/detachDisk")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Project name.",
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
	//   "path": "{project}/zones/{zone}/instances/{instance}/detachDisk",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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
func (r *InstancesService) Get(project string, zone string, instance string) *InstancesGetCall {
	c := &InstancesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	return c
}

func (c *InstancesGetCall) Do() (*Instance, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/instances/{instance}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Instance)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//   "path": "{project}/zones/{zone}/instances/{instance}",
	//   "response": {
	//     "$ref": "Instance"
	//   },
	//   "scopes": [
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
func (r *InstancesService) GetSerialPortOutput(project string, zone string, instance string) *InstancesGetSerialPortOutputCall {
	c := &InstancesGetSerialPortOutputCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	return c
}

func (c *InstancesGetSerialPortOutputCall) Do() (*SerialPortOutput, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/instances/{instance}/serialPort")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(SerialPortOutput)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//   "path": "{project}/zones/{zone}/instances/{instance}/serialPort",
	//   "response": {
	//     "$ref": "SerialPortOutput"
	//   },
	//   "scopes": [
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
func (r *InstancesService) Insert(project string, zone string, instance *Instance) *InstancesInsertCall {
	c := &InstancesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/instances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//   "path": "{project}/zones/{zone}/instances",
	//   "request": {
	//     "$ref": "Instance"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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
func (r *InstancesService) List(project string, zone string) *InstancesListCall {
	c := &InstancesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *InstancesListCall) Filter(filter string) *InstancesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *InstancesListCall) MaxResults(maxResults int64) *InstancesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *InstancesListCall) PageToken(pageToken string) *InstancesListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/instances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(InstanceList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/zones/{zone}/instances",
	//   "response": {
	//     "$ref": "InstanceList"
	//   },
	//   "scopes": [
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
func (r *InstancesService) Reset(project string, zone string, instance string) *InstancesResetCall {
	c := &InstancesResetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	return c
}

func (c *InstancesResetCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/instances/{instance}/reset")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//   "path": "{project}/zones/{zone}/instances/{instance}/reset",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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
func (r *InstancesService) SetMetadata(project string, zone string, instance string, metadata *Metadata) *InstancesSetMetadataCall {
	c := &InstancesSetMetadataCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.metadata = metadata
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/instances/{instance}/setMetadata")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//   "path": "{project}/zones/{zone}/instances/{instance}/setMetadata",
	//   "request": {
	//     "$ref": "Metadata"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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
func (r *InstancesService) SetTags(project string, zone string, instance string, tags *Tags) *InstancesSetTagsCall {
	c := &InstancesSetTagsCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.tags = tags
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/instances/{instance}/setTags")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//   "path": "{project}/zones/{zone}/instances/{instance}/setTags",
	//   "request": {
	//     "$ref": "Tags"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.kernels.get":

type KernelsGetCall struct {
	s       *Service
	project string
	kernel  string
	opt_    map[string]interface{}
}

// Get: Returns the specified kernel resource.
func (r *KernelsService) Get(project string, kernel string) *KernelsGetCall {
	c := &KernelsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.kernel = kernel
	return c
}

func (c *KernelsGetCall) Do() (*Kernel, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/kernels/{kernel}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{kernel}", url.QueryEscape(c.kernel), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Kernel)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified kernel resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.kernels.get",
	//   "parameterOrder": [
	//     "project",
	//     "kernel"
	//   ],
	//   "parameters": {
	//     "kernel": {
	//       "description": "Name of the kernel resource to return.",
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
	//   "path": "{project}/global/kernels/{kernel}",
	//   "response": {
	//     "$ref": "Kernel"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.kernels.list":

type KernelsListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Retrieves the list of kernel resources available to the
// specified project.
func (r *KernelsService) List(project string) *KernelsListCall {
	c := &KernelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *KernelsListCall) Filter(filter string) *KernelsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *KernelsListCall) MaxResults(maxResults int64) *KernelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *KernelsListCall) PageToken(pageToken string) *KernelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *KernelsListCall) Do() (*KernelList, error) {
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/kernels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(KernelList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of kernel resources available to the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.kernels.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/global/kernels",
	//   "response": {
	//     "$ref": "KernelList"
	//   },
	//   "scopes": [
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
func (r *MachineTypesService) AggregatedList(project string) *MachineTypesAggregatedListCall {
	c := &MachineTypesAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *MachineTypesAggregatedListCall) Filter(filter string) *MachineTypesAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *MachineTypesAggregatedListCall) MaxResults(maxResults int64) *MachineTypesAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *MachineTypesAggregatedListCall) PageToken(pageToken string) *MachineTypesAggregatedListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/aggregated/machineTypes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(MachineTypeAggregatedList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/aggregated/machineTypes",
	//   "response": {
	//     "$ref": "MachineTypeAggregatedList"
	//   },
	//   "scopes": [
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
func (r *MachineTypesService) Get(project string, zone string, machineType string) *MachineTypesGetCall {
	c := &MachineTypesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.machineType = machineType
	return c
}

func (c *MachineTypesGetCall) Do() (*MachineType, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/machineTypes/{machineType}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{machineType}", url.QueryEscape(c.machineType), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(MachineType)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//   "path": "{project}/zones/{zone}/machineTypes/{machineType}",
	//   "response": {
	//     "$ref": "MachineType"
	//   },
	//   "scopes": [
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
func (r *MachineTypesService) List(project string, zone string) *MachineTypesListCall {
	c := &MachineTypesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *MachineTypesListCall) Filter(filter string) *MachineTypesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *MachineTypesListCall) MaxResults(maxResults int64) *MachineTypesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *MachineTypesListCall) PageToken(pageToken string) *MachineTypesListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/machineTypes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(MachineTypeList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/zones/{zone}/machineTypes",
	//   "response": {
	//     "$ref": "MachineTypeList"
	//   },
	//   "scopes": [
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
func (r *NetworksService) Delete(project string, network string) *NetworksDeleteCall {
	c := &NetworksDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.network = network
	return c
}

func (c *NetworksDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/networks/{network}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{network}", url.QueryEscape(c.network), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project scoping this request.",
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
func (r *NetworksService) Get(project string, network string) *NetworksGetCall {
	c := &NetworksGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.network = network
	return c
}

func (c *NetworksGetCall) Do() (*Network, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/networks/{network}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{network}", url.QueryEscape(c.network), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Network)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project scoping this request.",
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
func (r *NetworksService) Insert(project string, network *Network) *NetworksInsertCall {
	c := &NetworksInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.network = network
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/networks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project scoping this request.",
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
func (r *NetworksService) List(project string) *NetworksListCall {
	c := &NetworksListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *NetworksListCall) Filter(filter string) *NetworksListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *NetworksListCall) MaxResults(maxResults int64) *NetworksListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *NetworksListCall) PageToken(pageToken string) *NetworksListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/networks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(NetworkList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/global/networks",
	//   "response": {
	//     "$ref": "NetworkList"
	//   },
	//   "scopes": [
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
func (r *ProjectsService) Get(project string) *ProjectsGetCall {
	c := &ProjectsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

func (c *ProjectsGetCall) Do() (*Project, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Project)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project resource to retrieve.",
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
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
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
func (r *ProjectsService) SetCommonInstanceMetadata(project string, metadata *Metadata) *ProjectsSetCommonInstanceMetadataCall {
	c := &ProjectsSetCommonInstanceMetadataCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.metadata = metadata
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/setCommonInstanceMetadata")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project scoping this request.",
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
	//     "https://www.googleapis.com/auth/compute"
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

// Delete: Deletes the specified region-specific operation resource.
func (r *RegionOperationsService) Delete(project string, region string, operation string) *RegionOperationsDeleteCall {
	c := &RegionOperationsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.operation = operation
	return c
}

func (c *RegionOperationsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/operations/{operation}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{operation}", url.QueryEscape(c.operation), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the specified region-specific operation resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.regionOperations.delete",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the operation resource to delete.",
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
	//   "path": "{project}/regions/{region}/operations/{operation}",
	//   "scopes": [
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

// Get: Retrieves the specified region-specific operation resource.
func (r *RegionOperationsService) Get(project string, region string, operation string) *RegionOperationsGetCall {
	c := &RegionOperationsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.operation = operation
	return c
}

func (c *RegionOperationsGetCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/operations/{operation}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{operation}", url.QueryEscape(c.operation), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the specified region-specific operation resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.regionOperations.get",
	//   "parameterOrder": [
	//     "project",
	//     "region",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the operation resource to return.",
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

// List: Retrieves the list of operation resources contained within the
// specified region.
func (r *RegionOperationsService) List(project string, region string) *RegionOperationsListCall {
	c := &RegionOperationsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *RegionOperationsListCall) Filter(filter string) *RegionOperationsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *RegionOperationsListCall) MaxResults(maxResults int64) *RegionOperationsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *RegionOperationsListCall) PageToken(pageToken string) *RegionOperationsListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/operations")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(OperationList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of operation resources contained within the specified region.",
	//   "httpMethod": "GET",
	//   "id": "compute.regionOperations.list",
	//   "parameterOrder": [
	//     "project",
	//     "region"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/regions/{region}/operations",
	//   "response": {
	//     "$ref": "OperationList"
	//   },
	//   "scopes": [
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
func (r *RegionsService) Get(project string, region string) *RegionsGetCall {
	c := &RegionsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	return c
}

func (c *RegionsGetCall) Do() (*Region, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Region)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project scoping this request.",
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
func (r *RegionsService) List(project string) *RegionsListCall {
	c := &RegionsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *RegionsListCall) Filter(filter string) *RegionsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *RegionsListCall) MaxResults(maxResults int64) *RegionsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *RegionsListCall) PageToken(pageToken string) *RegionsListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(RegionList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/regions",
	//   "response": {
	//     "$ref": "RegionList"
	//   },
	//   "scopes": [
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
func (r *RoutesService) Delete(project string, route string) *RoutesDeleteCall {
	c := &RoutesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.route = route
	return c
}

func (c *RoutesDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/routes/{route}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{route}", url.QueryEscape(c.route), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *RoutesService) Get(project string, route string) *RoutesGetCall {
	c := &RoutesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.route = route
	return c
}

func (c *RoutesGetCall) Do() (*Route, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/routes/{route}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{route}", url.QueryEscape(c.route), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Route)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *RoutesService) Insert(project string, route *Route) *RoutesInsertCall {
	c := &RoutesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.route = route
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/routes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *RoutesService) List(project string) *RoutesListCall {
	c := &RoutesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *RoutesListCall) Filter(filter string) *RoutesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *RoutesListCall) MaxResults(maxResults int64) *RoutesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *RoutesListCall) PageToken(pageToken string) *RoutesListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/routes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(RouteList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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

// Delete: Deletes the specified persistent disk snapshot resource.
func (r *SnapshotsService) Delete(project string, snapshot string) *SnapshotsDeleteCall {
	c := &SnapshotsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.snapshot = snapshot
	return c
}

func (c *SnapshotsDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/snapshots/{snapshot}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{snapshot}", url.QueryEscape(c.snapshot), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified persistent disk snapshot resource.",
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
	//       "description": "Name of the persistent disk snapshot resource to delete.",
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

// Get: Returns the specified persistent disk snapshot resource.
func (r *SnapshotsService) Get(project string, snapshot string) *SnapshotsGetCall {
	c := &SnapshotsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.snapshot = snapshot
	return c
}

func (c *SnapshotsGetCall) Do() (*Snapshot, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/snapshots/{snapshot}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{snapshot}", url.QueryEscape(c.snapshot), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Snapshot)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified persistent disk snapshot resource.",
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
	//       "description": "Name of the persistent disk snapshot resource to return.",
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

// List: Retrieves the list of persistent disk snapshot resources
// contained within the specified project.
func (r *SnapshotsService) List(project string) *SnapshotsListCall {
	c := &SnapshotsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *SnapshotsListCall) Filter(filter string) *SnapshotsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *SnapshotsListCall) MaxResults(maxResults int64) *SnapshotsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *SnapshotsListCall) PageToken(pageToken string) *SnapshotsListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/global/snapshots")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(SnapshotList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of persistent disk snapshot resources contained within the specified project.",
	//   "httpMethod": "GET",
	//   "id": "compute.snapshots.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.targetPools.addHealthCheck":

type TargetPoolsAddHealthCheckCall struct {
	s                    *Service
	project              string
	region               string
	targetPool           string
	healthcheckreference *HealthCheckReference
	opt_                 map[string]interface{}
}

// AddHealthCheck: Adds health check URL to targetPool.
func (r *TargetPoolsService) AddHealthCheck(project string, region string, targetPool string, healthcheckreference *HealthCheckReference) *TargetPoolsAddHealthCheckCall {
	c := &TargetPoolsAddHealthCheckCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	c.healthcheckreference = healthcheckreference
	return c
}

func (c *TargetPoolsAddHealthCheckCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.healthcheckreference)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/targetPools/{targetPool}/addHealthCheck")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{targetPool}", url.QueryEscape(c.targetPool), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the TargetPool resource to which health_check_url is to be added.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetPools/{targetPool}/addHealthCheck",
	//   "request": {
	//     "$ref": "HealthCheckReference"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetPools.addInstance":

type TargetPoolsAddInstanceCall struct {
	s                 *Service
	project           string
	region            string
	targetPool        string
	instancereference *InstanceReference
	opt_              map[string]interface{}
}

// AddInstance: Adds instance url to targetPool.
func (r *TargetPoolsService) AddInstance(project string, region string, targetPool string, instancereference *InstanceReference) *TargetPoolsAddInstanceCall {
	c := &TargetPoolsAddInstanceCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	c.instancereference = instancereference
	return c
}

func (c *TargetPoolsAddInstanceCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancereference)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/targetPools/{targetPool}/addInstance")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{targetPool}", url.QueryEscape(c.targetPool), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the TargetPool resource to which instance_url is to be added.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetPools/{targetPool}/addInstance",
	//   "request": {
	//     "$ref": "InstanceReference"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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
func (r *TargetPoolsService) AggregatedList(project string) *TargetPoolsAggregatedListCall {
	c := &TargetPoolsAggregatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *TargetPoolsAggregatedListCall) Filter(filter string) *TargetPoolsAggregatedListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *TargetPoolsAggregatedListCall) MaxResults(maxResults int64) *TargetPoolsAggregatedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *TargetPoolsAggregatedListCall) PageToken(pageToken string) *TargetPoolsAggregatedListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/aggregated/targetPools")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(TargetPoolAggregatedList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
func (r *TargetPoolsService) Delete(project string, region string, targetPool string) *TargetPoolsDeleteCall {
	c := &TargetPoolsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	return c
}

func (c *TargetPoolsDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/targetPools/{targetPool}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{targetPool}", url.QueryEscape(c.targetPool), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *TargetPoolsService) Get(project string, region string, targetPool string) *TargetPoolsGetCall {
	c := &TargetPoolsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	return c
}

func (c *TargetPoolsGetCall) Do() (*TargetPool, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/targetPools/{targetPool}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{targetPool}", url.QueryEscape(c.targetPool), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(TargetPool)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *TargetPoolsService) GetHealth(project string, region string, targetPool string, instancereference *InstanceReference) *TargetPoolsGetHealthCall {
	c := &TargetPoolsGetHealthCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	c.instancereference = instancereference
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/targetPools/{targetPool}/getHealth")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{targetPool}", url.QueryEscape(c.targetPool), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(TargetPoolInstanceHealth)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *TargetPoolsService) Insert(project string, region string, targetpool *TargetPool) *TargetPoolsInsertCall {
	c := &TargetPoolsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetpool = targetpool
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/targetPools")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
func (r *TargetPoolsService) List(project string, region string) *TargetPoolsListCall {
	c := &TargetPoolsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *TargetPoolsListCall) Filter(filter string) *TargetPoolsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *TargetPoolsListCall) MaxResults(maxResults int64) *TargetPoolsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *TargetPoolsListCall) PageToken(pageToken string) *TargetPoolsListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/targetPools")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(TargetPoolList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "compute.targetPools.removeHealthCheck":

type TargetPoolsRemoveHealthCheckCall struct {
	s                    *Service
	project              string
	region               string
	targetPool           string
	healthcheckreference *HealthCheckReference
	opt_                 map[string]interface{}
}

// RemoveHealthCheck: Removes health check URL from targetPool.
func (r *TargetPoolsService) RemoveHealthCheck(project string, region string, targetPool string, healthcheckreference *HealthCheckReference) *TargetPoolsRemoveHealthCheckCall {
	c := &TargetPoolsRemoveHealthCheckCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	c.healthcheckreference = healthcheckreference
	return c
}

func (c *TargetPoolsRemoveHealthCheckCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.healthcheckreference)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/targetPools/{targetPool}/removeHealthCheck")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{targetPool}", url.QueryEscape(c.targetPool), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the TargetPool resource to which health_check_url is to be removed.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetPools/{targetPool}/removeHealthCheck",
	//   "request": {
	//     "$ref": "HealthCheckReference"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "compute.targetPools.removeInstance":

type TargetPoolsRemoveInstanceCall struct {
	s                 *Service
	project           string
	region            string
	targetPool        string
	instancereference *InstanceReference
	opt_              map[string]interface{}
}

// RemoveInstance: Removes instance URL from targetPool.
func (r *TargetPoolsService) RemoveInstance(project string, region string, targetPool string, instancereference *InstanceReference) *TargetPoolsRemoveInstanceCall {
	c := &TargetPoolsRemoveInstanceCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.region = region
	c.targetPool = targetPool
	c.instancereference = instancereference
	return c
}

func (c *TargetPoolsRemoveInstanceCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancereference)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/targetPools/{targetPool}/removeInstance")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{targetPool}", url.QueryEscape(c.targetPool), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the TargetPool resource to which instance_url is to be removed.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/regions/{region}/targetPools/{targetPool}/removeInstance",
	//   "request": {
	//     "$ref": "InstanceReference"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/regions/{region}/targetPools/{targetPool}/setBackup")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{region}", url.QueryEscape(c.region), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{targetPool}", url.QueryEscape(c.targetPool), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//     "https://www.googleapis.com/auth/compute"
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

// Delete: Deletes the specified zone-specific operation resource.
func (r *ZoneOperationsService) Delete(project string, zone string, operation string) *ZoneOperationsDeleteCall {
	c := &ZoneOperationsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.operation = operation
	return c
}

func (c *ZoneOperationsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/operations/{operation}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{operation}", url.QueryEscape(c.operation), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the specified zone-specific operation resource.",
	//   "httpMethod": "DELETE",
	//   "id": "compute.zoneOperations.delete",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the operation resource to delete.",
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
	//   "path": "{project}/zones/{zone}/operations/{operation}",
	//   "scopes": [
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

// Get: Retrieves the specified zone-specific operation resource.
func (r *ZoneOperationsService) Get(project string, zone string, operation string) *ZoneOperationsGetCall {
	c := &ZoneOperationsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.operation = operation
	return c
}

func (c *ZoneOperationsGetCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/operations/{operation}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{operation}", url.QueryEscape(c.operation), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Operation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the specified zone-specific operation resource.",
	//   "httpMethod": "GET",
	//   "id": "compute.zoneOperations.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the operation resource to return.",
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
	//   "path": "{project}/zones/{zone}/operations/{operation}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
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

// List: Retrieves the list of operation resources contained within the
// specified zone.
func (r *ZoneOperationsService) List(project string, zone string) *ZoneOperationsListCall {
	c := &ZoneOperationsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *ZoneOperationsListCall) Filter(filter string) *ZoneOperationsListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *ZoneOperationsListCall) MaxResults(maxResults int64) *ZoneOperationsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *ZoneOperationsListCall) PageToken(pageToken string) *ZoneOperationsListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}/operations")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(OperationList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of operation resources contained within the specified zone.",
	//   "httpMethod": "GET",
	//   "id": "compute.zoneOperations.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/zones/{zone}/operations",
	//   "response": {
	//     "$ref": "OperationList"
	//   },
	//   "scopes": [
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
func (r *ZonesService) Get(project string, zone string) *ZonesGetCall {
	c := &ZonesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	return c
}

func (c *ZonesGetCall) Do() (*Zone, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones/{zone}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{zone}", url.QueryEscape(c.zone), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Zone)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Name of the project scoping this request.",
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
func (r *ZonesService) List(project string) *ZonesListCall {
	c := &ZonesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *ZonesListCall) Filter(filter string) *ZonesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 100.
func (c *ZonesListCall) MaxResults(maxResults int64) *ZonesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *ZonesListCall) PageToken(pageToken string) *ZonesListCall {
	c.opt_["pageToken"] = pageToken
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta15/projects/", "{project}/zones")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(ZoneList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
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
	//   "path": "{project}/zones",
	//   "response": {
	//     "$ref": "ZoneList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}
