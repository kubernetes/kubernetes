// Package compute provides access to the Compute Engine API.
//
// See https://developers.google.com/compute/docs/reference/v1beta14
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/compute/v1beta14"
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

const apiId = "compute:v1beta14"
const apiName = "compute"
const apiVersion = "v1beta14"
const basePath = "https://www.googleapis.com/compute/v1beta14/projects/"

// OAuth2 scopes used by this API.
const (
	// View and manage your Google Compute Engine resources
	ComputeScope = "https://www.googleapis.com/auth/compute"

	// View your Google Compute Engine resources
	ComputeReadonlyScope = "https://www.googleapis.com/auth/compute.readonly"

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
	s.Disks = NewDisksService(s)
	s.Firewalls = NewFirewallsService(s)
	s.GlobalOperations = NewGlobalOperationsService(s)
	s.Images = NewImagesService(s)
	s.Instances = NewInstancesService(s)
	s.Kernels = NewKernelsService(s)
	s.MachineTypes = NewMachineTypesService(s)
	s.Networks = NewNetworksService(s)
	s.Projects = NewProjectsService(s)
	s.Routes = NewRoutesService(s)
	s.Snapshots = NewSnapshotsService(s)
	s.ZoneOperations = NewZoneOperationsService(s)
	s.Zones = NewZonesService(s)
	return s, nil
}

type Service struct {
	client *http.Client

	Disks *DisksService

	Firewalls *FirewallsService

	GlobalOperations *GlobalOperationsService

	Images *ImagesService

	Instances *InstancesService

	Kernels *KernelsService

	MachineTypes *MachineTypesService

	Networks *NetworksService

	Projects *ProjectsService

	Routes *RoutesService

	Snapshots *SnapshotsService

	ZoneOperations *ZoneOperationsService

	Zones *ZonesService
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

func NewGlobalOperationsService(s *Service) *GlobalOperationsService {
	rs := &GlobalOperationsService{s: s}
	return rs
}

type GlobalOperationsService struct {
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
	// appropriate value.
	Index int64 `json:"index,omitempty"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Mode: The mode in which to attach this disk, either "READ_WRITE" or
	// "READ_ONLY".
	Mode string `json:"mode,omitempty"`

	// Source: Persistent disk only; the URL of the persistent disk
	// resource.
	Source string `json:"source,omitempty"`

	// Type: Type of the disk, either "EPHEMERAL" or "PERSISTENT". Note that
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

type Image struct {
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
	// AvailableZone: The zones that this machine type can run in.
	AvailableZone []string `json:"availableZone,omitempty"`

	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Deprecated: The deprecation status associated with this machine type.
	Deprecated *DeprecationStatus `json:"deprecated,omitempty"`

	// Description: An optional textual description of the resource.
	Description string `json:"description,omitempty"`

	// EphemeralDisks: List of extended ephemeral disks assigned to the
	// instance.
	EphemeralDisks []*MachineTypeEphemeralDisks `json:"ephemeralDisks,omitempty"`

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

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
}

type MachineTypeEphemeralDisks struct {
	// DiskGb: Size of the ephemeral disk, defined in GB.
	DiskGb int64 `json:"diskGb,omitempty"`
}

type MachineTypeList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The machine type resources.
	Items []*MachineType `json:"items,omitempty"`

	// Kind: Type of the resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for the resource (output only).
	SelfLink string `json:"selfLink,omitempty"`
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

	// NetworkIP: An optional IPV4 internal network address to assign to the
	// instance for this network interface. If not specified, one will be
	// assigned from the available range.
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

type Project struct {
	// CommonInstanceMetadata: Metadata key/value pairs available to all
	// instances contained in this project.
	CommonInstanceMetadata *Metadata `json:"commonInstanceMetadata,omitempty"`

	// CreationTimestamp: Creation timestamp in RFC3339 text format (output
	// only).
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource.
	Description string `json:"description,omitempty"`

	// ExternalIpAddresses: Internet available IP addresses available for
	// use in this project.
	ExternalIpAddresses []string `json:"externalIpAddresses,omitempty"`

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
	// been re-created.
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

type Zone struct {
	// AvailableMachineType: The machine types that can be used in this zone
	// (output only).
	AvailableMachineType []string `json:"availableMachineType,omitempty"`

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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/disks/{disk}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/disks/{disk}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/disks")
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
// results to be returned. Maximum and default value is 100.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/disks")
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
	//       "description": "Optional. Maximum count of results to be returned. Maximum and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/firewalls/{firewall}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/firewalls/{firewall}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/firewalls")
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
// results to be returned. Maximum and default value is 100.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/firewalls")
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
	//       "description": "Optional. Maximum count of results to be returned. Maximum and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/firewalls/{firewall}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/firewalls/{firewall}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/operations/{operation}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/operations/{operation}")
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
// results to be returned. Maximum and default value is 100.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/operations")
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
	//       "description": "Optional. Maximum count of results to be returned. Maximum and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/images/{image}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/images/{image}/deprecate")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/images/{image}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/images")
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
// results to be returned. Maximum and default value is 100.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/images")
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
	//       "description": "Optional. Maximum count of results to be returned. Maximum and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
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
	s                 *Service
	project           string
	zone              string
	instance          string
	network_interface string
	accessconfig      *AccessConfig
	opt_              map[string]interface{}
}

// AddAccessConfig: Adds an access config to an instance's network
// interface.
func (r *InstancesService) AddAccessConfig(project string, zone string, instance string, network_interface string, accessconfig *AccessConfig) *InstancesAddAccessConfigCall {
	c := &InstancesAddAccessConfigCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.network_interface = network_interface
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
	params.Set("network_interface", fmt.Sprintf("%v", c.network_interface))
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/instances/{instance}/addAccessConfig")
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
	//     "network_interface"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Instance name.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "network_interface": {
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/instances/{instance}/attachDisk")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/instances/{instance}")
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
	s                 *Service
	project           string
	zone              string
	instance          string
	access_config     string
	network_interface string
	opt_              map[string]interface{}
}

// DeleteAccessConfig: Deletes an access config from an instance's
// network interface.
func (r *InstancesService) DeleteAccessConfig(project string, zone string, instance string, access_config string, network_interface string) *InstancesDeleteAccessConfigCall {
	c := &InstancesDeleteAccessConfigCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.zone = zone
	c.instance = instance
	c.access_config = access_config
	c.network_interface = network_interface
	return c
}

func (c *InstancesDeleteAccessConfigCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("access_config", fmt.Sprintf("%v", c.access_config))
	params.Set("network_interface", fmt.Sprintf("%v", c.network_interface))
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/instances/{instance}/deleteAccessConfig")
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
	//     "access_config",
	//     "network_interface"
	//   ],
	//   "parameters": {
	//     "access_config": {
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
	//     "network_interface": {
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/instances/{instance}/detachDisk")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/instances/{instance}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/instances/{instance}/serialPort")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/instances")
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
// results to be returned. Maximum and default value is 100.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/instances")
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
	//       "description": "Optional. Maximum count of results to be returned. Maximum and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/instances/{instance}/reset")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/instances/{instance}/setMetadata")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/instances/{instance}/setTags")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/kernels/{kernel}")
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
// results to be returned. Maximum and default value is 100.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/kernels")
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
	//       "description": "Optional. Maximum count of results to be returned. Maximum and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
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

// method id "compute.machineTypes.get":

type MachineTypesGetCall struct {
	s           *Service
	project     string
	machineType string
	opt_        map[string]interface{}
}

// Get: Returns the specified machine type resource.
func (r *MachineTypesService) Get(project string, machineType string) *MachineTypesGetCall {
	c := &MachineTypesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.machineType = machineType
	return c
}

func (c *MachineTypesGetCall) Do() (*MachineType, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/machineTypes/{machineType}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
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
	//     }
	//   },
	//   "path": "{project}/global/machineTypes/{machineType}",
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
	opt_    map[string]interface{}
}

// List: Retrieves the list of machine type resources available to the
// specified project.
func (r *MachineTypesService) List(project string) *MachineTypesListCall {
	c := &MachineTypesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *MachineTypesListCall) Filter(filter string) *MachineTypesListCall {
	c.opt_["filter"] = filter
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum and default value is 100.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/machineTypes")
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
	//       "description": "Optional. Maximum count of results to be returned. Maximum and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
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
	//   "path": "{project}/global/machineTypes",
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/networks/{network}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/networks/{network}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/networks")
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
// results to be returned. Maximum and default value is 100.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/networks")
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
	//       "description": "Optional. Maximum count of results to be returned. Maximum and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/setCommonInstanceMetadata")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/routes/{route}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/routes/{route}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/routes")
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
// results to be returned. Maximum and default value is 100.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/routes")
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
	//       "description": "Optional. Maximum count of results to be returned. Maximum and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/snapshots/{snapshot}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/snapshots/{snapshot}")
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

// method id "compute.snapshots.insert":

type SnapshotsInsertCall struct {
	s        *Service
	project  string
	snapshot *Snapshot
	opt_     map[string]interface{}
}

// Insert: Creates a persistent disk snapshot resource in the specified
// project using the data included in the request.
func (r *SnapshotsService) Insert(project string, snapshot *Snapshot) *SnapshotsInsertCall {
	c := &SnapshotsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.snapshot = snapshot
	return c
}

func (c *SnapshotsInsertCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.snapshot)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/snapshots")
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
	//   "description": "Creates a persistent disk snapshot resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "compute.snapshots.insert",
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
	//   "path": "{project}/global/snapshots",
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
// results to be returned. Maximum and default value is 100.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/global/snapshots")
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
	//       "description": "Optional. Maximum count of results to be returned. Maximum and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/operations/{operation}")
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/operations/{operation}")
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
// results to be returned. Maximum and default value is 100.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}/operations")
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
	//       "description": "Optional. Maximum count of results to be returned. Maximum and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones/{zone}")
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
// results to be returned. Maximum and default value is 100.
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
	urls := googleapi.ResolveRelative("https://www.googleapis.com/compute/v1beta14/projects/", "{project}/zones")
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
	//       "description": "Optional. Maximum count of results to be returned. Maximum and default value is 100.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
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
