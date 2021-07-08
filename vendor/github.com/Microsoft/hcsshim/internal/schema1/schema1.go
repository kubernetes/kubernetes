package schema1

import (
	"encoding/json"
	"time"

	"github.com/Microsoft/go-winio/pkg/guid"
	hcsschema "github.com/Microsoft/hcsshim/internal/schema2"
)

// ProcessConfig is used as both the input of Container.CreateProcess
// and to convert the parameters to JSON for passing onto the HCS
type ProcessConfig struct {
	ApplicationName   string            `json:",omitempty"`
	CommandLine       string            `json:",omitempty"`
	CommandArgs       []string          `json:",omitempty"` // Used by Linux Containers on Windows
	User              string            `json:",omitempty"`
	WorkingDirectory  string            `json:",omitempty"`
	Environment       map[string]string `json:",omitempty"`
	EmulateConsole    bool              `json:",omitempty"`
	CreateStdInPipe   bool              `json:",omitempty"`
	CreateStdOutPipe  bool              `json:",omitempty"`
	CreateStdErrPipe  bool              `json:",omitempty"`
	ConsoleSize       [2]uint           `json:",omitempty"`
	CreateInUtilityVm bool              `json:",omitempty"` // Used by Linux Containers on Windows
	OCISpecification  *json.RawMessage  `json:",omitempty"` // Used by Linux Containers on Windows
}

type Layer struct {
	ID   string
	Path string
}

type MappedDir struct {
	HostPath          string
	ContainerPath     string
	ReadOnly          bool
	BandwidthMaximum  uint64
	IOPSMaximum       uint64
	CreateInUtilityVM bool
	// LinuxMetadata - Support added in 1803/RS4+.
	LinuxMetadata bool `json:",omitempty"`
}

type MappedPipe struct {
	HostPath          string
	ContainerPipeName string
}

type HvRuntime struct {
	ImagePath           string `json:",omitempty"`
	SkipTemplate        bool   `json:",omitempty"`
	LinuxInitrdFile     string `json:",omitempty"` // File under ImagePath on host containing an initrd image for starting a Linux utility VM
	LinuxKernelFile     string `json:",omitempty"` // File under ImagePath on host containing a kernel for starting a Linux utility VM
	LinuxBootParameters string `json:",omitempty"` // Additional boot parameters for starting a Linux Utility VM in initrd mode
	BootSource          string `json:",omitempty"` // "Vhd" for Linux Utility VM booting from VHD
	WritableBootSource  bool   `json:",omitempty"` // Linux Utility VM booting from VHD
}

type MappedVirtualDisk struct {
	HostPath          string `json:",omitempty"` // Path to VHD on the host
	ContainerPath     string // Platform-specific mount point path in the container
	CreateInUtilityVM bool   `json:",omitempty"`
	ReadOnly          bool   `json:",omitempty"`
	Cache             string `json:",omitempty"` // "" (Unspecified); "Disabled"; "Enabled"; "Private"; "PrivateAllowSharing"
	AttachOnly        bool   `json:",omitempty"`
}

// AssignedDevice represents a device that has been directly assigned to a container
//
// NOTE: Support added in RS5
type AssignedDevice struct {
	//  InterfaceClassGUID of the device to assign to container.
	InterfaceClassGUID string `json:"InterfaceClassGuid,omitempty"`
}

// ContainerConfig is used as both the input of CreateContainer
// and to convert the parameters to JSON for passing onto the HCS
type ContainerConfig struct {
	SystemType                  string              // HCS requires this to be hard-coded to "Container"
	Name                        string              // Name of the container. We use the docker ID.
	Owner                       string              `json:",omitempty"` // The management platform that created this container
	VolumePath                  string              `json:",omitempty"` // Windows volume path for scratch space. Used by Windows Server Containers only. Format \\?\\Volume{GUID}
	IgnoreFlushesDuringBoot     bool                `json:",omitempty"` // Optimization hint for container startup in Windows
	LayerFolderPath             string              `json:",omitempty"` // Where the layer folders are located. Used by Windows Server Containers only. Format  %root%\windowsfilter\containerID
	Layers                      []Layer             // List of storage layers. Required for Windows Server and Hyper-V Containers. Format ID=GUID;Path=%root%\windowsfilter\layerID
	Credentials                 string              `json:",omitempty"` // Credentials information
	ProcessorCount              uint32              `json:",omitempty"` // Number of processors to assign to the container.
	ProcessorWeight             uint64              `json:",omitempty"` // CPU shares (relative weight to other containers with cpu shares). Range is from 1 to 10000. A value of 0 results in default shares.
	ProcessorMaximum            int64               `json:",omitempty"` // Specifies the portion of processor cycles that this container can use as a percentage times 100. Range is from 1 to 10000. A value of 0 results in no limit.
	StorageIOPSMaximum          uint64              `json:",omitempty"` // Maximum Storage IOPS
	StorageBandwidthMaximum     uint64              `json:",omitempty"` // Maximum Storage Bandwidth in bytes per second
	StorageSandboxSize          uint64              `json:",omitempty"` // Size in bytes that the container system drive should be expanded to if smaller
	MemoryMaximumInMB           int64               `json:",omitempty"` // Maximum memory available to the container in Megabytes
	HostName                    string              `json:",omitempty"` // Hostname
	MappedDirectories           []MappedDir         `json:",omitempty"` // List of mapped directories (volumes/mounts)
	MappedPipes                 []MappedPipe        `json:",omitempty"` // List of mapped Windows named pipes
	HvPartition                 bool                // True if it a Hyper-V Container
	NetworkSharedContainerName  string              `json:",omitempty"` // Name (ID) of the container that we will share the network stack with.
	EndpointList                []string            `json:",omitempty"` // List of networking endpoints to be attached to container
	HvRuntime                   *HvRuntime          `json:",omitempty"` // Hyper-V container settings. Used by Hyper-V containers only. Format ImagePath=%root%\BaseLayerID\UtilityVM
	Servicing                   bool                `json:",omitempty"` // True if this container is for servicing
	AllowUnqualifiedDNSQuery    bool                `json:",omitempty"` // True to allow unqualified DNS name resolution
	DNSSearchList               string              `json:",omitempty"` // Comma seperated list of DNS suffixes to use for name resolution
	ContainerType               string              `json:",omitempty"` // "Linux" for Linux containers on Windows. Omitted otherwise.
	TerminateOnLastHandleClosed bool                `json:",omitempty"` // Should HCS terminate the container once all handles have been closed
	MappedVirtualDisks          []MappedVirtualDisk `json:",omitempty"` // Array of virtual disks to mount at start
	AssignedDevices             []AssignedDevice    `json:",omitempty"` // Array of devices to assign. NOTE: Support added in RS5
}

type ComputeSystemQuery struct {
	IDs    []string `json:"Ids,omitempty"`
	Types  []string `json:",omitempty"`
	Names  []string `json:",omitempty"`
	Owners []string `json:",omitempty"`
}

type PropertyType string

const (
	PropertyTypeStatistics        PropertyType = "Statistics"        // V1 and V2
	PropertyTypeProcessList       PropertyType = "ProcessList"       // V1 and V2
	PropertyTypeMappedVirtualDisk PropertyType = "MappedVirtualDisk" // Not supported in V2 schema call
	PropertyTypeGuestConnection   PropertyType = "GuestConnection"   // V1 and V2. Nil return from HCS before RS5
)

type PropertyQuery struct {
	PropertyTypes []PropertyType `json:",omitempty"`
}

// ContainerProperties holds the properties for a container and the processes running in that container
type ContainerProperties struct {
	ID                           string `json:"Id"`
	State                        string
	Name                         string
	SystemType                   string
	RuntimeOSType                string `json:"RuntimeOsType,omitempty"`
	Owner                        string
	SiloGUID                     string                              `json:"SiloGuid,omitempty"`
	RuntimeID                    guid.GUID                           `json:"RuntimeId,omitempty"`
	IsRuntimeTemplate            bool                                `json:",omitempty"`
	RuntimeImagePath             string                              `json:",omitempty"`
	Stopped                      bool                                `json:",omitempty"`
	ExitType                     string                              `json:",omitempty"`
	AreUpdatesPending            bool                                `json:",omitempty"`
	ObRoot                       string                              `json:",omitempty"`
	Statistics                   Statistics                          `json:",omitempty"`
	ProcessList                  []ProcessListItem                   `json:",omitempty"`
	MappedVirtualDiskControllers map[int]MappedVirtualDiskController `json:",omitempty"`
	GuestConnectionInfo          GuestConnectionInfo                 `json:",omitempty"`
}

// MemoryStats holds the memory statistics for a container
type MemoryStats struct {
	UsageCommitBytes            uint64 `json:"MemoryUsageCommitBytes,omitempty"`
	UsageCommitPeakBytes        uint64 `json:"MemoryUsageCommitPeakBytes,omitempty"`
	UsagePrivateWorkingSetBytes uint64 `json:"MemoryUsagePrivateWorkingSetBytes,omitempty"`
}

// ProcessorStats holds the processor statistics for a container
type ProcessorStats struct {
	TotalRuntime100ns  uint64 `json:",omitempty"`
	RuntimeUser100ns   uint64 `json:",omitempty"`
	RuntimeKernel100ns uint64 `json:",omitempty"`
}

// StorageStats holds the storage statistics for a container
type StorageStats struct {
	ReadCountNormalized  uint64 `json:",omitempty"`
	ReadSizeBytes        uint64 `json:",omitempty"`
	WriteCountNormalized uint64 `json:",omitempty"`
	WriteSizeBytes       uint64 `json:",omitempty"`
}

// NetworkStats holds the network statistics for a container
type NetworkStats struct {
	BytesReceived          uint64 `json:",omitempty"`
	BytesSent              uint64 `json:",omitempty"`
	PacketsReceived        uint64 `json:",omitempty"`
	PacketsSent            uint64 `json:",omitempty"`
	DroppedPacketsIncoming uint64 `json:",omitempty"`
	DroppedPacketsOutgoing uint64 `json:",omitempty"`
	EndpointId             string `json:",omitempty"`
	InstanceId             string `json:",omitempty"`
}

// Statistics is the structure returned by a statistics call on a container
type Statistics struct {
	Timestamp          time.Time      `json:",omitempty"`
	ContainerStartTime time.Time      `json:",omitempty"`
	Uptime100ns        uint64         `json:",omitempty"`
	Memory             MemoryStats    `json:",omitempty"`
	Processor          ProcessorStats `json:",omitempty"`
	Storage            StorageStats   `json:",omitempty"`
	Network            []NetworkStats `json:",omitempty"`
}

// ProcessList is the structure of an item returned by a ProcessList call on a container
type ProcessListItem struct {
	CreateTimestamp              time.Time `json:",omitempty"`
	ImageName                    string    `json:",omitempty"`
	KernelTime100ns              uint64    `json:",omitempty"`
	MemoryCommitBytes            uint64    `json:",omitempty"`
	MemoryWorkingSetPrivateBytes uint64    `json:",omitempty"`
	MemoryWorkingSetSharedBytes  uint64    `json:",omitempty"`
	ProcessId                    uint32    `json:",omitempty"`
	UserTime100ns                uint64    `json:",omitempty"`
}

// MappedVirtualDiskController is the structure of an item returned by a MappedVirtualDiskList call on a container
type MappedVirtualDiskController struct {
	MappedVirtualDisks map[int]MappedVirtualDisk `json:",omitempty"`
}

// GuestDefinedCapabilities is part of the GuestConnectionInfo returned by a GuestConnection call on a utility VM
type GuestDefinedCapabilities struct {
	NamespaceAddRequestSupported  bool `json:",omitempty"`
	SignalProcessSupported        bool `json:",omitempty"`
	DumpStacksSupported           bool `json:",omitempty"`
	DeleteContainerStateSupported bool `json:",omitempty"`
	UpdateContainerSupported      bool `json:",omitempty"`
}

// GuestConnectionInfo is the structure of an iterm return by a GuestConnection call on a utility VM
type GuestConnectionInfo struct {
	SupportedSchemaVersions  []hcsschema.Version      `json:",omitempty"`
	ProtocolVersion          uint32                   `json:",omitempty"`
	GuestDefinedCapabilities GuestDefinedCapabilities `json:",omitempty"`
}

// Type of Request Support in ModifySystem
type RequestType string

// Type of Resource Support in ModifySystem
type ResourceType string

// RequestType const
const (
	Add     RequestType  = "Add"
	Remove  RequestType  = "Remove"
	Network ResourceType = "Network"
)

// ResourceModificationRequestResponse is the structure used to send request to the container to modify the system
// Supported resource types are Network and Request Types are Add/Remove
type ResourceModificationRequestResponse struct {
	Resource ResourceType `json:"ResourceType"`
	Data     interface{}  `json:"Settings"`
	Request  RequestType  `json:"RequestType,omitempty"`
}
