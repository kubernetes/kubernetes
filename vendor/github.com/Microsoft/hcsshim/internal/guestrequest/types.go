package guestrequest

import (
	"github.com/Microsoft/hcsshim/internal/schema2"
)

// Arguably, many of these (at least CombinedLayers) should have been generated
// by swagger.
//
// This will also change package name due to an inbound breaking change.

// This class is used by a modify request to add or remove a combined layers
// structure in the guest. For windows, the GCS applies a filter in ContainerRootPath
// using the specified layers as the parent content. Ignores property ScratchPath
// since the container path is already the scratch path. For linux, the GCS unions
// the specified layers and ScratchPath together, placing the resulting union
// filesystem at ContainerRootPath.
type CombinedLayers struct {
	ContainerRootPath string            `json:"ContainerRootPath,omitempty"`
	Layers            []hcsschema.Layer `json:"Layers,omitempty"`
	ScratchPath       string            `json:"ScratchPath,omitempty"`
}

// Defines the schema for hosted settings passed to GCS and/or OpenGCS

// SCSI. Scratch space for remote file-system commands, or R/W layer for containers
type LCOWMappedVirtualDisk struct {
	MountPath  string `json:"MountPath,omitempty"` // /tmp/scratch for an LCOW utility VM being used as a service VM
	Lun        uint8  `json:"Lun,omitempty"`
	Controller uint8  `json:"Controller,omitempty"`
	ReadOnly   bool   `json:"ReadOnly,omitempty"`
}

type WCOWMappedVirtualDisk struct {
	ContainerPath string `json:"ContainerPath,omitempty"`
	Lun           int32  `json:"Lun,omitempty"`
}

type LCOWMappedDirectory struct {
	MountPath string `json:"MountPath,omitempty"`
	Port      int32  `json:"Port,omitempty"`
	ShareName string `json:"ShareName,omitempty"` // If empty not using ANames (not currently supported)
	ReadOnly  bool   `json:"ReadOnly,omitempty"`
}

// Read-only layers over VPMem
type LCOWMappedVPMemDevice struct {
	DeviceNumber uint32 `json:"DeviceNumber,omitempty"`
	MountPath    string `json:"MountPath,omitempty"` // /tmp/pN
}

type LCOWNetworkAdapter struct {
	NamespaceID     string `json:",omitempty"`
	ID              string `json:",omitempty"`
	MacAddress      string `json:",omitempty"`
	IPAddress       string `json:",omitempty"`
	PrefixLength    uint8  `json:",omitempty"`
	GatewayAddress  string `json:",omitempty"`
	DNSSuffix       string `json:",omitempty"`
	DNSServerList   string `json:",omitempty"`
	EnableLowMetric bool   `json:",omitempty"`
	EncapOverhead   uint16 `json:",omitempty"`
}

type ResourceType string

const (
	// These are constants for v2 schema modify guest requests.
	ResourceTypeMappedDirectory   ResourceType = "MappedDirectory"
	ResourceTypeMappedVirtualDisk ResourceType = "MappedVirtualDisk"
	ResourceTypeNetwork           ResourceType = "Network"
	ResourceTypeNetworkNamespace  ResourceType = "NetworkNamespace"
	ResourceTypeCombinedLayers    ResourceType = "CombinedLayers"
	ResourceTypeVPMemDevice       ResourceType = "VPMemDevice"
)

// GuestRequest is for modify commands passed to the guest.
type GuestRequest struct {
	RequestType  string       `json:"RequestType,omitempty"`
	ResourceType ResourceType `json:"ResourceType,omitempty"`
	Settings     interface{}  `json:"Settings,omitempty"`
}

type NetworkModifyRequest struct {
	AdapterId   string      `json:"AdapterId,omitempty"`
	RequestType string      `json:"RequestType,omitempty"`
	Settings    interface{} `json:"Settings,omitempty"`
}

type RS4NetworkModifyRequest struct {
	AdapterInstanceId string      `json:"AdapterInstanceId,omitempty"`
	RequestType       string      `json:"RequestType,omitempty"`
	Settings          interface{} `json:"Settings,omitempty"`
}

// SignalProcessOptions is the options passed to either WCOW or LCOW
// to signal a given process.
type SignalProcessOptions struct {
	Signal int `json:,omitempty`
}
