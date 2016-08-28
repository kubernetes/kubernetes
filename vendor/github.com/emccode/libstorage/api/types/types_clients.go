package types

import (
	"io"
	"strings"
)

// ClientType is a client's type.
type ClientType int

const (
	// UnknownClientType is an unknown client type.
	UnknownClientType ClientType = iota

	// IntegrationClient is the default client type -- a client that both
	// communicates with a remote libStorage endpoint as well as interacts with
	// the local host.
	IntegrationClient

	// ControllerClient is a libStorage client that has no interaction with
	// the local host, removing any need for access to libStorage executors.
	ControllerClient
)

// String returns the client type's string representation.
func (t ClientType) String() string {
	switch t {
	case IntegrationClient:
		return "integration"
	case ControllerClient:
		return "controller"
	default:
		return ""
	}
}

// ParseClientType parses a new client type.
func ParseClientType(str string) ClientType {
	str = strings.ToLower(str)
	switch str {
	case "integration":
		return IntegrationClient
	case "controller":
		return ControllerClient
	}
	return UnknownClientType
}

// Client is the libStorage client.
type Client interface {

	// API returns the underlying libStorage API client.
	API() APIClient

	// OS returns the client's OS driver instance.
	OS() OSDriver

	// Storage returns the client's storage driver instance.
	Storage() StorageDriver

	// IntegrationDriver returns the client's integration driver instance.
	Integration() IntegrationDriver

	// Executor returns the storage executor CLI.
	Executor() StorageExecutorCLI
}

// ProvidesAPIClient is any type that provides the API client.
type ProvidesAPIClient interface {

	// API provides the API client.
	API() APIClient
}

// APIClient is the libStorage API client used for communicating with a remote
// libStorage endpoint.
type APIClient interface {

	// ServerName returns the name of the server to which the client is
	// connected. This is not the same as the host name, rather it's the
	// randomly generated name the server creates for unique identification
	// when the server starts for the first time.
	ServerName() string

	// LogRequests enables or disables the logging of client HTTP requests.
	LogRequests(enabled bool)

	// LogResponses enables or disables the logging of client HTTP responses.
	LogResponses(enabled bool)

	// Root returns a list of root resources.
	Root(ctx Context) ([]string, error)

	// Instances returns a list of instances.
	Instances(ctx Context) (map[string]*Instance, error)

	// InstanceInspect inspects an instance.
	InstanceInspect(ctx Context, service string) (*Instance, error)

	// Services returns a map of the configured Services.
	Services(ctx Context) (map[string]*ServiceInfo, error)

	// ServiceInspect returns information about a service.
	ServiceInspect(ctx Context, name string) (*ServiceInfo, error)

	// Volumes returns a list of all Volumes for all Services.
	Volumes(
		ctx Context,
		attachments bool) (ServiceVolumeMap, error)

	// VolumesByService returns a list of all Volumes for a service.
	VolumesByService(
		ctx Context,
		service string,
		attachments bool) (VolumeMap, error)

	// VolumeInspect gets information about a single volume.
	VolumeInspect(
		ctx Context,
		service, volumeID string,
		attachments bool) (*Volume, error)

	// VolumeCreate creates a single volume.
	VolumeCreate(
		ctx Context,
		service string,
		request *VolumeCreateRequest) (*Volume, error)

	// VolumeCreateFromSnapshot creates a single volume from a snapshot.
	VolumeCreateFromSnapshot(
		ctx Context,
		service, snapshotID string,
		request *VolumeCreateRequest) (*Volume, error)

	// VolumeCopy copies a single volume.
	VolumeCopy(
		ctx Context,
		service, volumeID string,
		request *VolumeCopyRequest) (*Volume, error)

	// VolumeRemove removes a single volume.
	VolumeRemove(
		ctx Context,
		service, volumeID string) error

	// VolumeAttach attaches a single volume.
	VolumeAttach(
		ctx Context,
		service string,
		volumeID string,
		request *VolumeAttachRequest) (*Volume, string, error)

	// VolumeDetach attaches a single volume.
	VolumeDetach(
		ctx Context,
		service string,
		volumeID string,
		request *VolumeDetachRequest) (*Volume, error)

	// VolumeDetachAll attaches all volumes from all
	VolumeDetachAll(
		ctx Context,
		request *VolumeDetachRequest) (ServiceVolumeMap, error)

	// VolumeDetachAllForService detaches all volumes from a service.
	VolumeDetachAllForService(
		ctx Context,
		service string,
		request *VolumeDetachRequest) (VolumeMap, error)

	// VolumeSnapshot creates a single snapshot.
	VolumeSnapshot(
		ctx Context,
		service string,
		volumeID string,
		request *VolumeSnapshotRequest) (*Snapshot, error)

	// Snapshots returns a list of all Snapshots for all
	Snapshots(ctx Context) (ServiceSnapshotMap, error)

	// SnapshotsByService returns a list of all Snapshots for a single service.
	SnapshotsByService(
		ctx Context, service string) (SnapshotMap, error)

	// SnapshotInspect gets information about a single snapshot.
	SnapshotInspect(
		ctx Context,
		service, snapshotID string) (*Snapshot, error)

	// SnapshotRemove removes a single snapshot.
	SnapshotRemove(
		ctx Context,
		service, snapshotID string) error

	// SnapshotCopy copies a snapshot to a new snapshot.
	SnapshotCopy(
		ctx Context,
		service, snapshotID string,
		request *SnapshotCopyRequest) (*Snapshot, error)

	// Executors returns information about the executors.
	Executors(
		ctx Context) (map[string]*ExecutorInfo, error)

	// ExecutorHead returns information about an executor.
	ExecutorHead(
		ctx Context,
		name string) (*ExecutorInfo, error)

	// ExecutorGet downloads an executor.
	ExecutorGet(
		ctx Context, name string) (io.ReadCloser, error)
}
