package hcsshim

import (
	"fmt"
	"os"
	"time"

	"github.com/Microsoft/hcsshim/internal/hcs"
	"github.com/Microsoft/hcsshim/internal/mergemaps"
	"github.com/Microsoft/hcsshim/internal/schema1"
)

// ContainerProperties holds the properties for a container and the processes running in that container
type ContainerProperties = schema1.ContainerProperties

// MemoryStats holds the memory statistics for a container
type MemoryStats = schema1.MemoryStats

// ProcessorStats holds the processor statistics for a container
type ProcessorStats = schema1.ProcessorStats

// StorageStats holds the storage statistics for a container
type StorageStats = schema1.StorageStats

// NetworkStats holds the network statistics for a container
type NetworkStats = schema1.NetworkStats

// Statistics is the structure returned by a statistics call on a container
type Statistics = schema1.Statistics

// ProcessList is the structure of an item returned by a ProcessList call on a container
type ProcessListItem = schema1.ProcessListItem

// MappedVirtualDiskController is the structure of an item returned by a MappedVirtualDiskList call on a container
type MappedVirtualDiskController = schema1.MappedVirtualDiskController

// Type of Request Support in ModifySystem
type RequestType = schema1.RequestType

// Type of Resource Support in ModifySystem
type ResourceType = schema1.ResourceType

// RequestType const
const (
	Add     = schema1.Add
	Remove  = schema1.Remove
	Network = schema1.Network
)

// ResourceModificationRequestResponse is the structure used to send request to the container to modify the system
// Supported resource types are Network and Request Types are Add/Remove
type ResourceModificationRequestResponse = schema1.ResourceModificationRequestResponse

type container struct {
	system *hcs.System
}

// createComputeSystemAdditionalJSON is read from the environment at initialisation
// time. It allows an environment variable to define additional JSON which
// is merged in the CreateComputeSystem call to HCS.
var createContainerAdditionalJSON []byte

func init() {
	createContainerAdditionalJSON = ([]byte)(os.Getenv("HCSSHIM_CREATECONTAINER_ADDITIONALJSON"))
}

// CreateContainer creates a new container with the given configuration but does not start it.
func CreateContainer(id string, c *ContainerConfig) (Container, error) {
	fullConfig, err := mergemaps.MergeJSON(c, createContainerAdditionalJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to merge additional JSON '%s': %s", createContainerAdditionalJSON, err)
	}

	system, err := hcs.CreateComputeSystem(id, fullConfig)
	if err != nil {
		return nil, err
	}
	return &container{system}, err
}

// OpenContainer opens an existing container by ID.
func OpenContainer(id string) (Container, error) {
	system, err := hcs.OpenComputeSystem(id)
	if err != nil {
		return nil, err
	}
	return &container{system}, err
}

// GetContainers gets a list of the containers on the system that match the query
func GetContainers(q ComputeSystemQuery) ([]ContainerProperties, error) {
	return hcs.GetComputeSystems(q)
}

// Start synchronously starts the container.
func (container *container) Start() error {
	return convertSystemError(container.system.Start(), container)
}

// Shutdown requests a container shutdown, but it may not actually be shutdown until Wait() succeeds.
func (container *container) Shutdown() error {
	return convertSystemError(container.system.Shutdown(), container)
}

// Terminate requests a container terminate, but it may not actually be terminated until Wait() succeeds.
func (container *container) Terminate() error {
	return convertSystemError(container.system.Terminate(), container)
}

// Waits synchronously waits for the container to shutdown or terminate.
func (container *container) Wait() error {
	return convertSystemError(container.system.Wait(), container)
}

// WaitTimeout synchronously waits for the container to terminate or the duration to elapse. It
// returns false if timeout occurs.
func (container *container) WaitTimeout(t time.Duration) error {
	return convertSystemError(container.system.WaitTimeout(t), container)
}

// Pause pauses the execution of a container.
func (container *container) Pause() error {
	return convertSystemError(container.system.Pause(), container)
}

// Resume resumes the execution of a container.
func (container *container) Resume() error {
	return convertSystemError(container.system.Resume(), container)
}

// HasPendingUpdates returns true if the container has updates pending to install
func (container *container) HasPendingUpdates() (bool, error) {
	return false, nil
}

// Statistics returns statistics for the container. This is a legacy v1 call
func (container *container) Statistics() (Statistics, error) {
	properties, err := container.system.Properties(schema1.PropertyTypeStatistics)
	if err != nil {
		return Statistics{}, convertSystemError(err, container)
	}

	return properties.Statistics, nil
}

// ProcessList returns an array of ProcessListItems for the container. This is a legacy v1 call
func (container *container) ProcessList() ([]ProcessListItem, error) {
	properties, err := container.system.Properties(schema1.PropertyTypeProcessList)
	if err != nil {
		return nil, convertSystemError(err, container)
	}

	return properties.ProcessList, nil
}

// This is a legacy v1 call
func (container *container) MappedVirtualDisks() (map[int]MappedVirtualDiskController, error) {
	properties, err := container.system.Properties(schema1.PropertyTypeMappedVirtualDisk)
	if err != nil {
		return nil, convertSystemError(err, container)
	}

	return properties.MappedVirtualDiskControllers, nil
}

// CreateProcess launches a new process within the container.
func (container *container) CreateProcess(c *ProcessConfig) (Process, error) {
	p, err := container.system.CreateProcess(c)
	if err != nil {
		return nil, convertSystemError(err, container)
	}
	return &process{p}, nil
}

// OpenProcess gets an interface to an existing process within the container.
func (container *container) OpenProcess(pid int) (Process, error) {
	p, err := container.system.OpenProcess(pid)
	if err != nil {
		return nil, convertSystemError(err, container)
	}
	return &process{p}, nil
}

// Close cleans up any state associated with the container but does not terminate or wait for it.
func (container *container) Close() error {
	return convertSystemError(container.system.Close(), container)
}

// Modify the System
func (container *container) Modify(config *ResourceModificationRequestResponse) error {
	return convertSystemError(container.system.Modify(config), container)
}
