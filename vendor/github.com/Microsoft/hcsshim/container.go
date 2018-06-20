package hcsshim

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"syscall"
	"time"

	"github.com/sirupsen/logrus"
)

var (
	defaultTimeout = time.Minute * 4
)

const (
	pendingUpdatesQuery    = `{ "PropertyTypes" : ["PendingUpdates"]}`
	statisticsQuery        = `{ "PropertyTypes" : ["Statistics"]}`
	processListQuery       = `{ "PropertyTypes" : ["ProcessList"]}`
	mappedVirtualDiskQuery = `{ "PropertyTypes" : ["MappedVirtualDisk"]}`
)

type container struct {
	handleLock     sync.RWMutex
	handle         hcsSystem
	id             string
	callbackNumber uintptr
}

// ContainerProperties holds the properties for a container and the processes running in that container
type ContainerProperties struct {
	ID                           string `json:"Id"`
	Name                         string
	SystemType                   string
	Owner                        string
	SiloGUID                     string                              `json:"SiloGuid,omitempty"`
	RuntimeID                    string                              `json:"RuntimeId,omitempty"`
	IsRuntimeTemplate            bool                                `json:",omitempty"`
	RuntimeImagePath             string                              `json:",omitempty"`
	Stopped                      bool                                `json:",omitempty"`
	ExitType                     string                              `json:",omitempty"`
	AreUpdatesPending            bool                                `json:",omitempty"`
	ObRoot                       string                              `json:",omitempty"`
	Statistics                   Statistics                          `json:",omitempty"`
	ProcessList                  []ProcessListItem                   `json:",omitempty"`
	MappedVirtualDiskControllers map[int]MappedVirtualDiskController `json:",omitempty"`
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

// createContainerAdditionalJSON is read from the environment at initialisation
// time. It allows an environment variable to define additional JSON which
// is merged in the CreateContainer call to HCS.
var createContainerAdditionalJSON string

func init() {
	createContainerAdditionalJSON = os.Getenv("HCSSHIM_CREATECONTAINER_ADDITIONALJSON")
}

// CreateContainer creates a new container with the given configuration but does not start it.
func CreateContainer(id string, c *ContainerConfig) (Container, error) {
	return createContainerWithJSON(id, c, "")
}

// CreateContainerWithJSON creates a new container with the given configuration but does not start it.
// It is identical to CreateContainer except that optional additional JSON can be merged before passing to HCS.
func CreateContainerWithJSON(id string, c *ContainerConfig, additionalJSON string) (Container, error) {
	return createContainerWithJSON(id, c, additionalJSON)
}

func createContainerWithJSON(id string, c *ContainerConfig, additionalJSON string) (Container, error) {
	operation := "CreateContainer"
	title := "HCSShim::" + operation

	container := &container{
		id: id,
	}

	configurationb, err := json.Marshal(c)
	if err != nil {
		return nil, err
	}

	configuration := string(configurationb)
	logrus.Debugf(title+" id=%s config=%s", id, configuration)

	// Merge any additional JSON. Priority is given to what is passed in explicitly,
	// falling back to what's set in the environment.
	if additionalJSON == "" && createContainerAdditionalJSON != "" {
		additionalJSON = createContainerAdditionalJSON
	}
	if additionalJSON != "" {
		configurationMap := map[string]interface{}{}
		if err := json.Unmarshal([]byte(configuration), &configurationMap); err != nil {
			return nil, fmt.Errorf("failed to unmarshal %s: %s", configuration, err)
		}

		additionalMap := map[string]interface{}{}
		if err := json.Unmarshal([]byte(additionalJSON), &additionalMap); err != nil {
			return nil, fmt.Errorf("failed to unmarshal %s: %s", additionalJSON, err)
		}

		mergedMap := mergeMaps(additionalMap, configurationMap)
		mergedJSON, err := json.Marshal(mergedMap)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal merged configuration map %+v: %s", mergedMap, err)
		}

		configuration = string(mergedJSON)
		logrus.Debugf(title+" id=%s merged config=%s", id, configuration)
	}

	var (
		resultp  *uint16
		identity syscall.Handle
	)
	createError := hcsCreateComputeSystem(id, configuration, identity, &container.handle, &resultp)

	if createError == nil || IsPending(createError) {
		if err := container.registerCallback(); err != nil {
			// Terminate the container if it still exists. We're okay to ignore a failure here.
			container.Terminate()
			return nil, makeContainerError(container, operation, "", err)
		}
	}

	err = processAsyncHcsResult(createError, resultp, container.callbackNumber, hcsNotificationSystemCreateCompleted, &defaultTimeout)
	if err != nil {
		if err == ErrTimeout {
			// Terminate the container if it still exists. We're okay to ignore a failure here.
			container.Terminate()
		}
		return nil, makeContainerError(container, operation, configuration, err)
	}

	logrus.Debugf(title+" succeeded id=%s handle=%d", id, container.handle)
	return container, nil
}

// mergeMaps recursively merges map `fromMap` into map `ToMap`. Any pre-existing values
// in ToMap are overwritten. Values in fromMap are added to ToMap.
// From http://stackoverflow.com/questions/40491438/merging-two-json-strings-in-golang
func mergeMaps(fromMap, ToMap interface{}) interface{} {
	switch fromMap := fromMap.(type) {
	case map[string]interface{}:
		ToMap, ok := ToMap.(map[string]interface{})
		if !ok {
			return fromMap
		}
		for keyToMap, valueToMap := range ToMap {
			if valueFromMap, ok := fromMap[keyToMap]; ok {
				fromMap[keyToMap] = mergeMaps(valueFromMap, valueToMap)
			} else {
				fromMap[keyToMap] = valueToMap
			}
		}
	case nil:
		// merge(nil, map[string]interface{...}) -> map[string]interface{...}
		ToMap, ok := ToMap.(map[string]interface{})
		if ok {
			return ToMap
		}
	}
	return fromMap
}

// OpenContainer opens an existing container by ID.
func OpenContainer(id string) (Container, error) {
	operation := "OpenContainer"
	title := "HCSShim::" + operation
	logrus.Debugf(title+" id=%s", id)

	container := &container{
		id: id,
	}

	var (
		handle  hcsSystem
		resultp *uint16
	)
	err := hcsOpenComputeSystem(id, &handle, &resultp)
	err = processHcsResult(err, resultp)
	if err != nil {
		return nil, makeContainerError(container, operation, "", err)
	}

	container.handle = handle

	if err := container.registerCallback(); err != nil {
		return nil, makeContainerError(container, operation, "", err)
	}

	logrus.Debugf(title+" succeeded id=%s handle=%d", id, handle)
	return container, nil
}

// GetContainers gets a list of the containers on the system that match the query
func GetContainers(q ComputeSystemQuery) ([]ContainerProperties, error) {
	operation := "GetContainers"
	title := "HCSShim::" + operation

	queryb, err := json.Marshal(q)
	if err != nil {
		return nil, err
	}

	query := string(queryb)
	logrus.Debugf(title+" query=%s", query)

	var (
		resultp         *uint16
		computeSystemsp *uint16
	)
	err = hcsEnumerateComputeSystems(query, &computeSystemsp, &resultp)
	err = processHcsResult(err, resultp)
	if err != nil {
		return nil, err
	}

	if computeSystemsp == nil {
		return nil, ErrUnexpectedValue
	}
	computeSystemsRaw := convertAndFreeCoTaskMemBytes(computeSystemsp)
	computeSystems := []ContainerProperties{}
	if err := json.Unmarshal(computeSystemsRaw, &computeSystems); err != nil {
		return nil, err
	}

	logrus.Debugf(title + " succeeded")
	return computeSystems, nil
}

// Start synchronously starts the container.
func (container *container) Start() error {
	container.handleLock.RLock()
	defer container.handleLock.RUnlock()
	operation := "Start"
	title := "HCSShim::Container::" + operation
	logrus.Debugf(title+" id=%s", container.id)

	if container.handle == 0 {
		return makeContainerError(container, operation, "", ErrAlreadyClosed)
	}

	var resultp *uint16
	err := hcsStartComputeSystem(container.handle, "", &resultp)
	err = processAsyncHcsResult(err, resultp, container.callbackNumber, hcsNotificationSystemStartCompleted, &defaultTimeout)
	if err != nil {
		return makeContainerError(container, operation, "", err)
	}

	logrus.Debugf(title+" succeeded id=%s", container.id)
	return nil
}

// Shutdown requests a container shutdown, if IsPending() on the error returned is true,
// it may not actually be shut down until Wait() succeeds.
func (container *container) Shutdown() error {
	container.handleLock.RLock()
	defer container.handleLock.RUnlock()
	operation := "Shutdown"
	title := "HCSShim::Container::" + operation
	logrus.Debugf(title+" id=%s", container.id)

	if container.handle == 0 {
		return makeContainerError(container, operation, "", ErrAlreadyClosed)
	}

	var resultp *uint16
	err := hcsShutdownComputeSystem(container.handle, "", &resultp)
	err = processHcsResult(err, resultp)
	if err != nil {
		return makeContainerError(container, operation, "", err)
	}

	logrus.Debugf(title+" succeeded id=%s", container.id)
	return nil
}

// Terminate requests a container terminate, if IsPending() on the error returned is true,
// it may not actually be shut down until Wait() succeeds.
func (container *container) Terminate() error {
	container.handleLock.RLock()
	defer container.handleLock.RUnlock()
	operation := "Terminate"
	title := "HCSShim::Container::" + operation
	logrus.Debugf(title+" id=%s", container.id)

	if container.handle == 0 {
		return makeContainerError(container, operation, "", ErrAlreadyClosed)
	}

	var resultp *uint16
	err := hcsTerminateComputeSystem(container.handle, "", &resultp)
	err = processHcsResult(err, resultp)
	if err != nil {
		return makeContainerError(container, operation, "", err)
	}

	logrus.Debugf(title+" succeeded id=%s", container.id)
	return nil
}

// Wait synchronously waits for the container to shutdown or terminate.
func (container *container) Wait() error {
	operation := "Wait"
	title := "HCSShim::Container::" + operation
	logrus.Debugf(title+" id=%s", container.id)

	err := waitForNotification(container.callbackNumber, hcsNotificationSystemExited, nil)
	if err != nil {
		return makeContainerError(container, operation, "", err)
	}

	logrus.Debugf(title+" succeeded id=%s", container.id)
	return nil
}

// WaitTimeout synchronously waits for the container to terminate or the duration to elapse.
// If the timeout expires, IsTimeout(err) == true
func (container *container) WaitTimeout(timeout time.Duration) error {
	operation := "WaitTimeout"
	title := "HCSShim::Container::" + operation
	logrus.Debugf(title+" id=%s", container.id)

	err := waitForNotification(container.callbackNumber, hcsNotificationSystemExited, &timeout)
	if err != nil {
		return makeContainerError(container, operation, "", err)
	}

	logrus.Debugf(title+" succeeded id=%s", container.id)
	return nil
}

func (container *container) properties(query string) (*ContainerProperties, error) {
	var (
		resultp     *uint16
		propertiesp *uint16
	)
	err := hcsGetComputeSystemProperties(container.handle, query, &propertiesp, &resultp)
	err = processHcsResult(err, resultp)
	if err != nil {
		return nil, err
	}

	if propertiesp == nil {
		return nil, ErrUnexpectedValue
	}
	propertiesRaw := convertAndFreeCoTaskMemBytes(propertiesp)
	properties := &ContainerProperties{}
	if err := json.Unmarshal(propertiesRaw, properties); err != nil {
		return nil, err
	}
	return properties, nil
}

// HasPendingUpdates returns true if the container has updates pending to install
func (container *container) HasPendingUpdates() (bool, error) {
	container.handleLock.RLock()
	defer container.handleLock.RUnlock()
	operation := "HasPendingUpdates"
	title := "HCSShim::Container::" + operation
	logrus.Debugf(title+" id=%s", container.id)

	if container.handle == 0 {
		return false, makeContainerError(container, operation, "", ErrAlreadyClosed)
	}

	properties, err := container.properties(pendingUpdatesQuery)
	if err != nil {
		return false, makeContainerError(container, operation, "", err)
	}

	logrus.Debugf(title+" succeeded id=%s", container.id)
	return properties.AreUpdatesPending, nil
}

// Statistics returns statistics for the container
func (container *container) Statistics() (Statistics, error) {
	container.handleLock.RLock()
	defer container.handleLock.RUnlock()
	operation := "Statistics"
	title := "HCSShim::Container::" + operation
	logrus.Debugf(title+" id=%s", container.id)

	if container.handle == 0 {
		return Statistics{}, makeContainerError(container, operation, "", ErrAlreadyClosed)
	}

	properties, err := container.properties(statisticsQuery)
	if err != nil {
		return Statistics{}, makeContainerError(container, operation, "", err)
	}

	logrus.Debugf(title+" succeeded id=%s", container.id)
	return properties.Statistics, nil
}

// ProcessList returns an array of ProcessListItems for the container
func (container *container) ProcessList() ([]ProcessListItem, error) {
	container.handleLock.RLock()
	defer container.handleLock.RUnlock()
	operation := "ProcessList"
	title := "HCSShim::Container::" + operation
	logrus.Debugf(title+" id=%s", container.id)

	if container.handle == 0 {
		return nil, makeContainerError(container, operation, "", ErrAlreadyClosed)
	}

	properties, err := container.properties(processListQuery)
	if err != nil {
		return nil, makeContainerError(container, operation, "", err)
	}

	logrus.Debugf(title+" succeeded id=%s", container.id)
	return properties.ProcessList, nil
}

// MappedVirtualDisks returns a map of the controllers and the disks mapped
// to a container.
//
// Example of JSON returned by the query.
//{
//   "Id":"1126e8d7d279c707a666972a15976371d365eaf622c02cea2c442b84f6f550a3_svm",
//   "SystemType":"Container",
//   "RuntimeOsType":"Linux",
//   "RuntimeId":"00000000-0000-0000-0000-000000000000",
//   "State":"Running",
//   "MappedVirtualDiskControllers":{
//      "0":{
//         "MappedVirtualDisks":{
//            "2":{
//               "HostPath":"C:\\lcow\\lcow\\scratch\\1126e8d7d279c707a666972a15976371d365eaf622c02cea2c442b84f6f550a3.vhdx",
//               "ContainerPath":"/mnt/gcs/LinuxServiceVM/scratch",
//               "Lun":2,
//               "CreateInUtilityVM":true
//            },
//            "3":{
//               "HostPath":"C:\\lcow\\lcow\\1126e8d7d279c707a666972a15976371d365eaf622c02cea2c442b84f6f550a3\\sandbox.vhdx",
//               "Lun":3,
//               "CreateInUtilityVM":true,
//               "AttachOnly":true
//            }
//         }
//      }
//   }
//}
func (container *container) MappedVirtualDisks() (map[int]MappedVirtualDiskController, error) {
	container.handleLock.RLock()
	defer container.handleLock.RUnlock()
	operation := "MappedVirtualDiskList"
	title := "HCSShim::Container::" + operation
	logrus.Debugf(title+" id=%s", container.id)

	if container.handle == 0 {
		return nil, makeContainerError(container, operation, "", ErrAlreadyClosed)
	}

	properties, err := container.properties(mappedVirtualDiskQuery)
	if err != nil {
		return nil, makeContainerError(container, operation, "", err)
	}

	logrus.Debugf(title+" succeeded id=%s", container.id)
	return properties.MappedVirtualDiskControllers, nil
}

// Pause pauses the execution of the container. This feature is not enabled in TP5.
func (container *container) Pause() error {
	container.handleLock.RLock()
	defer container.handleLock.RUnlock()
	operation := "Pause"
	title := "HCSShim::Container::" + operation
	logrus.Debugf(title+" id=%s", container.id)

	if container.handle == 0 {
		return makeContainerError(container, operation, "", ErrAlreadyClosed)
	}

	var resultp *uint16
	err := hcsPauseComputeSystem(container.handle, "", &resultp)
	err = processAsyncHcsResult(err, resultp, container.callbackNumber, hcsNotificationSystemPauseCompleted, &defaultTimeout)
	if err != nil {
		return makeContainerError(container, operation, "", err)
	}

	logrus.Debugf(title+" succeeded id=%s", container.id)
	return nil
}

// Resume resumes the execution of the container. This feature is not enabled in TP5.
func (container *container) Resume() error {
	container.handleLock.RLock()
	defer container.handleLock.RUnlock()
	operation := "Resume"
	title := "HCSShim::Container::" + operation
	logrus.Debugf(title+" id=%s", container.id)

	if container.handle == 0 {
		return makeContainerError(container, operation, "", ErrAlreadyClosed)
	}

	var resultp *uint16
	err := hcsResumeComputeSystem(container.handle, "", &resultp)
	err = processAsyncHcsResult(err, resultp, container.callbackNumber, hcsNotificationSystemResumeCompleted, &defaultTimeout)
	if err != nil {
		return makeContainerError(container, operation, "", err)
	}

	logrus.Debugf(title+" succeeded id=%s", container.id)
	return nil
}

// CreateProcess launches a new process within the container.
func (container *container) CreateProcess(c *ProcessConfig) (Process, error) {
	container.handleLock.RLock()
	defer container.handleLock.RUnlock()
	operation := "CreateProcess"
	title := "HCSShim::Container::" + operation
	var (
		processInfo   hcsProcessInformation
		processHandle hcsProcess
		resultp       *uint16
	)

	if container.handle == 0 {
		return nil, makeContainerError(container, operation, "", ErrAlreadyClosed)
	}

	// If we are not emulating a console, ignore any console size passed to us
	if !c.EmulateConsole {
		c.ConsoleSize[0] = 0
		c.ConsoleSize[1] = 0
	}

	configurationb, err := json.Marshal(c)
	if err != nil {
		return nil, makeContainerError(container, operation, "", err)
	}

	configuration := string(configurationb)
	logrus.Debugf(title+" id=%s config=%s", container.id, configuration)

	err = hcsCreateProcess(container.handle, configuration, &processInfo, &processHandle, &resultp)
	err = processHcsResult(err, resultp)
	if err != nil {
		return nil, makeContainerError(container, operation, configuration, err)
	}

	process := &process{
		handle:    processHandle,
		processID: int(processInfo.ProcessId),
		container: container,
		cachedPipes: &cachedPipes{
			stdIn:  processInfo.StdInput,
			stdOut: processInfo.StdOutput,
			stdErr: processInfo.StdError,
		},
	}

	if err := process.registerCallback(); err != nil {
		return nil, makeContainerError(container, operation, "", err)
	}

	logrus.Debugf(title+" succeeded id=%s processid=%d", container.id, process.processID)
	return process, nil
}

// OpenProcess gets an interface to an existing process within the container.
func (container *container) OpenProcess(pid int) (Process, error) {
	container.handleLock.RLock()
	defer container.handleLock.RUnlock()
	operation := "OpenProcess"
	title := "HCSShim::Container::" + operation
	logrus.Debugf(title+" id=%s, processid=%d", container.id, pid)
	var (
		processHandle hcsProcess
		resultp       *uint16
	)

	if container.handle == 0 {
		return nil, makeContainerError(container, operation, "", ErrAlreadyClosed)
	}

	err := hcsOpenProcess(container.handle, uint32(pid), &processHandle, &resultp)
	err = processHcsResult(err, resultp)
	if err != nil {
		return nil, makeContainerError(container, operation, "", err)
	}

	process := &process{
		handle:    processHandle,
		processID: pid,
		container: container,
	}

	if err := process.registerCallback(); err != nil {
		return nil, makeContainerError(container, operation, "", err)
	}

	logrus.Debugf(title+" succeeded id=%s processid=%s", container.id, process.processID)
	return process, nil
}

// Close cleans up any state associated with the container but does not terminate or wait for it.
func (container *container) Close() error {
	container.handleLock.Lock()
	defer container.handleLock.Unlock()
	operation := "Close"
	title := "HCSShim::Container::" + operation
	logrus.Debugf(title+" id=%s", container.id)

	// Don't double free this
	if container.handle == 0 {
		return nil
	}

	if err := container.unregisterCallback(); err != nil {
		return makeContainerError(container, operation, "", err)
	}

	if err := hcsCloseComputeSystem(container.handle); err != nil {
		return makeContainerError(container, operation, "", err)
	}

	container.handle = 0

	logrus.Debugf(title+" succeeded id=%s", container.id)
	return nil
}

func (container *container) registerCallback() error {
	context := &notifcationWatcherContext{
		channels: newChannels(),
	}

	callbackMapLock.Lock()
	callbackNumber := nextCallback
	nextCallback++
	callbackMap[callbackNumber] = context
	callbackMapLock.Unlock()

	var callbackHandle hcsCallback
	err := hcsRegisterComputeSystemCallback(container.handle, notificationWatcherCallback, callbackNumber, &callbackHandle)
	if err != nil {
		return err
	}
	context.handle = callbackHandle
	container.callbackNumber = callbackNumber

	return nil
}

func (container *container) unregisterCallback() error {
	callbackNumber := container.callbackNumber

	callbackMapLock.RLock()
	context := callbackMap[callbackNumber]
	callbackMapLock.RUnlock()

	if context == nil {
		return nil
	}

	handle := context.handle

	if handle == 0 {
		return nil
	}

	// hcsUnregisterComputeSystemCallback has its own syncronization
	// to wait for all callbacks to complete. We must NOT hold the callbackMapLock.
	err := hcsUnregisterComputeSystemCallback(handle)
	if err != nil {
		return err
	}

	closeChannels(context.channels)

	callbackMapLock.Lock()
	callbackMap[callbackNumber] = nil
	callbackMapLock.Unlock()

	handle = 0

	return nil
}

// Modifies the System by sending a request to HCS
func (container *container) Modify(config *ResourceModificationRequestResponse) error {
	container.handleLock.RLock()
	defer container.handleLock.RUnlock()
	operation := "Modify"
	title := "HCSShim::Container::" + operation

	if container.handle == 0 {
		return makeContainerError(container, operation, "", ErrAlreadyClosed)
	}

	requestJSON, err := json.Marshal(config)
	if err != nil {
		return err
	}

	requestString := string(requestJSON)
	logrus.Debugf(title+" id=%s request=%s", container.id, requestString)

	var resultp *uint16
	err = hcsModifyComputeSystem(container.handle, requestString, &resultp)
	err = processHcsResult(err, resultp)
	if err != nil {
		return makeContainerError(container, operation, "", err)
	}
	logrus.Debugf(title+" succeeded id=%s", container.id)
	return nil
}
