package hcs

import (
	"encoding/json"
	"os"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/Microsoft/hcsshim/internal/logfields"
	"github.com/Microsoft/hcsshim/internal/schema1"
	"github.com/Microsoft/hcsshim/internal/timeout"
	"github.com/sirupsen/logrus"
)

// currentContainerStarts is used to limit the number of concurrent container
// starts.
var currentContainerStarts containerStarts

type containerStarts struct {
	maxParallel int
	inProgress  int
	sync.Mutex
}

func init() {
	mpsS := os.Getenv("HCSSHIM_MAX_PARALLEL_START")
	if len(mpsS) > 0 {
		mpsI, err := strconv.Atoi(mpsS)
		if err != nil || mpsI < 0 {
			return
		}
		currentContainerStarts.maxParallel = mpsI
	}
}

type System struct {
	handleLock     sync.RWMutex
	handle         hcsSystem
	id             string
	callbackNumber uintptr

	logctx logrus.Fields
}

func newSystem(id string) *System {
	return &System{
		id: id,
		logctx: logrus.Fields{
			logfields.ContainerID: id,
		},
	}
}

func (computeSystem *System) logOperationBegin(operation string) {
	logOperationBegin(
		computeSystem.logctx,
		operation+" - Begin Operation")
}

func (computeSystem *System) logOperationEnd(operation string, err error) {
	var result string
	if err == nil {
		result = "Success"
	} else {
		result = "Error"
	}

	logOperationEnd(
		computeSystem.logctx,
		operation+" - End Operation - "+result,
		err)
}

// CreateComputeSystem creates a new compute system with the given configuration but does not start it.
func CreateComputeSystem(id string, hcsDocumentInterface interface{}) (_ *System, err error) {
	operation := "hcsshim::CreateComputeSystem"

	computeSystem := newSystem(id)
	computeSystem.logOperationBegin(operation)
	defer func() { computeSystem.logOperationEnd(operation, err) }()

	hcsDocumentB, err := json.Marshal(hcsDocumentInterface)
	if err != nil {
		return nil, err
	}

	hcsDocument := string(hcsDocumentB)

	logrus.WithFields(computeSystem.logctx).
		WithField(logfields.JSON, hcsDocument).
		Debug("HCS ComputeSystem Document")

	var (
		resultp     *uint16
		identity    syscall.Handle
		createError error
	)
	syscallWatcher(computeSystem.logctx, func() {
		createError = hcsCreateComputeSystem(id, hcsDocument, identity, &computeSystem.handle, &resultp)
	})

	if createError == nil || IsPending(createError) {
		if err = computeSystem.registerCallback(); err != nil {
			// Terminate the compute system if it still exists. We're okay to
			// ignore a failure here.
			computeSystem.Terminate()
			return nil, makeSystemError(computeSystem, operation, "", err, nil)
		}
	}

	events, err := processAsyncHcsResult(createError, resultp, computeSystem.callbackNumber, hcsNotificationSystemCreateCompleted, &timeout.SystemCreate)
	if err != nil {
		if err == ErrTimeout {
			// Terminate the compute system if it still exists. We're okay to
			// ignore a failure here.
			computeSystem.Terminate()
		}
		return nil, makeSystemError(computeSystem, operation, hcsDocument, err, events)
	}

	return computeSystem, nil
}

// OpenComputeSystem opens an existing compute system by ID.
func OpenComputeSystem(id string) (_ *System, err error) {
	operation := "hcsshim::OpenComputeSystem"

	computeSystem := newSystem(id)
	computeSystem.logOperationBegin(operation)
	defer func() {
		if IsNotExist(err) {
			computeSystem.logOperationEnd(operation, nil)
		} else {
			computeSystem.logOperationEnd(operation, err)
		}
	}()

	var (
		handle  hcsSystem
		resultp *uint16
	)
	err = hcsOpenComputeSystem(id, &handle, &resultp)
	events := processHcsResult(resultp)
	if err != nil {
		return nil, makeSystemError(computeSystem, operation, "", err, events)
	}

	computeSystem.handle = handle

	if err = computeSystem.registerCallback(); err != nil {
		return nil, makeSystemError(computeSystem, operation, "", err, nil)
	}

	return computeSystem, nil
}

// GetComputeSystems gets a list of the compute systems on the system that match the query
func GetComputeSystems(q schema1.ComputeSystemQuery) (_ []schema1.ContainerProperties, err error) {
	operation := "hcsshim::GetComputeSystems"
	fields := logrus.Fields{}
	logOperationBegin(
		fields,
		operation+" - Begin Operation")

	defer func() {
		var result string
		if err == nil {
			result = "Success"
		} else {
			result = "Error"
		}

		logOperationEnd(
			fields,
			operation+" - End Operation - "+result,
			err)
	}()

	queryb, err := json.Marshal(q)
	if err != nil {
		return nil, err
	}

	query := string(queryb)

	logrus.WithFields(fields).
		WithField(logfields.JSON, query).
		Debug("HCS ComputeSystem Query")

	var (
		resultp         *uint16
		computeSystemsp *uint16
	)

	syscallWatcher(fields, func() {
		err = hcsEnumerateComputeSystems(query, &computeSystemsp, &resultp)
	})
	events := processHcsResult(resultp)
	if err != nil {
		return nil, &HcsError{Op: operation, Err: err, Events: events}
	}

	if computeSystemsp == nil {
		return nil, ErrUnexpectedValue
	}
	computeSystemsRaw := interop.ConvertAndFreeCoTaskMemBytes(computeSystemsp)
	computeSystems := []schema1.ContainerProperties{}
	if err = json.Unmarshal(computeSystemsRaw, &computeSystems); err != nil {
		return nil, err
	}

	return computeSystems, nil
}

// Start synchronously starts the computeSystem.
func (computeSystem *System) Start() (err error) {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	operation := "hcsshim::ComputeSystem::Start"
	computeSystem.logOperationBegin(operation)
	defer func() { computeSystem.logOperationEnd(operation, err) }()

	if computeSystem.handle == 0 {
		return makeSystemError(computeSystem, "Start", "", ErrAlreadyClosed, nil)
	}

	// This is a very simple backoff-retry loop to limit the number
	// of parallel container starts if environment variable
	// HCSSHIM_MAX_PARALLEL_START is set to a positive integer.
	// It should generally only be used as a workaround to various
	// platform issues that exist between RS1 and RS4 as of Aug 2018
	if currentContainerStarts.maxParallel > 0 {
		for {
			currentContainerStarts.Lock()
			if currentContainerStarts.inProgress < currentContainerStarts.maxParallel {
				currentContainerStarts.inProgress++
				currentContainerStarts.Unlock()
				break
			}
			if currentContainerStarts.inProgress == currentContainerStarts.maxParallel {
				currentContainerStarts.Unlock()
				time.Sleep(100 * time.Millisecond)
			}
		}
		// Make sure we decrement the count when we are done.
		defer func() {
			currentContainerStarts.Lock()
			currentContainerStarts.inProgress--
			currentContainerStarts.Unlock()
		}()
	}

	var resultp *uint16
	syscallWatcher(computeSystem.logctx, func() {
		err = hcsStartComputeSystem(computeSystem.handle, "", &resultp)
	})
	events, err := processAsyncHcsResult(err, resultp, computeSystem.callbackNumber, hcsNotificationSystemStartCompleted, &timeout.SystemStart)
	if err != nil {
		return makeSystemError(computeSystem, "Start", "", err, events)
	}

	return nil
}

// ID returns the compute system's identifier.
func (computeSystem *System) ID() string {
	return computeSystem.id
}

// Shutdown requests a compute system shutdown, if IsPending() on the error returned is true,
// it may not actually be shut down until Wait() succeeds.
func (computeSystem *System) Shutdown() (err error) {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	operation := "hcsshim::ComputeSystem::Shutdown"
	computeSystem.logOperationBegin(operation)
	defer func() {
		if IsAlreadyStopped(err) {
			computeSystem.logOperationEnd(operation, nil)
		} else {
			computeSystem.logOperationEnd(operation, err)
		}
	}()

	if computeSystem.handle == 0 {
		return makeSystemError(computeSystem, "Shutdown", "", ErrAlreadyClosed, nil)
	}

	var resultp *uint16
	syscallWatcher(computeSystem.logctx, func() {
		err = hcsShutdownComputeSystem(computeSystem.handle, "", &resultp)
	})
	events := processHcsResult(resultp)
	if err != nil {
		return makeSystemError(computeSystem, "Shutdown", "", err, events)
	}

	return nil
}

// Terminate requests a compute system terminate, if IsPending() on the error returned is true,
// it may not actually be shut down until Wait() succeeds.
func (computeSystem *System) Terminate() (err error) {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	operation := "hcsshim::ComputeSystem::Terminate"
	computeSystem.logOperationBegin(operation)
	defer func() {
		if IsPending(err) {
			computeSystem.logOperationEnd(operation, nil)
		} else {
			computeSystem.logOperationEnd(operation, err)
		}
	}()

	if computeSystem.handle == 0 {
		return makeSystemError(computeSystem, "Terminate", "", ErrAlreadyClosed, nil)
	}

	var resultp *uint16
	syscallWatcher(computeSystem.logctx, func() {
		err = hcsTerminateComputeSystem(computeSystem.handle, "", &resultp)
	})
	events := processHcsResult(resultp)
	if err != nil && err != ErrVmcomputeAlreadyStopped {
		return makeSystemError(computeSystem, "Terminate", "", err, events)
	}

	return nil
}

// Wait synchronously waits for the compute system to shutdown or terminate.
func (computeSystem *System) Wait() (err error) {
	operation := "hcsshim::ComputeSystem::Wait"
	computeSystem.logOperationBegin(operation)
	defer func() { computeSystem.logOperationEnd(operation, err) }()

	err = waitForNotification(computeSystem.callbackNumber, hcsNotificationSystemExited, nil)
	if err != nil {
		return makeSystemError(computeSystem, "Wait", "", err, nil)
	}

	return nil
}

// WaitExpectedError synchronously waits for the compute system to shutdown or
// terminate, and ignores the passed error if it occurs.
func (computeSystem *System) WaitExpectedError(expected error) (err error) {
	operation := "hcsshim::ComputeSystem::WaitExpectedError"
	computeSystem.logOperationBegin(operation)
	defer func() { computeSystem.logOperationEnd(operation, err) }()

	err = waitForNotification(computeSystem.callbackNumber, hcsNotificationSystemExited, nil)
	if err != nil && getInnerError(err) != expected {
		return makeSystemError(computeSystem, "WaitExpectedError", "", err, nil)
	}

	return nil
}

// WaitTimeout synchronously waits for the compute system to terminate or the duration to elapse.
// If the timeout expires, IsTimeout(err) == true
func (computeSystem *System) WaitTimeout(timeout time.Duration) (err error) {
	operation := "hcsshim::ComputeSystem::WaitTimeout"
	computeSystem.logOperationBegin(operation)
	defer func() { computeSystem.logOperationEnd(operation, err) }()

	err = waitForNotification(computeSystem.callbackNumber, hcsNotificationSystemExited, &timeout)
	if err != nil {
		return makeSystemError(computeSystem, "WaitTimeout", "", err, nil)
	}

	return nil
}

func (computeSystem *System) Properties(types ...schema1.PropertyType) (_ *schema1.ContainerProperties, err error) {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	operation := "hcsshim::ComputeSystem::Properties"
	computeSystem.logOperationBegin(operation)
	defer func() { computeSystem.logOperationEnd(operation, err) }()

	queryj, err := json.Marshal(schema1.PropertyQuery{types})
	if err != nil {
		return nil, makeSystemError(computeSystem, "Properties", "", err, nil)
	}

	logrus.WithFields(computeSystem.logctx).
		WithField(logfields.JSON, queryj).
		Debug("HCS ComputeSystem Properties Query")

	var resultp, propertiesp *uint16
	syscallWatcher(computeSystem.logctx, func() {
		err = hcsGetComputeSystemProperties(computeSystem.handle, string(queryj), &propertiesp, &resultp)
	})
	events := processHcsResult(resultp)
	if err != nil {
		return nil, makeSystemError(computeSystem, "Properties", "", err, events)
	}

	if propertiesp == nil {
		return nil, ErrUnexpectedValue
	}
	propertiesRaw := interop.ConvertAndFreeCoTaskMemBytes(propertiesp)
	properties := &schema1.ContainerProperties{}
	if err := json.Unmarshal(propertiesRaw, properties); err != nil {
		return nil, makeSystemError(computeSystem, "Properties", "", err, nil)
	}

	return properties, nil
}

// Pause pauses the execution of the computeSystem. This feature is not enabled in TP5.
func (computeSystem *System) Pause() (err error) {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	operation := "hcsshim::ComputeSystem::Pause"
	computeSystem.logOperationBegin(operation)
	defer func() { computeSystem.logOperationEnd(operation, err) }()

	if computeSystem.handle == 0 {
		return makeSystemError(computeSystem, "Pause", "", ErrAlreadyClosed, nil)
	}

	var resultp *uint16
	syscallWatcher(computeSystem.logctx, func() {
		err = hcsPauseComputeSystem(computeSystem.handle, "", &resultp)
	})
	events, err := processAsyncHcsResult(err, resultp, computeSystem.callbackNumber, hcsNotificationSystemPauseCompleted, &timeout.SystemPause)
	if err != nil {
		return makeSystemError(computeSystem, "Pause", "", err, events)
	}

	return nil
}

// Resume resumes the execution of the computeSystem. This feature is not enabled in TP5.
func (computeSystem *System) Resume() (err error) {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	operation := "hcsshim::ComputeSystem::Resume"
	computeSystem.logOperationBegin(operation)
	defer func() { computeSystem.logOperationEnd(operation, err) }()

	if computeSystem.handle == 0 {
		return makeSystemError(computeSystem, "Resume", "", ErrAlreadyClosed, nil)
	}

	var resultp *uint16
	syscallWatcher(computeSystem.logctx, func() {
		err = hcsResumeComputeSystem(computeSystem.handle, "", &resultp)
	})
	events, err := processAsyncHcsResult(err, resultp, computeSystem.callbackNumber, hcsNotificationSystemResumeCompleted, &timeout.SystemResume)
	if err != nil {
		return makeSystemError(computeSystem, "Resume", "", err, events)
	}

	return nil
}

// CreateProcess launches a new process within the computeSystem.
func (computeSystem *System) CreateProcess(c interface{}) (_ *Process, err error) {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	operation := "hcsshim::ComputeSystem::CreateProcess"
	computeSystem.logOperationBegin(operation)
	defer func() { computeSystem.logOperationEnd(operation, err) }()

	var (
		processInfo   hcsProcessInformation
		processHandle hcsProcess
		resultp       *uint16
	)

	if computeSystem.handle == 0 {
		return nil, makeSystemError(computeSystem, "CreateProcess", "", ErrAlreadyClosed, nil)
	}

	configurationb, err := json.Marshal(c)
	if err != nil {
		return nil, makeSystemError(computeSystem, "CreateProcess", "", err, nil)
	}

	configuration := string(configurationb)

	logrus.WithFields(computeSystem.logctx).
		WithField(logfields.JSON, configuration).
		Debug("HCS ComputeSystem Process Document")

	syscallWatcher(computeSystem.logctx, func() {
		err = hcsCreateProcess(computeSystem.handle, configuration, &processInfo, &processHandle, &resultp)
	})
	events := processHcsResult(resultp)
	if err != nil {
		return nil, makeSystemError(computeSystem, "CreateProcess", configuration, err, events)
	}

	logrus.WithFields(computeSystem.logctx).
		WithField(logfields.ProcessID, processInfo.ProcessId).
		Debug("HCS ComputeSystem CreateProcess PID")

	process := newProcess(processHandle, int(processInfo.ProcessId), computeSystem)
	process.cachedPipes = &cachedPipes{
		stdIn:  processInfo.StdInput,
		stdOut: processInfo.StdOutput,
		stdErr: processInfo.StdError,
	}

	if err = process.registerCallback(); err != nil {
		return nil, makeSystemError(computeSystem, "CreateProcess", "", err, nil)
	}

	return process, nil
}

// OpenProcess gets an interface to an existing process within the computeSystem.
func (computeSystem *System) OpenProcess(pid int) (_ *Process, err error) {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	// Add PID for the context of this operation
	computeSystem.logctx[logfields.ProcessID] = pid
	defer delete(computeSystem.logctx, logfields.ProcessID)

	operation := "hcsshim::ComputeSystem::OpenProcess"
	computeSystem.logOperationBegin(operation)
	defer func() { computeSystem.logOperationEnd(operation, err) }()

	var (
		processHandle hcsProcess
		resultp       *uint16
	)

	if computeSystem.handle == 0 {
		return nil, makeSystemError(computeSystem, "OpenProcess", "", ErrAlreadyClosed, nil)
	}

	syscallWatcher(computeSystem.logctx, func() {
		err = hcsOpenProcess(computeSystem.handle, uint32(pid), &processHandle, &resultp)
	})
	events := processHcsResult(resultp)
	if err != nil {
		return nil, makeSystemError(computeSystem, "OpenProcess", "", err, events)
	}

	process := newProcess(processHandle, pid, computeSystem)
	if err = process.registerCallback(); err != nil {
		return nil, makeSystemError(computeSystem, "OpenProcess", "", err, nil)
	}

	return process, nil
}

// Close cleans up any state associated with the compute system but does not terminate or wait for it.
func (computeSystem *System) Close() (err error) {
	computeSystem.handleLock.Lock()
	defer computeSystem.handleLock.Unlock()

	operation := "hcsshim::ComputeSystem::Close"
	computeSystem.logOperationBegin(operation)
	defer func() { computeSystem.logOperationEnd(operation, err) }()

	// Don't double free this
	if computeSystem.handle == 0 {
		return nil
	}

	if err = computeSystem.unregisterCallback(); err != nil {
		return makeSystemError(computeSystem, "Close", "", err, nil)
	}

	syscallWatcher(computeSystem.logctx, func() {
		err = hcsCloseComputeSystem(computeSystem.handle)
	})
	if err != nil {
		return makeSystemError(computeSystem, "Close", "", err, nil)
	}

	computeSystem.handle = 0

	return nil
}

func (computeSystem *System) registerCallback() error {
	context := &notifcationWatcherContext{
		channels: newChannels(),
	}

	callbackMapLock.Lock()
	callbackNumber := nextCallback
	nextCallback++
	callbackMap[callbackNumber] = context
	callbackMapLock.Unlock()

	var callbackHandle hcsCallback
	err := hcsRegisterComputeSystemCallback(computeSystem.handle, notificationWatcherCallback, callbackNumber, &callbackHandle)
	if err != nil {
		return err
	}
	context.handle = callbackHandle
	computeSystem.callbackNumber = callbackNumber

	return nil
}

func (computeSystem *System) unregisterCallback() error {
	callbackNumber := computeSystem.callbackNumber

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

// Modify the System by sending a request to HCS
func (computeSystem *System) Modify(config interface{}) (err error) {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	operation := "hcsshim::ComputeSystem::Modify"
	computeSystem.logOperationBegin(operation)
	defer func() { computeSystem.logOperationEnd(operation, err) }()

	if computeSystem.handle == 0 {
		return makeSystemError(computeSystem, "Modify", "", ErrAlreadyClosed, nil)
	}

	requestJSON, err := json.Marshal(config)
	if err != nil {
		return err
	}

	requestString := string(requestJSON)

	logrus.WithFields(computeSystem.logctx).
		WithField(logfields.JSON, requestString).
		Debug("HCS ComputeSystem Modify Document")

	var resultp *uint16
	syscallWatcher(computeSystem.logctx, func() {
		err = hcsModifyComputeSystem(computeSystem.handle, requestString, &resultp)
	})
	events := processHcsResult(resultp)
	if err != nil {
		return makeSystemError(computeSystem, "Modify", requestString, err, events)
	}

	return nil
}
