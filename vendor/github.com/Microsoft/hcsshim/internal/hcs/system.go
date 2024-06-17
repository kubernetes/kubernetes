package hcs

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"syscall"

	"github.com/Microsoft/hcsshim/internal/cow"
	"github.com/Microsoft/hcsshim/internal/hcs/schema1"
	hcsschema "github.com/Microsoft/hcsshim/internal/hcs/schema2"
	"github.com/Microsoft/hcsshim/internal/log"
	"github.com/Microsoft/hcsshim/internal/oc"
	"github.com/Microsoft/hcsshim/internal/timeout"
	"github.com/Microsoft/hcsshim/internal/vmcompute"
	"go.opencensus.io/trace"
)

type System struct {
	handleLock     sync.RWMutex
	handle         vmcompute.HcsSystem
	id             string
	callbackNumber uintptr

	closedWaitOnce sync.Once
	waitBlock      chan struct{}
	waitError      error
	exitError      error
	os, typ        string
}

func newSystem(id string) *System {
	return &System{
		id:        id,
		waitBlock: make(chan struct{}),
	}
}

// CreateComputeSystem creates a new compute system with the given configuration but does not start it.
func CreateComputeSystem(ctx context.Context, id string, hcsDocumentInterface interface{}) (_ *System, err error) {
	operation := "hcs::CreateComputeSystem"

	// hcsCreateComputeSystemContext is an async operation. Start the outer span
	// here to measure the full create time.
	ctx, span := trace.StartSpan(ctx, operation)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("cid", id))

	computeSystem := newSystem(id)

	hcsDocumentB, err := json.Marshal(hcsDocumentInterface)
	if err != nil {
		return nil, err
	}

	hcsDocument := string(hcsDocumentB)

	var (
		identity    syscall.Handle
		resultJSON  string
		createError error
	)
	computeSystem.handle, resultJSON, createError = vmcompute.HcsCreateComputeSystem(ctx, id, hcsDocument, identity)
	if createError == nil || IsPending(createError) {
		defer func() {
			if err != nil {
				computeSystem.Close()
			}
		}()
		if err = computeSystem.registerCallback(ctx); err != nil {
			// Terminate the compute system if it still exists. We're okay to
			// ignore a failure here.
			_ = computeSystem.Terminate(ctx)
			return nil, makeSystemError(computeSystem, operation, err, nil)
		}
	}

	events, err := processAsyncHcsResult(ctx, createError, resultJSON, computeSystem.callbackNumber, hcsNotificationSystemCreateCompleted, &timeout.SystemCreate)
	if err != nil {
		if err == ErrTimeout {
			// Terminate the compute system if it still exists. We're okay to
			// ignore a failure here.
			_ = computeSystem.Terminate(ctx)
		}
		return nil, makeSystemError(computeSystem, operation, err, events)
	}
	go computeSystem.waitBackground()
	if err = computeSystem.getCachedProperties(ctx); err != nil {
		return nil, err
	}
	return computeSystem, nil
}

// OpenComputeSystem opens an existing compute system by ID.
func OpenComputeSystem(ctx context.Context, id string) (*System, error) {
	operation := "hcs::OpenComputeSystem"

	computeSystem := newSystem(id)
	handle, resultJSON, err := vmcompute.HcsOpenComputeSystem(ctx, id)
	events := processHcsResult(ctx, resultJSON)
	if err != nil {
		return nil, makeSystemError(computeSystem, operation, err, events)
	}
	computeSystem.handle = handle
	defer func() {
		if err != nil {
			computeSystem.Close()
		}
	}()
	if err = computeSystem.registerCallback(ctx); err != nil {
		return nil, makeSystemError(computeSystem, operation, err, nil)
	}
	go computeSystem.waitBackground()
	if err = computeSystem.getCachedProperties(ctx); err != nil {
		return nil, err
	}
	return computeSystem, nil
}

func (computeSystem *System) getCachedProperties(ctx context.Context) error {
	props, err := computeSystem.Properties(ctx)
	if err != nil {
		return err
	}
	computeSystem.typ = strings.ToLower(props.SystemType)
	computeSystem.os = strings.ToLower(props.RuntimeOSType)
	if computeSystem.os == "" && computeSystem.typ == "container" {
		// Pre-RS5 HCS did not return the OS, but it only supported containers
		// that ran Windows.
		computeSystem.os = "windows"
	}
	return nil
}

// OS returns the operating system of the compute system, "linux" or "windows".
func (computeSystem *System) OS() string {
	return computeSystem.os
}

// IsOCI returns whether processes in the compute system should be created via
// OCI.
func (computeSystem *System) IsOCI() bool {
	return computeSystem.os == "linux" && computeSystem.typ == "container"
}

// GetComputeSystems gets a list of the compute systems on the system that match the query
func GetComputeSystems(ctx context.Context, q schema1.ComputeSystemQuery) ([]schema1.ContainerProperties, error) {
	operation := "hcs::GetComputeSystems"

	queryb, err := json.Marshal(q)
	if err != nil {
		return nil, err
	}

	computeSystemsJSON, resultJSON, err := vmcompute.HcsEnumerateComputeSystems(ctx, string(queryb))
	events := processHcsResult(ctx, resultJSON)
	if err != nil {
		return nil, &HcsError{Op: operation, Err: err, Events: events}
	}

	if computeSystemsJSON == "" {
		return nil, ErrUnexpectedValue
	}
	computeSystems := []schema1.ContainerProperties{}
	if err = json.Unmarshal([]byte(computeSystemsJSON), &computeSystems); err != nil {
		return nil, err
	}

	return computeSystems, nil
}

// Start synchronously starts the computeSystem.
func (computeSystem *System) Start(ctx context.Context) (err error) {
	operation := "hcs::System::Start"

	// hcsStartComputeSystemContext is an async operation. Start the outer span
	// here to measure the full start time.
	ctx, span := trace.StartSpan(ctx, operation)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("cid", computeSystem.id))

	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	if computeSystem.handle == 0 {
		return makeSystemError(computeSystem, operation, ErrAlreadyClosed, nil)
	}

	resultJSON, err := vmcompute.HcsStartComputeSystem(ctx, computeSystem.handle, "")
	events, err := processAsyncHcsResult(ctx, err, resultJSON, computeSystem.callbackNumber, hcsNotificationSystemStartCompleted, &timeout.SystemStart)
	if err != nil {
		return makeSystemError(computeSystem, operation, err, events)
	}

	return nil
}

// ID returns the compute system's identifier.
func (computeSystem *System) ID() string {
	return computeSystem.id
}

// Shutdown requests a compute system shutdown.
func (computeSystem *System) Shutdown(ctx context.Context) error {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	operation := "hcs::System::Shutdown"

	if computeSystem.handle == 0 {
		return nil
	}

	resultJSON, err := vmcompute.HcsShutdownComputeSystem(ctx, computeSystem.handle, "")
	events := processHcsResult(ctx, resultJSON)
	switch err {
	case nil, ErrVmcomputeAlreadyStopped, ErrComputeSystemDoesNotExist, ErrVmcomputeOperationPending:
	default:
		return makeSystemError(computeSystem, operation, err, events)
	}
	return nil
}

// Terminate requests a compute system terminate.
func (computeSystem *System) Terminate(ctx context.Context) error {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	operation := "hcs::System::Terminate"

	if computeSystem.handle == 0 {
		return nil
	}

	resultJSON, err := vmcompute.HcsTerminateComputeSystem(ctx, computeSystem.handle, "")
	events := processHcsResult(ctx, resultJSON)
	switch err {
	case nil, ErrVmcomputeAlreadyStopped, ErrComputeSystemDoesNotExist, ErrVmcomputeOperationPending:
	default:
		return makeSystemError(computeSystem, operation, err, events)
	}
	return nil
}

// waitBackground waits for the compute system exit notification. Once received
// sets `computeSystem.waitError` (if any) and unblocks all `Wait` calls.
//
// This MUST be called exactly once per `computeSystem.handle` but `Wait` is
// safe to call multiple times.
func (computeSystem *System) waitBackground() {
	operation := "hcs::System::waitBackground"
	ctx, span := trace.StartSpan(context.Background(), operation)
	defer span.End()
	span.AddAttributes(trace.StringAttribute("cid", computeSystem.id))

	err := waitForNotification(ctx, computeSystem.callbackNumber, hcsNotificationSystemExited, nil)
	switch err {
	case nil:
		log.G(ctx).Debug("system exited")
	case ErrVmcomputeUnexpectedExit:
		log.G(ctx).Debug("unexpected system exit")
		computeSystem.exitError = makeSystemError(computeSystem, operation, err, nil)
		err = nil
	default:
		err = makeSystemError(computeSystem, operation, err, nil)
	}
	computeSystem.closedWaitOnce.Do(func() {
		computeSystem.waitError = err
		close(computeSystem.waitBlock)
	})
	oc.SetSpanStatus(span, err)
}

func (computeSystem *System) WaitChannel() <-chan struct{} {
	return computeSystem.waitBlock
}

func (computeSystem *System) WaitError() error {
	return computeSystem.waitError
}

// Wait synchronously waits for the compute system to shutdown or terminate. If
// the compute system has already exited returns the previous error (if any).
func (computeSystem *System) Wait() error {
	<-computeSystem.WaitChannel()
	return computeSystem.WaitError()
}

// ExitError returns an error describing the reason the compute system terminated.
func (computeSystem *System) ExitError() error {
	select {
	case <-computeSystem.waitBlock:
		if computeSystem.waitError != nil {
			return computeSystem.waitError
		}
		return computeSystem.exitError
	default:
		return errors.New("container not exited")
	}
}

// Properties returns the requested container properties targeting a V1 schema container.
func (computeSystem *System) Properties(ctx context.Context, types ...schema1.PropertyType) (*schema1.ContainerProperties, error) {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	operation := "hcs::System::Properties"

	queryBytes, err := json.Marshal(schema1.PropertyQuery{PropertyTypes: types})
	if err != nil {
		return nil, makeSystemError(computeSystem, operation, err, nil)
	}

	propertiesJSON, resultJSON, err := vmcompute.HcsGetComputeSystemProperties(ctx, computeSystem.handle, string(queryBytes))
	events := processHcsResult(ctx, resultJSON)
	if err != nil {
		return nil, makeSystemError(computeSystem, operation, err, events)
	}

	if propertiesJSON == "" {
		return nil, ErrUnexpectedValue
	}
	properties := &schema1.ContainerProperties{}
	if err := json.Unmarshal([]byte(propertiesJSON), properties); err != nil {
		return nil, makeSystemError(computeSystem, operation, err, nil)
	}

	return properties, nil
}

// PropertiesV2 returns the requested container properties targeting a V2 schema container.
func (computeSystem *System) PropertiesV2(ctx context.Context, types ...hcsschema.PropertyType) (*hcsschema.Properties, error) {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	operation := "hcs::System::PropertiesV2"

	queryBytes, err := json.Marshal(hcsschema.PropertyQuery{PropertyTypes: types})
	if err != nil {
		return nil, makeSystemError(computeSystem, operation, err, nil)
	}

	propertiesJSON, resultJSON, err := vmcompute.HcsGetComputeSystemProperties(ctx, computeSystem.handle, string(queryBytes))
	events := processHcsResult(ctx, resultJSON)
	if err != nil {
		return nil, makeSystemError(computeSystem, operation, err, events)
	}

	if propertiesJSON == "" {
		return nil, ErrUnexpectedValue
	}
	properties := &hcsschema.Properties{}
	if err := json.Unmarshal([]byte(propertiesJSON), properties); err != nil {
		return nil, makeSystemError(computeSystem, operation, err, nil)
	}

	return properties, nil
}

// Pause pauses the execution of the computeSystem. This feature is not enabled in TP5.
func (computeSystem *System) Pause(ctx context.Context) (err error) {
	operation := "hcs::System::Pause"

	// hcsPauseComputeSystemContext is an async peration. Start the outer span
	// here to measure the full pause time.
	ctx, span := trace.StartSpan(ctx, operation)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("cid", computeSystem.id))

	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	if computeSystem.handle == 0 {
		return makeSystemError(computeSystem, operation, ErrAlreadyClosed, nil)
	}

	resultJSON, err := vmcompute.HcsPauseComputeSystem(ctx, computeSystem.handle, "")
	events, err := processAsyncHcsResult(ctx, err, resultJSON, computeSystem.callbackNumber, hcsNotificationSystemPauseCompleted, &timeout.SystemPause)
	if err != nil {
		return makeSystemError(computeSystem, operation, err, events)
	}

	return nil
}

// Resume resumes the execution of the computeSystem. This feature is not enabled in TP5.
func (computeSystem *System) Resume(ctx context.Context) (err error) {
	operation := "hcs::System::Resume"

	// hcsResumeComputeSystemContext is an async operation. Start the outer span
	// here to measure the full restore time.
	ctx, span := trace.StartSpan(ctx, operation)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("cid", computeSystem.id))

	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	if computeSystem.handle == 0 {
		return makeSystemError(computeSystem, operation, ErrAlreadyClosed, nil)
	}

	resultJSON, err := vmcompute.HcsResumeComputeSystem(ctx, computeSystem.handle, "")
	events, err := processAsyncHcsResult(ctx, err, resultJSON, computeSystem.callbackNumber, hcsNotificationSystemResumeCompleted, &timeout.SystemResume)
	if err != nil {
		return makeSystemError(computeSystem, operation, err, events)
	}

	return nil
}

// Save the compute system
func (computeSystem *System) Save(ctx context.Context, options interface{}) (err error) {
	operation := "hcs::System::Save"

	// hcsSaveComputeSystemContext is an async peration. Start the outer span
	// here to measure the full save time.
	ctx, span := trace.StartSpan(ctx, operation)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("cid", computeSystem.id))

	saveOptions, err := json.Marshal(options)
	if err != nil {
		return err
	}

	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	if computeSystem.handle == 0 {
		return makeSystemError(computeSystem, operation, ErrAlreadyClosed, nil)
	}

	result, err := vmcompute.HcsSaveComputeSystem(ctx, computeSystem.handle, string(saveOptions))
	events, err := processAsyncHcsResult(ctx, err, result, computeSystem.callbackNumber, hcsNotificationSystemSaveCompleted, &timeout.SystemSave)
	if err != nil {
		return makeSystemError(computeSystem, operation, err, events)
	}

	return nil
}

func (computeSystem *System) createProcess(ctx context.Context, operation string, c interface{}) (*Process, *vmcompute.HcsProcessInformation, error) {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	if computeSystem.handle == 0 {
		return nil, nil, makeSystemError(computeSystem, operation, ErrAlreadyClosed, nil)
	}

	configurationb, err := json.Marshal(c)
	if err != nil {
		return nil, nil, makeSystemError(computeSystem, operation, err, nil)
	}

	configuration := string(configurationb)
	processInfo, processHandle, resultJSON, err := vmcompute.HcsCreateProcess(ctx, computeSystem.handle, configuration)
	events := processHcsResult(ctx, resultJSON)
	if err != nil {
		return nil, nil, makeSystemError(computeSystem, operation, err, events)
	}

	log.G(ctx).WithField("pid", processInfo.ProcessId).Debug("created process pid")
	return newProcess(processHandle, int(processInfo.ProcessId), computeSystem), &processInfo, nil
}

// CreateProcess launches a new process within the computeSystem.
func (computeSystem *System) CreateProcess(ctx context.Context, c interface{}) (cow.Process, error) {
	operation := "hcs::System::CreateProcess"
	process, processInfo, err := computeSystem.createProcess(ctx, operation, c)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			process.Close()
		}
	}()

	pipes, err := makeOpenFiles([]syscall.Handle{processInfo.StdInput, processInfo.StdOutput, processInfo.StdError})
	if err != nil {
		return nil, makeSystemError(computeSystem, operation, err, nil)
	}
	process.stdin = pipes[0]
	process.stdout = pipes[1]
	process.stderr = pipes[2]
	process.hasCachedStdio = true

	if err = process.registerCallback(ctx); err != nil {
		return nil, makeSystemError(computeSystem, operation, err, nil)
	}
	go process.waitBackground()

	return process, nil
}

// OpenProcess gets an interface to an existing process within the computeSystem.
func (computeSystem *System) OpenProcess(ctx context.Context, pid int) (*Process, error) {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	operation := "hcs::System::OpenProcess"

	if computeSystem.handle == 0 {
		return nil, makeSystemError(computeSystem, operation, ErrAlreadyClosed, nil)
	}

	processHandle, resultJSON, err := vmcompute.HcsOpenProcess(ctx, computeSystem.handle, uint32(pid))
	events := processHcsResult(ctx, resultJSON)
	if err != nil {
		return nil, makeSystemError(computeSystem, operation, err, events)
	}

	process := newProcess(processHandle, pid, computeSystem)
	if err = process.registerCallback(ctx); err != nil {
		return nil, makeSystemError(computeSystem, operation, err, nil)
	}
	go process.waitBackground()

	return process, nil
}

// Close cleans up any state associated with the compute system but does not terminate or wait for it.
func (computeSystem *System) Close() (err error) {
	operation := "hcs::System::Close"
	ctx, span := trace.StartSpan(context.Background(), operation)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("cid", computeSystem.id))

	computeSystem.handleLock.Lock()
	defer computeSystem.handleLock.Unlock()

	// Don't double free this
	if computeSystem.handle == 0 {
		return nil
	}

	if err = computeSystem.unregisterCallback(ctx); err != nil {
		return makeSystemError(computeSystem, operation, err, nil)
	}

	err = vmcompute.HcsCloseComputeSystem(ctx, computeSystem.handle)
	if err != nil {
		return makeSystemError(computeSystem, operation, err, nil)
	}

	computeSystem.handle = 0
	computeSystem.closedWaitOnce.Do(func() {
		computeSystem.waitError = ErrAlreadyClosed
		close(computeSystem.waitBlock)
	})

	return nil
}

func (computeSystem *System) registerCallback(ctx context.Context) error {
	callbackContext := &notificationWatcherContext{
		channels: newSystemChannels(),
		systemID: computeSystem.id,
	}

	callbackMapLock.Lock()
	callbackNumber := nextCallback
	nextCallback++
	callbackMap[callbackNumber] = callbackContext
	callbackMapLock.Unlock()

	callbackHandle, err := vmcompute.HcsRegisterComputeSystemCallback(ctx, computeSystem.handle, notificationWatcherCallback, callbackNumber)
	if err != nil {
		return err
	}
	callbackContext.handle = callbackHandle
	computeSystem.callbackNumber = callbackNumber

	return nil
}

func (computeSystem *System) unregisterCallback(ctx context.Context) error {
	callbackNumber := computeSystem.callbackNumber

	callbackMapLock.RLock()
	callbackContext := callbackMap[callbackNumber]
	callbackMapLock.RUnlock()

	if callbackContext == nil {
		return nil
	}

	handle := callbackContext.handle

	if handle == 0 {
		return nil
	}

	// hcsUnregisterComputeSystemCallback has its own syncronization
	// to wait for all callbacks to complete. We must NOT hold the callbackMapLock.
	err := vmcompute.HcsUnregisterComputeSystemCallback(ctx, handle)
	if err != nil {
		return err
	}

	closeChannels(callbackContext.channels)

	callbackMapLock.Lock()
	delete(callbackMap, callbackNumber)
	callbackMapLock.Unlock()

	handle = 0 //nolint:ineffassign

	return nil
}

// Modify the System by sending a request to HCS
func (computeSystem *System) Modify(ctx context.Context, config interface{}) error {
	computeSystem.handleLock.RLock()
	defer computeSystem.handleLock.RUnlock()

	operation := "hcs::System::Modify"

	if computeSystem.handle == 0 {
		return makeSystemError(computeSystem, operation, ErrAlreadyClosed, nil)
	}

	requestBytes, err := json.Marshal(config)
	if err != nil {
		return err
	}

	requestJSON := string(requestBytes)
	resultJSON, err := vmcompute.HcsModifyComputeSystem(ctx, computeSystem.handle, requestJSON)
	events := processHcsResult(ctx, resultJSON)
	if err != nil {
		return makeSystemError(computeSystem, operation, err, events)
	}

	return nil
}
