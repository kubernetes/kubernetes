//go:build windows

package jobobject

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/Microsoft/hcsshim/internal/queue"
	"github.com/Microsoft/hcsshim/internal/winapi"
	"golang.org/x/sys/windows"
)

// JobObject is a high level wrapper around a Windows job object. Holds a handle to
// the job, a queue to receive iocp notifications about the lifecycle
// of the job and a mutex for synchronized handle access.
type JobObject struct {
	handle windows.Handle
	// All accesses to this MUST be done atomically except in `Open` as the object
	// is being created in the function. 1 signifies that this job is currently a silo.
	silo       uint32
	mq         *queue.MessageQueue
	handleLock sync.RWMutex
}

// JobLimits represents the resource constraints that can be applied to a job object.
type JobLimits struct {
	CPULimit           uint32
	CPUWeight          uint32
	MemoryLimitInBytes uint64
	MaxIOPS            int64
	MaxBandwidth       int64
}

type CPURateControlType uint32

const (
	WeightBased CPURateControlType = iota
	RateBased
)

// Processor resource controls
const (
	cpuLimitMin  = 1
	cpuLimitMax  = 10000
	cpuWeightMin = 1
	cpuWeightMax = 9
)

var (
	ErrAlreadyClosed = errors.New("the handle has already been closed")
	ErrNotRegistered = errors.New("job is not registered to receive notifications")
	ErrNotSilo       = errors.New("job is not a silo")
)

// Options represents the set of configurable options when making or opening a job object.
type Options struct {
	// `Name` specifies the name of the job object if a named job object is desired.
	Name string
	// `Notifications` specifies if the job will be registered to receive notifications.
	// Defaults to false.
	Notifications bool
	// `UseNTVariant` specifies if we should use the `Nt` variant of Open/CreateJobObject.
	// Defaults to false.
	UseNTVariant bool
	// `Silo` specifies to promote the job to a silo. This additionally sets the flag
	// JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE as it is required for the upgrade to complete.
	Silo bool
	// `IOTracking` enables tracking I/O statistics on the job object. More specifically this
	// calls SetInformationJobObject with the JobObjectIoAttribution class.
	EnableIOTracking bool
}

// Create creates a job object.
//
// If options.Name is an empty string, the job will not be assigned a name.
//
// If options.Notifications are not enabled `PollNotifications` will return immediately with error `errNotRegistered`.
//
// If `options` is nil, use default option values.
//
// Returns a JobObject structure and an error if there is one.
func Create(ctx context.Context, options *Options) (_ *JobObject, err error) {
	if options == nil {
		options = &Options{}
	}

	var jobName *winapi.UnicodeString
	if options.Name != "" {
		jobName, err = winapi.NewUnicodeString(options.Name)
		if err != nil {
			return nil, err
		}
	}

	var jobHandle windows.Handle
	if options.UseNTVariant {
		oa := winapi.ObjectAttributes{
			Length:     unsafe.Sizeof(winapi.ObjectAttributes{}),
			ObjectName: jobName,
			Attributes: 0,
		}
		status := winapi.NtCreateJobObject(&jobHandle, winapi.JOB_OBJECT_ALL_ACCESS, &oa)
		if status != 0 {
			return nil, winapi.RtlNtStatusToDosError(status)
		}
	} else {
		var jobNameBuf *uint16
		if jobName != nil && jobName.Buffer != nil {
			jobNameBuf = jobName.Buffer
		}
		jobHandle, err = windows.CreateJobObject(nil, jobNameBuf)
		if err != nil {
			return nil, err
		}
	}

	defer func() {
		if err != nil {
			windows.Close(jobHandle)
		}
	}()

	job := &JobObject{
		handle: jobHandle,
	}

	// If the IOCP we'll be using to receive messages for all jobs hasn't been
	// created, create it and start polling.
	if options.Notifications {
		mq, err := setupNotifications(ctx, job)
		if err != nil {
			return nil, err
		}
		job.mq = mq
	}

	if options.EnableIOTracking {
		if err := enableIOTracking(jobHandle); err != nil {
			return nil, err
		}
	}

	if options.Silo {
		// This is a required setting for upgrading to a silo.
		if err := job.SetTerminateOnLastHandleClose(); err != nil {
			return nil, err
		}
		if err := job.PromoteToSilo(); err != nil {
			return nil, err
		}
	}

	return job, nil
}

// Open opens an existing job object with name provided in `options`. If no name is provided
// return an error since we need to know what job object to open.
//
// If options.Notifications is false `PollNotifications` will return immediately with error `errNotRegistered`.
//
// Returns a JobObject structure and an error if there is one.
func Open(ctx context.Context, options *Options) (_ *JobObject, err error) {
	if options == nil || options.Name == "" {
		return nil, errors.New("no job object name specified to open")
	}

	unicodeJobName, err := winapi.NewUnicodeString(options.Name)
	if err != nil {
		return nil, err
	}

	var jobHandle windows.Handle
	if options.UseNTVariant {
		oa := winapi.ObjectAttributes{
			Length:     unsafe.Sizeof(winapi.ObjectAttributes{}),
			ObjectName: unicodeJobName,
			Attributes: 0,
		}
		status := winapi.NtOpenJobObject(&jobHandle, winapi.JOB_OBJECT_ALL_ACCESS, &oa)
		if status != 0 {
			return nil, winapi.RtlNtStatusToDosError(status)
		}
	} else {
		jobHandle, err = winapi.OpenJobObject(winapi.JOB_OBJECT_ALL_ACCESS, 0, unicodeJobName.Buffer)
		if err != nil {
			return nil, err
		}
	}

	defer func() {
		if err != nil {
			windows.Close(jobHandle)
		}
	}()

	job := &JobObject{
		handle: jobHandle,
	}

	if isJobSilo(jobHandle) {
		job.silo = 1
	}

	// If the IOCP we'll be using to receive messages for all jobs hasn't been
	// created, create it and start polling.
	if options.Notifications {
		mq, err := setupNotifications(ctx, job)
		if err != nil {
			return nil, err
		}
		job.mq = mq
	}

	return job, nil
}

// helper function to setup notifications for creating/opening a job object
func setupNotifications(ctx context.Context, job *JobObject) (*queue.MessageQueue, error) {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return nil, ErrAlreadyClosed
	}

	ioInitOnce.Do(func() {
		h, err := windows.CreateIoCompletionPort(windows.InvalidHandle, 0, 0, 0xffffffff)
		if err != nil {
			initIOErr = err
			return
		}
		ioCompletionPort = h
		go pollIOCP(ctx, h)
	})

	if initIOErr != nil {
		return nil, initIOErr
	}

	mq := queue.NewMessageQueue()
	jobMap.Store(uintptr(job.handle), mq)
	if err := attachIOCP(job.handle, ioCompletionPort); err != nil {
		jobMap.Delete(uintptr(job.handle))
		return nil, fmt.Errorf("failed to attach job to IO completion port: %w", err)
	}
	return mq, nil
}

// PollNotification will poll for a job object notification. This call should only be called once
// per job (ideally in a goroutine loop) and will block if there is not a notification ready.
// This call will return immediately with error `ErrNotRegistered` if the job was not registered
// to receive notifications during `Create`. Internally, messages will be queued and there
// is no worry of messages being dropped.
func (job *JobObject) PollNotification() (interface{}, error) {
	if job.mq == nil {
		return nil, ErrNotRegistered
	}
	return job.mq.Dequeue()
}

// UpdateProcThreadAttribute updates the passed in ProcThreadAttributeList to contain what is necessary to
// launch a process in a job at creation time. This can be used to avoid having to call Assign() after a process
// has already started running.
func (job *JobObject) UpdateProcThreadAttribute(attrList *windows.ProcThreadAttributeListContainer) error {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return ErrAlreadyClosed
	}

	if err := attrList.Update(
		winapi.PROC_THREAD_ATTRIBUTE_JOB_LIST,
		unsafe.Pointer(&job.handle),
		unsafe.Sizeof(job.handle),
	); err != nil {
		return fmt.Errorf("failed to update proc thread attributes for job object: %w", err)
	}

	return nil
}

// Close closes the job object handle.
func (job *JobObject) Close() error {
	job.handleLock.Lock()
	defer job.handleLock.Unlock()

	if job.handle == 0 {
		return ErrAlreadyClosed
	}

	if err := windows.Close(job.handle); err != nil {
		return err
	}

	if job.mq != nil {
		job.mq.Close()
	}
	// Handles now invalid so if the map entry to receive notifications for this job still
	// exists remove it so we can stop receiving notifications.
	if _, ok := jobMap.Load(uintptr(job.handle)); ok {
		jobMap.Delete(uintptr(job.handle))
	}

	job.handle = 0
	return nil
}

// Assign assigns a process to the job object.
func (job *JobObject) Assign(pid uint32) error {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return ErrAlreadyClosed
	}

	if pid == 0 {
		return errors.New("invalid pid: 0")
	}
	hProc, err := windows.OpenProcess(winapi.PROCESS_ALL_ACCESS, true, pid)
	if err != nil {
		return err
	}
	defer windows.Close(hProc)
	return windows.AssignProcessToJobObject(job.handle, hProc)
}

// Terminate terminates the job, essentially calls TerminateProcess on every process in the
// job.
func (job *JobObject) Terminate(exitCode uint32) error {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()
	if job.handle == 0 {
		return ErrAlreadyClosed
	}
	return windows.TerminateJobObject(job.handle, exitCode)
}

// Pids returns all of the process IDs in the job object.
func (job *JobObject) Pids() ([]uint32, error) {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return nil, ErrAlreadyClosed
	}

	info := winapi.JOBOBJECT_BASIC_PROCESS_ID_LIST{}
	err := winapi.QueryInformationJobObject(
		job.handle,
		winapi.JobObjectBasicProcessIdList,
		unsafe.Pointer(&info),
		uint32(unsafe.Sizeof(info)),
		nil,
	)

	// This is either the case where there is only one process or no processes in
	// the job. Any other case will result in ERROR_MORE_DATA. Check if info.NumberOfProcessIdsInList
	// is 1 and just return this, otherwise return an empty slice.
	if err == nil {
		if info.NumberOfProcessIdsInList == 1 {
			return []uint32{uint32(info.ProcessIdList[0])}, nil
		}
		// Return empty slice instead of nil to play well with the caller of this.
		// Do not return an error if no processes are running inside the job
		return []uint32{}, nil
	}

	if err != winapi.ERROR_MORE_DATA { //nolint:errorlint
		return nil, fmt.Errorf("failed initial query for PIDs in job object: %w", err)
	}

	jobBasicProcessIDListSize := unsafe.Sizeof(info) + (unsafe.Sizeof(info.ProcessIdList[0]) * uintptr(info.NumberOfAssignedProcesses-1))
	buf := make([]byte, jobBasicProcessIDListSize)
	if err = winapi.QueryInformationJobObject(
		job.handle,
		winapi.JobObjectBasicProcessIdList,
		unsafe.Pointer(&buf[0]),
		uint32(len(buf)),
		nil,
	); err != nil {
		return nil, fmt.Errorf("failed to query for PIDs in job object: %w", err)
	}

	bufInfo := (*winapi.JOBOBJECT_BASIC_PROCESS_ID_LIST)(unsafe.Pointer(&buf[0]))
	pids := make([]uint32, bufInfo.NumberOfProcessIdsInList)
	for i, bufPid := range bufInfo.AllPids() {
		pids[i] = uint32(bufPid)
	}
	return pids, nil
}

// QueryMemoryStats gets the memory stats for the job object.
func (job *JobObject) QueryMemoryStats() (*winapi.JOBOBJECT_MEMORY_USAGE_INFORMATION, error) {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return nil, ErrAlreadyClosed
	}

	info := winapi.JOBOBJECT_MEMORY_USAGE_INFORMATION{}
	if err := winapi.QueryInformationJobObject(
		job.handle,
		winapi.JobObjectMemoryUsageInformation,
		unsafe.Pointer(&info),
		uint32(unsafe.Sizeof(info)),
		nil,
	); err != nil {
		return nil, fmt.Errorf("failed to query for job object memory stats: %w", err)
	}
	return &info, nil
}

// QueryProcessorStats gets the processor stats for the job object.
func (job *JobObject) QueryProcessorStats() (*winapi.JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, error) {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return nil, ErrAlreadyClosed
	}

	info := winapi.JOBOBJECT_BASIC_ACCOUNTING_INFORMATION{}
	if err := winapi.QueryInformationJobObject(
		job.handle,
		winapi.JobObjectBasicAccountingInformation,
		unsafe.Pointer(&info),
		uint32(unsafe.Sizeof(info)),
		nil,
	); err != nil {
		return nil, fmt.Errorf("failed to query for job object process stats: %w", err)
	}
	return &info, nil
}

// QueryStorageStats gets the storage (I/O) stats for the job object. This call will error
// if either `EnableIOTracking` wasn't set to true on creation of the job, or SetIOTracking()
// hasn't been called since creation of the job.
func (job *JobObject) QueryStorageStats() (*winapi.JOBOBJECT_IO_ATTRIBUTION_INFORMATION, error) {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return nil, ErrAlreadyClosed
	}

	info := winapi.JOBOBJECT_IO_ATTRIBUTION_INFORMATION{
		ControlFlags: winapi.JOBOBJECT_IO_ATTRIBUTION_CONTROL_ENABLE,
	}
	if err := winapi.QueryInformationJobObject(
		job.handle,
		winapi.JobObjectIoAttribution,
		unsafe.Pointer(&info),
		uint32(unsafe.Sizeof(info)),
		nil,
	); err != nil {
		return nil, fmt.Errorf("failed to query for job object storage stats: %w", err)
	}
	return &info, nil
}

// ApplyFileBinding makes a file binding using the Bind Filter from target to root. If the job has
// not been upgraded to a silo this call will fail. The binding is only applied and visible for processes
// running in the job, any processes on the host or in another job will not be able to see the binding.
func (job *JobObject) ApplyFileBinding(root, target string, readOnly bool) error {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return ErrAlreadyClosed
	}

	if !job.isSilo() {
		return ErrNotSilo
	}

	// The parent directory needs to exist for the bind to work. MkdirAll stats and
	// returns nil if the directory exists internally so we should be fine to mkdirall
	// every time.
	if err := os.MkdirAll(filepath.Dir(root), 0); err != nil {
		return err
	}

	rootPtr, err := windows.UTF16PtrFromString(root)
	if err != nil {
		return err
	}

	targetPtr, err := windows.UTF16PtrFromString(target)
	if err != nil {
		return err
	}

	flags := winapi.BINDFLT_FLAG_USE_CURRENT_SILO_MAPPING
	if readOnly {
		flags |= winapi.BINDFLT_FLAG_READ_ONLY_MAPPING
	}

	if err := winapi.BfSetupFilter(
		job.handle,
		flags,
		rootPtr,
		targetPtr,
		nil,
		0,
	); err != nil {
		return fmt.Errorf("failed to bind target %q to root %q for job object: %w", target, root, err)
	}
	return nil
}

// isJobSilo is a helper to determine if a job object that was opened is a silo. This should ONLY be called
// from `Open` and any callers in this package afterwards should use `job.isSilo()`
func isJobSilo(h windows.Handle) bool {
	// None of the information from the structure that this info class expects will be used, this is just used as
	// the call will fail if the job hasn't been upgraded to a silo so we can use this to tell when we open a job
	// if it's a silo or not. Because none of the info matters simply define a dummy struct with the size that the call
	// expects which is 16 bytes.
	type isSiloObj struct {
		_ [16]byte
	}
	var siloInfo isSiloObj
	err := winapi.QueryInformationJobObject(
		h,
		winapi.JobObjectSiloBasicInformation,
		unsafe.Pointer(&siloInfo),
		uint32(unsafe.Sizeof(siloInfo)),
		nil,
	)
	return err == nil
}

// PromoteToSilo promotes a job object to a silo. There must be no running processess
// in the job for this to succeed. If the job is already a silo this is a no-op.
func (job *JobObject) PromoteToSilo() error {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return ErrAlreadyClosed
	}

	if job.isSilo() {
		return nil
	}

	pids, err := job.Pids()
	if err != nil {
		return err
	}

	if len(pids) != 0 {
		return fmt.Errorf("job cannot have running processes to be promoted to a silo, found %d running processes", len(pids))
	}

	_, err = windows.SetInformationJobObject(
		job.handle,
		winapi.JobObjectCreateSilo,
		0,
		0,
	)
	if err != nil {
		return fmt.Errorf("failed to promote job to silo: %w", err)
	}

	atomic.StoreUint32(&job.silo, 1)
	return nil
}

// isSilo returns if the job object is a silo.
func (job *JobObject) isSilo() bool {
	return atomic.LoadUint32(&job.silo) == 1
}

// QueryPrivateWorkingSet returns the private working set size for the job. This is calculated by adding up the
// private working set for every process running in the job.
func (job *JobObject) QueryPrivateWorkingSet() (uint64, error) {
	pids, err := job.Pids()
	if err != nil {
		return 0, err
	}

	openAndQueryWorkingSet := func(pid uint32) (uint64, error) {
		h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, pid)
		if err != nil {
			// Continue to the next if OpenProcess doesn't return a valid handle (fails). Handles a
			// case where one of the pids in the job exited before we open.
			return 0, nil
		}
		defer func() {
			_ = windows.Close(h)
		}()
		// Check if the process is actually running in the job still. There's a small chance
		// that the process could have exited and had its pid re-used between grabbing the pids
		// in the job and opening the handle to it above.
		var inJob int32
		if err := winapi.IsProcessInJob(h, job.handle, &inJob); err != nil {
			// This shouldn't fail unless we have incorrect access rights which we control
			// here so probably best to error out if this failed.
			return 0, err
		}
		// Don't report stats for this process as it's not running in the job. This shouldn't be
		// an error condition though.
		if inJob == 0 {
			return 0, nil
		}

		var vmCounters winapi.VM_COUNTERS_EX2
		status := winapi.NtQueryInformationProcess(
			h,
			winapi.ProcessVmCounters,
			unsafe.Pointer(&vmCounters),
			uint32(unsafe.Sizeof(vmCounters)),
			nil,
		)
		if !winapi.NTSuccess(status) {
			return 0, fmt.Errorf("failed to query information for process: %w", winapi.RtlNtStatusToDosError(status))
		}
		return uint64(vmCounters.PrivateWorkingSetSize), nil
	}

	var jobWorkingSetSize uint64
	for _, pid := range pids {
		workingSet, err := openAndQueryWorkingSet(pid)
		if err != nil {
			return 0, err
		}
		jobWorkingSetSize += workingSet
	}

	return jobWorkingSetSize, nil
}

// SetIOTracking enables IO tracking for processes in the job object.
// This enables use of the QueryStorageStats method.
func (job *JobObject) SetIOTracking() error {
	job.handleLock.RLock()
	defer job.handleLock.RUnlock()

	if job.handle == 0 {
		return ErrAlreadyClosed
	}

	return enableIOTracking(job.handle)
}

func enableIOTracking(job windows.Handle) error {
	info := winapi.JOBOBJECT_IO_ATTRIBUTION_INFORMATION{
		ControlFlags: winapi.JOBOBJECT_IO_ATTRIBUTION_CONTROL_ENABLE,
	}
	if _, err := windows.SetInformationJobObject(
		job,
		winapi.JobObjectIoAttribution,
		uintptr(unsafe.Pointer(&info)),
		uint32(unsafe.Sizeof(info)),
	); err != nil {
		return fmt.Errorf("failed to enable IO tracking on job object: %w", err)
	}
	return nil
}
