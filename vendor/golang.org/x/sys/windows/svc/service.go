// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows
// +build windows

// Package svc provides everything required to build Windows service.
//
package svc

import (
	"errors"
	"sync"
	"unsafe"

	"golang.org/x/sys/internal/unsafeheader"
	"golang.org/x/sys/windows"
)

// State describes service execution state (Stopped, Running and so on).
type State uint32

const (
	Stopped         = State(windows.SERVICE_STOPPED)
	StartPending    = State(windows.SERVICE_START_PENDING)
	StopPending     = State(windows.SERVICE_STOP_PENDING)
	Running         = State(windows.SERVICE_RUNNING)
	ContinuePending = State(windows.SERVICE_CONTINUE_PENDING)
	PausePending    = State(windows.SERVICE_PAUSE_PENDING)
	Paused          = State(windows.SERVICE_PAUSED)
)

// Cmd represents service state change request. It is sent to a service
// by the service manager, and should be actioned upon by the service.
type Cmd uint32

const (
	Stop                  = Cmd(windows.SERVICE_CONTROL_STOP)
	Pause                 = Cmd(windows.SERVICE_CONTROL_PAUSE)
	Continue              = Cmd(windows.SERVICE_CONTROL_CONTINUE)
	Interrogate           = Cmd(windows.SERVICE_CONTROL_INTERROGATE)
	Shutdown              = Cmd(windows.SERVICE_CONTROL_SHUTDOWN)
	ParamChange           = Cmd(windows.SERVICE_CONTROL_PARAMCHANGE)
	NetBindAdd            = Cmd(windows.SERVICE_CONTROL_NETBINDADD)
	NetBindRemove         = Cmd(windows.SERVICE_CONTROL_NETBINDREMOVE)
	NetBindEnable         = Cmd(windows.SERVICE_CONTROL_NETBINDENABLE)
	NetBindDisable        = Cmd(windows.SERVICE_CONTROL_NETBINDDISABLE)
	DeviceEvent           = Cmd(windows.SERVICE_CONTROL_DEVICEEVENT)
	HardwareProfileChange = Cmd(windows.SERVICE_CONTROL_HARDWAREPROFILECHANGE)
	PowerEvent            = Cmd(windows.SERVICE_CONTROL_POWEREVENT)
	SessionChange         = Cmd(windows.SERVICE_CONTROL_SESSIONCHANGE)
	PreShutdown           = Cmd(windows.SERVICE_CONTROL_PRESHUTDOWN)
)

// Accepted is used to describe commands accepted by the service.
// Note that Interrogate is always accepted.
type Accepted uint32

const (
	AcceptStop                  = Accepted(windows.SERVICE_ACCEPT_STOP)
	AcceptShutdown              = Accepted(windows.SERVICE_ACCEPT_SHUTDOWN)
	AcceptPauseAndContinue      = Accepted(windows.SERVICE_ACCEPT_PAUSE_CONTINUE)
	AcceptParamChange           = Accepted(windows.SERVICE_ACCEPT_PARAMCHANGE)
	AcceptNetBindChange         = Accepted(windows.SERVICE_ACCEPT_NETBINDCHANGE)
	AcceptHardwareProfileChange = Accepted(windows.SERVICE_ACCEPT_HARDWAREPROFILECHANGE)
	AcceptPowerEvent            = Accepted(windows.SERVICE_ACCEPT_POWEREVENT)
	AcceptSessionChange         = Accepted(windows.SERVICE_ACCEPT_SESSIONCHANGE)
	AcceptPreShutdown           = Accepted(windows.SERVICE_ACCEPT_PRESHUTDOWN)
)

// Status combines State and Accepted commands to fully describe running service.
type Status struct {
	State                   State
	Accepts                 Accepted
	CheckPoint              uint32 // used to report progress during a lengthy operation
	WaitHint                uint32 // estimated time required for a pending operation, in milliseconds
	ProcessId               uint32 // if the service is running, the process identifier of it, and otherwise zero
	Win32ExitCode           uint32 // set if the service has exited with a win32 exit code
	ServiceSpecificExitCode uint32 // set if the service has exited with a service-specific exit code
}

// StartReason is the reason that the service was started.
type StartReason uint32

const (
	StartReasonDemand           = StartReason(windows.SERVICE_START_REASON_DEMAND)
	StartReasonAuto             = StartReason(windows.SERVICE_START_REASON_AUTO)
	StartReasonTrigger          = StartReason(windows.SERVICE_START_REASON_TRIGGER)
	StartReasonRestartOnFailure = StartReason(windows.SERVICE_START_REASON_RESTART_ON_FAILURE)
	StartReasonDelayedAuto      = StartReason(windows.SERVICE_START_REASON_DELAYEDAUTO)
)

// ChangeRequest is sent to the service Handler to request service status change.
type ChangeRequest struct {
	Cmd           Cmd
	EventType     uint32
	EventData     uintptr
	CurrentStatus Status
	Context       uintptr
}

// Handler is the interface that must be implemented to build Windows service.
type Handler interface {
	// Execute will be called by the package code at the start of
	// the service, and the service will exit once Execute completes.
	// Inside Execute you must read service change requests from r and
	// act accordingly. You must keep service control manager up to date
	// about state of your service by writing into s as required.
	// args contains service name followed by argument strings passed
	// to the service.
	// You can provide service exit code in exitCode return parameter,
	// with 0 being "no error". You can also indicate if exit code,
	// if any, is service specific or not by using svcSpecificEC
	// parameter.
	Execute(args []string, r <-chan ChangeRequest, s chan<- Status) (svcSpecificEC bool, exitCode uint32)
}

type ctlEvent struct {
	cmd       Cmd
	eventType uint32
	eventData uintptr
	context   uintptr
	errno     uint32
}

// service provides access to windows service api.
type service struct {
	name    string
	h       windows.Handle
	c       chan ctlEvent
	handler Handler
}

type exitCode struct {
	isSvcSpecific bool
	errno         uint32
}

func (s *service) updateStatus(status *Status, ec *exitCode) error {
	if s.h == 0 {
		return errors.New("updateStatus with no service status handle")
	}
	var t windows.SERVICE_STATUS
	t.ServiceType = windows.SERVICE_WIN32_OWN_PROCESS
	t.CurrentState = uint32(status.State)
	if status.Accepts&AcceptStop != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_STOP
	}
	if status.Accepts&AcceptShutdown != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_SHUTDOWN
	}
	if status.Accepts&AcceptPauseAndContinue != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_PAUSE_CONTINUE
	}
	if status.Accepts&AcceptParamChange != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_PARAMCHANGE
	}
	if status.Accepts&AcceptNetBindChange != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_NETBINDCHANGE
	}
	if status.Accepts&AcceptHardwareProfileChange != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_HARDWAREPROFILECHANGE
	}
	if status.Accepts&AcceptPowerEvent != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_POWEREVENT
	}
	if status.Accepts&AcceptSessionChange != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_SESSIONCHANGE
	}
	if status.Accepts&AcceptPreShutdown != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_PRESHUTDOWN
	}
	if ec.errno == 0 {
		t.Win32ExitCode = windows.NO_ERROR
		t.ServiceSpecificExitCode = windows.NO_ERROR
	} else if ec.isSvcSpecific {
		t.Win32ExitCode = uint32(windows.ERROR_SERVICE_SPECIFIC_ERROR)
		t.ServiceSpecificExitCode = ec.errno
	} else {
		t.Win32ExitCode = ec.errno
		t.ServiceSpecificExitCode = windows.NO_ERROR
	}
	t.CheckPoint = status.CheckPoint
	t.WaitHint = status.WaitHint
	return windows.SetServiceStatus(s.h, &t)
}

var (
	initCallbacks       sync.Once
	ctlHandlerCallback  uintptr
	serviceMainCallback uintptr
)

func ctlHandler(ctl, evtype, evdata, context uintptr) uintptr {
	s := (*service)(unsafe.Pointer(context))
	e := ctlEvent{cmd: Cmd(ctl), eventType: uint32(evtype), eventData: evdata, context: 123456} // Set context to 123456 to test issue #25660.
	s.c <- e
	return 0
}

var theService service // This is, unfortunately, a global, which means only one service per process.

// serviceMain is the entry point called by the service manager, registered earlier by
// the call to StartServiceCtrlDispatcher.
func serviceMain(argc uint32, argv **uint16) uintptr {
	handle, err := windows.RegisterServiceCtrlHandlerEx(windows.StringToUTF16Ptr(theService.name), ctlHandlerCallback, uintptr(unsafe.Pointer(&theService)))
	if sysErr, ok := err.(windows.Errno); ok {
		return uintptr(sysErr)
	} else if err != nil {
		return uintptr(windows.ERROR_UNKNOWN_EXCEPTION)
	}
	theService.h = handle
	defer func() {
		theService.h = 0
	}()
	var args16 []*uint16
	hdr := (*unsafeheader.Slice)(unsafe.Pointer(&args16))
	hdr.Data = unsafe.Pointer(argv)
	hdr.Len = int(argc)
	hdr.Cap = int(argc)

	args := make([]string, len(args16))
	for i, a := range args16 {
		args[i] = windows.UTF16PtrToString(a)
	}

	cmdsToHandler := make(chan ChangeRequest)
	changesFromHandler := make(chan Status)
	exitFromHandler := make(chan exitCode)

	go func() {
		ss, errno := theService.handler.Execute(args, cmdsToHandler, changesFromHandler)
		exitFromHandler <- exitCode{ss, errno}
	}()

	ec := exitCode{isSvcSpecific: true, errno: 0}
	outcr := ChangeRequest{
		CurrentStatus: Status{State: Stopped},
	}
	var outch chan ChangeRequest
	inch := theService.c
loop:
	for {
		select {
		case r := <-inch:
			if r.errno != 0 {
				ec.errno = r.errno
				break loop
			}
			inch = nil
			outch = cmdsToHandler
			outcr.Cmd = r.cmd
			outcr.EventType = r.eventType
			outcr.EventData = r.eventData
			outcr.Context = r.context
		case outch <- outcr:
			inch = theService.c
			outch = nil
		case c := <-changesFromHandler:
			err := theService.updateStatus(&c, &ec)
			if err != nil {
				ec.errno = uint32(windows.ERROR_EXCEPTION_IN_SERVICE)
				if err2, ok := err.(windows.Errno); ok {
					ec.errno = uint32(err2)
				}
				break loop
			}
			outcr.CurrentStatus = c
		case ec = <-exitFromHandler:
			break loop
		}
	}

	theService.updateStatus(&Status{State: Stopped}, &ec)

	return windows.NO_ERROR
}

// Run executes service name by calling appropriate handler function.
func Run(name string, handler Handler) error {
	initCallbacks.Do(func() {
		ctlHandlerCallback = windows.NewCallback(ctlHandler)
		serviceMainCallback = windows.NewCallback(serviceMain)
	})
	theService.name = name
	theService.handler = handler
	theService.c = make(chan ctlEvent)
	t := []windows.SERVICE_TABLE_ENTRY{
		{ServiceName: windows.StringToUTF16Ptr(theService.name), ServiceProc: serviceMainCallback},
		{ServiceName: nil, ServiceProc: 0},
	}
	return windows.StartServiceCtrlDispatcher(&t[0])
}

// StatusHandle returns service status handle. It is safe to call this function
// from inside the Handler.Execute because then it is guaranteed to be set.
func StatusHandle() windows.Handle {
	return theService.h
}

// DynamicStartReason returns the reason why the service was started. It is safe
// to call this function from inside the Handler.Execute because then it is
// guaranteed to be set.
func DynamicStartReason() (StartReason, error) {
	var allocReason *uint32
	err := windows.QueryServiceDynamicInformation(theService.h, windows.SERVICE_DYNAMIC_INFORMATION_LEVEL_START_REASON, unsafe.Pointer(&allocReason))
	if err != nil {
		return 0, err
	}
	reason := StartReason(*allocReason)
	windows.LocalFree(windows.Handle(unsafe.Pointer(allocReason)))
	return reason, nil
}
