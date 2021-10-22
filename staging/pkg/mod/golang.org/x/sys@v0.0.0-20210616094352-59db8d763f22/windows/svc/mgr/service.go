// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package mgr

import (
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/svc"
)

// TODO(brainman): Use EnumDependentServices to enumerate dependent services.

// Service is used to access Windows service.
type Service struct {
	Name   string
	Handle windows.Handle
}

// Delete marks service s for deletion from the service control manager database.
func (s *Service) Delete() error {
	return windows.DeleteService(s.Handle)
}

// Close relinquish access to the service s.
func (s *Service) Close() error {
	return windows.CloseServiceHandle(s.Handle)
}

// Start starts service s.
// args will be passed to svc.Handler.Execute.
func (s *Service) Start(args ...string) error {
	var p **uint16
	if len(args) > 0 {
		vs := make([]*uint16, len(args))
		for i := range vs {
			vs[i] = syscall.StringToUTF16Ptr(args[i])
		}
		p = &vs[0]
	}
	return windows.StartService(s.Handle, uint32(len(args)), p)
}

// Control sends state change request c to the service s.
func (s *Service) Control(c svc.Cmd) (svc.Status, error) {
	var t windows.SERVICE_STATUS
	err := windows.ControlService(s.Handle, uint32(c), &t)
	if err != nil {
		return svc.Status{}, err
	}
	return svc.Status{
		State:   svc.State(t.CurrentState),
		Accepts: svc.Accepted(t.ControlsAccepted),
	}, nil
}

// Query returns current status of service s.
func (s *Service) Query() (svc.Status, error) {
	var t windows.SERVICE_STATUS_PROCESS
	var needed uint32
	err := windows.QueryServiceStatusEx(s.Handle, windows.SC_STATUS_PROCESS_INFO, (*byte)(unsafe.Pointer(&t)), uint32(unsafe.Sizeof(t)), &needed)
	if err != nil {
		return svc.Status{}, err
	}
	return svc.Status{
		State:                   svc.State(t.CurrentState),
		Accepts:                 svc.Accepted(t.ControlsAccepted),
		ProcessId:               t.ProcessId,
		Win32ExitCode:           t.Win32ExitCode,
		ServiceSpecificExitCode: t.ServiceSpecificExitCode,
	}, nil
}
