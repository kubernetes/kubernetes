// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package mgr

import (
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/svc"
)

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

// Control sends state change request c to the service s. It returns the most
// recent status the service reported to the service control manager, and an
// error if the state change request was not accepted.
// Note that the returned service status is only set if the status change
// request succeeded, or if it failed with error ERROR_INVALID_SERVICE_CONTROL,
// ERROR_SERVICE_CANNOT_ACCEPT_CTRL, or ERROR_SERVICE_NOT_ACTIVE.
func (s *Service) Control(c svc.Cmd) (svc.Status, error) {
	var t windows.SERVICE_STATUS
	err := windows.ControlService(s.Handle, uint32(c), &t)
	if err != nil &&
		err != windows.ERROR_INVALID_SERVICE_CONTROL &&
		err != windows.ERROR_SERVICE_CANNOT_ACCEPT_CTRL &&
		err != windows.ERROR_SERVICE_NOT_ACTIVE {
		return svc.Status{}, err
	}
	return svc.Status{
		State:   svc.State(t.CurrentState),
		Accepts: svc.Accepted(t.ControlsAccepted),
	}, err
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

// ListDependentServices returns the names of the services dependent on service s, which match the given status.
func (s *Service) ListDependentServices(status svc.ActivityStatus) ([]string, error) {
	var bytesNeeded, returnedServiceCount uint32
	var services []windows.ENUM_SERVICE_STATUS
	for {
		var servicesPtr *windows.ENUM_SERVICE_STATUS
		if len(services) > 0 {
			servicesPtr = &services[0]
		}
		allocatedBytes := uint32(len(services)) * uint32(unsafe.Sizeof(windows.ENUM_SERVICE_STATUS{}))
		err := windows.EnumDependentServices(s.Handle, uint32(status), servicesPtr, allocatedBytes, &bytesNeeded,
			&returnedServiceCount)
		if err == nil {
			break
		}
		if err != syscall.ERROR_MORE_DATA {
			return nil, err
		}
		if bytesNeeded <= allocatedBytes {
			return nil, err
		}
		// ERROR_MORE_DATA indicates the provided buffer was too small, run the call again after resizing the buffer
		requiredSliceLen := bytesNeeded / uint32(unsafe.Sizeof(windows.ENUM_SERVICE_STATUS{}))
		if bytesNeeded%uint32(unsafe.Sizeof(windows.ENUM_SERVICE_STATUS{})) != 0 {
			requiredSliceLen += 1
		}
		services = make([]windows.ENUM_SERVICE_STATUS, requiredSliceLen)
	}
	if returnedServiceCount == 0 {
		return nil, nil
	}

	// The slice mutated by EnumDependentServices may have a length greater than returnedServiceCount, any elements
	// past that should be ignored.
	var dependents []string
	for i := 0; i < int(returnedServiceCount); i++ {
		dependents = append(dependents, windows.UTF16PtrToString(services[i].ServiceName))
	}
	return dependents, nil
}
