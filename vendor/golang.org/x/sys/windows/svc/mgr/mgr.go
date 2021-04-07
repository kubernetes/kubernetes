// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

// Package mgr can be used to manage Windows service programs.
// It can be used to install and remove them. It can also start,
// stop and pause them. The package can query / change current
// service state and config parameters.
//
package mgr

import (
	"syscall"
	"time"
	"unicode/utf16"
	"unsafe"

	"golang.org/x/sys/internal/unsafeheader"
	"golang.org/x/sys/windows"
)

// Mgr is used to manage Windows service.
type Mgr struct {
	Handle windows.Handle
}

// Connect establishes a connection to the service control manager.
func Connect() (*Mgr, error) {
	return ConnectRemote("")
}

// ConnectRemote establishes a connection to the
// service control manager on computer named host.
func ConnectRemote(host string) (*Mgr, error) {
	var s *uint16
	if host != "" {
		s = syscall.StringToUTF16Ptr(host)
	}
	h, err := windows.OpenSCManager(s, nil, windows.SC_MANAGER_ALL_ACCESS)
	if err != nil {
		return nil, err
	}
	return &Mgr{Handle: h}, nil
}

// Disconnect closes connection to the service control manager m.
func (m *Mgr) Disconnect() error {
	return windows.CloseServiceHandle(m.Handle)
}

type LockStatus struct {
	IsLocked bool          // Whether the SCM has been locked.
	Age      time.Duration // For how long the SCM has been locked.
	Owner    string        // The name of the user who has locked the SCM.
}

// LockStatus returns whether the service control manager is locked by
// the system, for how long, and by whom. A locked SCM indicates that
// most service actions will block until the system unlocks the SCM.
func (m *Mgr) LockStatus() (*LockStatus, error) {
	bytesNeeded := uint32(unsafe.Sizeof(windows.QUERY_SERVICE_LOCK_STATUS{}) + 1024)
	for {
		bytes := make([]byte, bytesNeeded)
		lockStatus := (*windows.QUERY_SERVICE_LOCK_STATUS)(unsafe.Pointer(&bytes[0]))
		err := windows.QueryServiceLockStatus(m.Handle, lockStatus, uint32(len(bytes)), &bytesNeeded)
		if err == windows.ERROR_INSUFFICIENT_BUFFER && bytesNeeded >= uint32(unsafe.Sizeof(windows.QUERY_SERVICE_LOCK_STATUS{})) {
			continue
		}
		if err != nil {
			return nil, err
		}
		status := &LockStatus{
			IsLocked: lockStatus.IsLocked != 0,
			Age:      time.Duration(lockStatus.LockDuration) * time.Second,
			Owner:    windows.UTF16PtrToString(lockStatus.LockOwner),
		}
		return status, nil
	}
}

func toPtr(s string) *uint16 {
	if len(s) == 0 {
		return nil
	}
	return syscall.StringToUTF16Ptr(s)
}

// toStringBlock terminates strings in ss with 0, and then
// concatenates them together. It also adds extra 0 at the end.
func toStringBlock(ss []string) *uint16 {
	if len(ss) == 0 {
		return nil
	}
	t := ""
	for _, s := range ss {
		if s != "" {
			t += s + "\x00"
		}
	}
	if t == "" {
		return nil
	}
	t += "\x00"
	return &utf16.Encode([]rune(t))[0]
}

// CreateService installs new service name on the system.
// The service will be executed by running exepath binary.
// Use config c to specify service parameters.
// Any args will be passed as command-line arguments when
// the service is started; these arguments are distinct from
// the arguments passed to Service.Start or via the "Start
// parameters" field in the service's Properties dialog box.
func (m *Mgr) CreateService(name, exepath string, c Config, args ...string) (*Service, error) {
	if c.StartType == 0 {
		c.StartType = StartManual
	}
	if c.ServiceType == 0 {
		c.ServiceType = windows.SERVICE_WIN32_OWN_PROCESS
	}
	s := syscall.EscapeArg(exepath)
	for _, v := range args {
		s += " " + syscall.EscapeArg(v)
	}
	h, err := windows.CreateService(m.Handle, toPtr(name), toPtr(c.DisplayName),
		windows.SERVICE_ALL_ACCESS, c.ServiceType,
		c.StartType, c.ErrorControl, toPtr(s), toPtr(c.LoadOrderGroup),
		nil, toStringBlock(c.Dependencies), toPtr(c.ServiceStartName), toPtr(c.Password))
	if err != nil {
		return nil, err
	}
	if c.SidType != windows.SERVICE_SID_TYPE_NONE {
		err = updateSidType(h, c.SidType)
		if err != nil {
			windows.DeleteService(h)
			windows.CloseServiceHandle(h)
			return nil, err
		}
	}
	if c.Description != "" {
		err = updateDescription(h, c.Description)
		if err != nil {
			windows.DeleteService(h)
			windows.CloseServiceHandle(h)
			return nil, err
		}
	}
	if c.DelayedAutoStart {
		err = updateStartUp(h, c.DelayedAutoStart)
		if err != nil {
			windows.DeleteService(h)
			windows.CloseServiceHandle(h)
			return nil, err
		}
	}
	return &Service{Name: name, Handle: h}, nil
}

// OpenService retrieves access to service name, so it can
// be interrogated and controlled.
func (m *Mgr) OpenService(name string) (*Service, error) {
	h, err := windows.OpenService(m.Handle, syscall.StringToUTF16Ptr(name), windows.SERVICE_ALL_ACCESS)
	if err != nil {
		return nil, err
	}
	return &Service{Name: name, Handle: h}, nil
}

// ListServices enumerates services in the specified
// service control manager database m.
// If the caller does not have the SERVICE_QUERY_STATUS
// access right to a service, the service is silently
// omitted from the list of services returned.
func (m *Mgr) ListServices() ([]string, error) {
	var err error
	var bytesNeeded, servicesReturned uint32
	var buf []byte
	for {
		var p *byte
		if len(buf) > 0 {
			p = &buf[0]
		}
		err = windows.EnumServicesStatusEx(m.Handle, windows.SC_ENUM_PROCESS_INFO,
			windows.SERVICE_WIN32, windows.SERVICE_STATE_ALL,
			p, uint32(len(buf)), &bytesNeeded, &servicesReturned, nil, nil)
		if err == nil {
			break
		}
		if err != syscall.ERROR_MORE_DATA {
			return nil, err
		}
		if bytesNeeded <= uint32(len(buf)) {
			return nil, err
		}
		buf = make([]byte, bytesNeeded)
	}
	if servicesReturned == 0 {
		return nil, nil
	}

	var services []windows.ENUM_SERVICE_STATUS_PROCESS
	hdr := (*unsafeheader.Slice)(unsafe.Pointer(&services))
	hdr.Data = unsafe.Pointer(&buf[0])
	hdr.Len = int(servicesReturned)
	hdr.Cap = int(servicesReturned)

	var names []string
	for _, s := range services {
		name := windows.UTF16PtrToString(s.ServiceName)
		names = append(names, name)
	}
	return names, nil
}
