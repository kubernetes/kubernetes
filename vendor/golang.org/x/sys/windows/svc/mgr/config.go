// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows
// +build windows

package mgr

import (
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	// Service start types.
	StartManual    = windows.SERVICE_DEMAND_START // the service must be started manually
	StartAutomatic = windows.SERVICE_AUTO_START   // the service will start by itself whenever the computer reboots
	StartDisabled  = windows.SERVICE_DISABLED     // the service cannot be started

	// The severity of the error, and action taken,
	// if this service fails to start.
	ErrorCritical = windows.SERVICE_ERROR_CRITICAL
	ErrorIgnore   = windows.SERVICE_ERROR_IGNORE
	ErrorNormal   = windows.SERVICE_ERROR_NORMAL
	ErrorSevere   = windows.SERVICE_ERROR_SEVERE
)

// TODO(brainman): Password is not returned by windows.QueryServiceConfig, not sure how to get it.

type Config struct {
	ServiceType      uint32
	StartType        uint32
	ErrorControl     uint32
	BinaryPathName   string // fully qualified path to the service binary file, can also include arguments for an auto-start service
	LoadOrderGroup   string
	TagId            uint32
	Dependencies     []string
	ServiceStartName string // name of the account under which the service should run
	DisplayName      string
	Password         string
	Description      string
	SidType          uint32 // one of SERVICE_SID_TYPE, the type of sid to use for the service
	DelayedAutoStart bool   // the service is started after other auto-start services are started plus a short delay
}

func toStringSlice(ps *uint16) []string {
	r := make([]string, 0)
	p := unsafe.Pointer(ps)

	for {
		s := windows.UTF16PtrToString((*uint16)(p))
		if len(s) == 0 {
			break
		}

		r = append(r, s)
		offset := unsafe.Sizeof(uint16(0)) * (uintptr)(len(s)+1)
		p = unsafe.Pointer(uintptr(p) + offset)
	}

	return r
}

// Config retrieves service s configuration paramteres.
func (s *Service) Config() (Config, error) {
	var p *windows.QUERY_SERVICE_CONFIG
	n := uint32(1024)
	for {
		b := make([]byte, n)
		p = (*windows.QUERY_SERVICE_CONFIG)(unsafe.Pointer(&b[0]))
		err := windows.QueryServiceConfig(s.Handle, p, n, &n)
		if err == nil {
			break
		}
		if err.(syscall.Errno) != syscall.ERROR_INSUFFICIENT_BUFFER {
			return Config{}, err
		}
		if n <= uint32(len(b)) {
			return Config{}, err
		}
	}

	b, err := s.queryServiceConfig2(windows.SERVICE_CONFIG_DESCRIPTION)
	if err != nil {
		return Config{}, err
	}
	p2 := (*windows.SERVICE_DESCRIPTION)(unsafe.Pointer(&b[0]))

	b, err = s.queryServiceConfig2(windows.SERVICE_CONFIG_DELAYED_AUTO_START_INFO)
	if err != nil {
		return Config{}, err
	}
	p3 := (*windows.SERVICE_DELAYED_AUTO_START_INFO)(unsafe.Pointer(&b[0]))
	delayedStart := false
	if p3.IsDelayedAutoStartUp != 0 {
		delayedStart = true
	}

	b, err = s.queryServiceConfig2(windows.SERVICE_CONFIG_SERVICE_SID_INFO)
	if err != nil {
		return Config{}, err
	}
	sidType := *(*uint32)(unsafe.Pointer(&b[0]))

	return Config{
		ServiceType:      p.ServiceType,
		StartType:        p.StartType,
		ErrorControl:     p.ErrorControl,
		BinaryPathName:   windows.UTF16PtrToString(p.BinaryPathName),
		LoadOrderGroup:   windows.UTF16PtrToString(p.LoadOrderGroup),
		TagId:            p.TagId,
		Dependencies:     toStringSlice(p.Dependencies),
		ServiceStartName: windows.UTF16PtrToString(p.ServiceStartName),
		DisplayName:      windows.UTF16PtrToString(p.DisplayName),
		Description:      windows.UTF16PtrToString(p2.Description),
		DelayedAutoStart: delayedStart,
		SidType:          sidType,
	}, nil
}

func updateDescription(handle windows.Handle, desc string) error {
	d := windows.SERVICE_DESCRIPTION{Description: toPtr(desc)}
	return windows.ChangeServiceConfig2(handle,
		windows.SERVICE_CONFIG_DESCRIPTION, (*byte)(unsafe.Pointer(&d)))
}

func updateSidType(handle windows.Handle, sidType uint32) error {
	return windows.ChangeServiceConfig2(handle, windows.SERVICE_CONFIG_SERVICE_SID_INFO, (*byte)(unsafe.Pointer(&sidType)))
}

func updateStartUp(handle windows.Handle, isDelayed bool) error {
	var d windows.SERVICE_DELAYED_AUTO_START_INFO
	if isDelayed {
		d.IsDelayedAutoStartUp = 1
	}
	return windows.ChangeServiceConfig2(handle,
		windows.SERVICE_CONFIG_DELAYED_AUTO_START_INFO, (*byte)(unsafe.Pointer(&d)))
}

// UpdateConfig updates service s configuration parameters.
func (s *Service) UpdateConfig(c Config) error {
	err := windows.ChangeServiceConfig(s.Handle, c.ServiceType, c.StartType,
		c.ErrorControl, toPtr(c.BinaryPathName), toPtr(c.LoadOrderGroup),
		nil, toStringBlock(c.Dependencies), toPtr(c.ServiceStartName),
		toPtr(c.Password), toPtr(c.DisplayName))
	if err != nil {
		return err
	}
	err = updateSidType(s.Handle, c.SidType)
	if err != nil {
		return err
	}

	err = updateStartUp(s.Handle, c.DelayedAutoStart)
	if err != nil {
		return err
	}

	return updateDescription(s.Handle, c.Description)
}

// queryServiceConfig2 calls Windows QueryServiceConfig2 with infoLevel parameter and returns retrieved service configuration information.
func (s *Service) queryServiceConfig2(infoLevel uint32) ([]byte, error) {
	n := uint32(1024)
	for {
		b := make([]byte, n)
		err := windows.QueryServiceConfig2(s.Handle, infoLevel, &b[0], n, &n)
		if err == nil {
			return b, nil
		}
		if err.(syscall.Errno) != syscall.ERROR_INSUFFICIENT_BUFFER {
			return nil, err
		}
		if n <= uint32(len(b)) {
			return nil, err
		}
	}
}
