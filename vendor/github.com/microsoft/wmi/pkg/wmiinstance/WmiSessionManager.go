// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

package cim

import (
	"errors"
	"fmt"

	ole "github.com/go-ole/go-ole"
	"github.com/go-ole/go-ole/oleutil"
)

const S_FALSE = 0x00000001

var IID_ISWbemLocator = &ole.GUID{0x76a6415b, 0xcb41, 0x11d1, [8]byte{0x8b, 0x02, 0x00, 0x60, 0x08, 0x06, 0xd9, 0xb6}}

// Reference https://github.com/StackExchange/wmi
// Reference https://docs.microsoft.com/en-us/windows/desktop/WmiSdk/swbemlocator-connectserver

type WmiSessionManager struct {
	unknown  *ole.IUnknown
	wmi      *ole.IDispatch
	sessions map[string]*WmiSession
}

func NewWmiSessionManager() *WmiSessionManager {
	wsm := &WmiSessionManager{}

	err := wsm.init()
	if err != nil {
		panic("couldn't initialize the WmiSessionManager")
	}

	return wsm
}

func (c *WmiSessionManager) init() error {
	err := ole.CoInitializeEx(0, ole.COINIT_MULTITHREADED)
	if err != nil {
		oleCode := err.(*ole.OleError).Code()
		if oleCode != ole.S_OK && oleCode != S_FALSE {
			return err
		}
	}

	// Initialize COM security for the whole process
	err = CoInitializeSecurity(RPC_C_AUTHN_LEVEL_PKT_PRIVACY, RPC_C_IMP_LEVEL_IMPERSONATE)
	if err != nil {
		oleCode := err.(*ole.OleError).Code()

		// Note: RPC_E_TOO_LATE means we have already initialized security.
		if oleCode != ole.S_OK && oleCode != S_FALSE && oleCode != uintptr(RPC_E_TOO_LATE) {
			panic(fmt.Sprintf("Couldn't initialize COM/DCOM security. Error: [%v]", err))
		}
	}

	c.unknown, err = oleutil.CreateObject("WbemScripting.SWbemLocator")
	if err != nil {
		c.Dispose()
		return err
	}
	if c.unknown == nil {
		c.Dispose()
		return errors.New("CreateObject failed")
	}

	c.wmi, err = c.unknown.QueryInterface(IID_ISWbemLocator)
	if err != nil {
		c.Dispose()
		return err
	}

	return nil
}

// Dispose clears the WmiSessionManager
func (c *WmiSessionManager) Dispose() {
	c.Close()
}

// Close
func (c *WmiSessionManager) Close() error {
	// clear the Sessions

	if c.wmi != nil {
		c.wmi.Release()
	}
	// clear ole object
	if c.unknown != nil {
		c.unknown.Release()
	}

	// clear com
	ole.CoUninitialize()

	return nil
}

// GetSession
func (c *WmiSessionManager) GetSession(wmiNamespace, serverName, domain, userName, password string) (*WmiSession, error) {
	return CreateSession(c.wmi, wmiNamespace, serverName, domain, userName, password)
}

// GetLocalSession
func (c *WmiSessionManager) GetLocalSession(wmiNamespace string) (*WmiSession, error) {
	return CreateSession(c.wmi, wmiNamespace, "", "", "", "")
}
