// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// This class implements the SWbemServices object
// Documentation: https://docs.microsoft.com/en-us/windows/win32/wmisdk/swbemservices

package cim

import (
	"fmt"
	"log"
	"runtime/debug"
	"strings"

	ole "github.com/go-ole/go-ole"
	"github.com/go-ole/go-ole/oleutil"
	"github.com/microsoft/wmi/go/wmi"
	"github.com/microsoft/wmi/pkg/base/host"
)

// WmiSession struct to hold the current session information
type WmiSession struct {
	ServerName    string
	Namespace     string
	Username      string
	Password      string
	Domain        string
	WMIHost       *host.WmiHost
	Status        wmi.SessionStatus
	RawSession    *ole.VARIANT
	Session       *ole.IDispatch
	CimwmiService *ole.IDispatch
}

// CreateSessionEx creates a session based on credentials
func CreateSessionEx(CimwmiService *ole.IDispatch, serverName, wmiNamespace string, credentials wmi.Credentials) (*WmiSession, error) {
	return CreateSession(CimwmiService, wmiNamespace, serverName, credentials.Domain, credentials.UserName, credentials.Password)
}

// CreateSession creates a new session with the server and namespace
func CreateSession(CimwmiService *ole.IDispatch, wmiNamespace, serverName, domain, userName, password string) (*WmiSession, error) {
	return &WmiSession{
		CimwmiService: CimwmiService,
		ServerName:    serverName,
		Namespace:     wmiNamespace,
		Username:      userName,
		Password:      password,
		Domain:        domain,
		Status:        wmi.Created,
		WMIHost:       host.NewWmiHostWithCredential(serverName, userName, password, domain),
	}, nil
}

type WbemAuthenticationLevelEnum uint32

const (
	wbemAuthenticationLevelDefault      WbemAuthenticationLevelEnum = 0
	wbemAuthenticationLevelNone         WbemAuthenticationLevelEnum = 1
	wbemAuthenticationLevelConnect      WbemAuthenticationLevelEnum = 2
	wbemAuthenticationLevelCall         WbemAuthenticationLevelEnum = 3
	wbemAuthenticationLevelPkt          WbemAuthenticationLevelEnum = 4
	wbemAuthenticationLevelPktIntegrity WbemAuthenticationLevelEnum = 5
	wbemAuthenticationLevelPktPrivacy   WbemAuthenticationLevelEnum = 6
)

type WbemImpersonationLevel uint32

const (
	wbemImpersonationLevelAnonymous   WbemImpersonationLevel = 1
	wbemImpersonationLevelIdentify    WbemImpersonationLevel = 2
	wbemImpersonationLevelImpersonate WbemImpersonationLevel = 3
	wbemImpersonationLevelDelegate    WbemImpersonationLevel = 4
)

// Connect the wmi session
func (c *WmiSession) Connect() (bool, error) {
	var err error
	// Node that we are connected through SWbemLocator, which uses the scripting language syntax for ConnectServer
	// This means the first parameter of the call is the name of the server, and the second parameter is the name of the namespace
	// (as opposed to C++ where these two are exposed as one parameter)
	// See here for an example illustrating the scripting syntax: https://docs.microsoft.com/en-us/windows/win32/wmisdk/connecting-to-wmi-with-vbscript
	c.RawSession, err = oleutil.CallMethod(
		c.CimwmiService, "ConnectServer", strings.Join([]string{c.ServerName, c.Domain}, "."), c.Namespace, c.Username, c.Password, "MS_409")
	if err != nil {
		return false, err
	}
	c.Session = c.RawSession.ToIDispatch()
	c.Status = wmi.Connected

	if c.Session == nil {
		panic("Returned session is null")
	}

	// Set the authentication level to packet privacy and impersonation to impersonate
	// Relevant documentations:
	// https://docs.microsoft.com/en-us/windows/win32/wmisdk/setting-security-on-an-asynchronous-call
	// https://docs.microsoft.com/en-us/windows/win32/wmisdk/setting-the-default-process-security-level-using-vbscript
	// https://docs.microsoft.com/en-us/windows/win32/wmisdk/swbemsecurity
	// https://docs.microsoft.com/en-us/windows/win32/wmisdk/maintaining-wmi-security
	// https://docs.microsoft.com/en-us/windows/win32/wmisdk/securing-scripting-clients
	// https://docs.microsoft.com/en-us/windows/win32/wmisdk/providing-events-securely
	rawSecurityObject, err := oleutil.GetProperty(c.Session, "Security_")
	if err != nil {
		debug.PrintStack()
		return false, err
	}
	securityObject := rawSecurityObject.ToIDispatch()
	defer rawSecurityObject.Clear()

	rawImpersonationLevel, err := oleutil.PutProperty(securityObject, "ImpersonationLevel", uint32(wbemImpersonationLevelImpersonate))
	if err != nil {
		debug.PrintStack()
		return false, err
	}
	defer rawImpersonationLevel.Clear()

	rawAuthenticationLevel, err := oleutil.PutProperty(securityObject, "AuthenticationLevel", uint32(wbemAuthenticationLevelPktPrivacy))
	if err != nil {
		debug.PrintStack()
		return false, err
	}
	defer rawAuthenticationLevel.Clear()

	return true, nil
}

// Close the wmi session
func (c *WmiSession) Close() {
	c.RawSession.Clear()
	c.Status = wmi.Disconnected
}

// Dispose the wmi session
func (c *WmiSession) Dispose() {
	if c.Status != wmi.Disposed {
		c.Close()
		c.Status = wmi.Disposed
	}
}

// TestConnection
func (c *WmiSession) TestConnection() bool {
	panic("not implemented")
}

// Tells WMI to create a new class for us
func (c *WmiSession) CreateNewClass() (*WmiClass, error) {
	rawResult, err := c.Session.CallMethod("Get")
	if err != nil {
		return nil, err
	}

	return CreateWmiClass(rawResult, c)
}

func (c *WmiSession) GetClass(classDefinition string) (*WmiClass, error) {
	rawResult, err := c.Session.CallMethod("Get", classDefinition)
	if err != nil {
		return nil, err
	}

	return CreateWmiClass(rawResult, c)
}

// EnumerateClasses
func (c *WmiSession) EnumerateClasses(className string) ([]*WmiClass, error) {
	return c.QueryClasses("SELECT * FROM meta_class")
}

// QueryClasses
func (c *WmiSession) QueryClasses(queryString string) ([]*WmiClass, error) {
	enum, err := c.PerformRawQuery(queryString)
	if err != nil {
		return nil, err
	}
	defer enum.Release()

	wmiClasses := []*WmiClass{}
	for tmp, length, err := enum.Next(1); length > 0; tmp, length, err = enum.Next(1) {
		if err != nil {
			return nil, err
		}

		wmiClass, err := CreateWmiClass(&tmp, c)
		if err != nil {
			return nil, err
		}

		wmiClasses = append(wmiClasses, wmiClass)
	}

	return wmiClasses, nil
}

// GetInstance
func (c *WmiSession) GetInstance(path string) (*WmiInstance, error) {
	rawResult, err := c.Session.CallMethod("Get", path)
	if err != nil {
		return nil, err
	}

	return CreateWmiInstance(rawResult, c)
}

// EnumerateInstances
func (c *WmiSession) EnumerateInstances(className string) ([]*WmiInstance, error) {
	return c.QueryInstances("SELECT * FROM " + className)
}

// QueryInstances
func (c *WmiSession) QueryInstances(queryExpression string) ([]*WmiInstance, error) {
	enum, err := c.PerformRawQuery(queryExpression)
	if err != nil {
		return nil, err
	}
	defer enum.Release()

	wmiInstances := []*WmiInstance{}
	for tmp, length, err := enum.Next(1); length > 0; tmp, length, err = enum.Next(1) {
		if err != nil {
			return nil, err
		}

		wmiInstance, err := CreateWmiInstance(&tmp, c)
		if err != nil {
			return nil, err
		}

		wmiInstances = append(wmiInstances, wmiInstance)
	}

	log.Printf("[WMI] QueryInstances [%s]=> [%d]\n", queryExpression, len(wmiInstances))
	return wmiInstances, nil
}

// QueryInstancesEx
func (c *WmiSession) QueryInstancesEx(query wmi.Query) (*[]wmi.Instance, error) {
	panic("not implemented")
}

// EnumerateReferencingInstances
func (c *WmiSession) EnumerateReferencingInstances(namespaceName string, sourceInstance WmiInstance, associationClassName, sourceRole string) (*[]WmiInstance, error) {
	panic("not implemented")
}

func (c *WmiSession) PerformRawQuery(queryExpression string) (*ole.IEnumVARIANT, error) {
	rawResult, err := c.Session.CallMethod("ExecQuery", queryExpression)
	if err != nil {
		return nil, err
	}

	result := rawResult.ToIDispatch()
	defer rawResult.Clear()

	// Doc: https://docs.microsoft.com/en-us/previous-versions/windows/desktop/automat/dispid-constants
	enum_property, err := result.GetProperty("_NewEnum")
	if err != nil {
		return nil, err
	}
	defer enum_property.Clear()

	// https://docs.microsoft.com/en-us/windows/win32/api/oaidl/nn-oaidl-ienumvariant
	enum, err := enum_property.ToIUnknown().IEnumVARIANT(ole.IID_IEnumVariant)
	if err != nil {
		return nil, err
	}
	if enum == nil {
		return nil, fmt.Errorf("Enum is nil")
	}

	return enum, err
}

// Credentials
func (c *WmiSession) Credentials() *wmi.Credentials {
	credentials := wmi.Credentials{
		UserName: c.Username,
		Password: c.Password,
		Domain:   c.Domain,
	}

	return &credentials
}

// Asynchronous approach to handling events
// Documentation: https://docs.microsoft.com/en-us/windows/win32/wmisdk/receiving-synchronous-and-semisynchronous-event-notifications
// Security considerations with Asynchronous calls: https://docs.microsoft.com/en-us/windows/win32/wmisdk/making-an-asynchronous-call-with-vbscript
func (c *WmiSession) ExecNotificationQueryAsync(eventSink *WmiEventSink, query string) (interface{}, error) {
	rawResult, err := oleutil.CallMethod(c.Session, "ExecNotificationQueryAsync", eventSink.instance, query)
	if err != nil {
		return nil, err
	}
	defer rawResult.Clear()

	return GetVariantValue(rawResult)
}

// Synchronous approach to handling events
// Documentation: https://docs.microsoft.com/en-us/windows/win32/wmisdk/receiving-synchronous-and-semisynchronous-event-notifications
func (c *WmiSession) ExecNotificationQuery(query string) (*WmiSynchronousEventsList, error) {
	rawResult, err := oleutil.CallMethod(c.Session, "ExecNotificationQuery", query)
	if err != nil {
		return nil, err
	}

	return CreateWmiSynchronousEventsList(rawResult, c), nil
}
