//go:build windows
// +build windows

package cim

import (
	"errors"
	"fmt"
	"runtime"

	"github.com/go-ole/go-ole"
	"github.com/microsoft/wmi/pkg/base/query"
	wmierrors "github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"golang.org/x/sys/windows"
	"k8s.io/klog/v2"
)

const (
	WMINamespaceCimV2   = "Root\\CimV2"
	WMINamespaceStorage = "Root\\Microsoft\\Windows\\Storage"
	WMINamespaceSmb     = "Root\\Microsoft\\Windows\\Smb"
)

type InstanceHandler func(instance *cim.WmiInstance) (bool, error)

// NewWMISession creates a new local WMI session for the given namespace, defaulting
// to root namespace if none specified.
func NewWMISession(namespace string) (*cim.WmiSession, error) {
	if namespace == "" {
		namespace = WMINamespaceCimV2
	}

	sessionManager := cim.NewWmiSessionManager()
	defer sessionManager.Dispose()

	session, err := sessionManager.GetLocalSession(namespace)
	if err != nil {
		return nil, fmt.Errorf("failed to get local WMI session for namespace %s. error: %w", namespace, err)
	}

	connected, err := session.Connect()
	if !connected || err != nil {
		return nil, fmt.Errorf("failed to connect to WMI. error: %w", err)
	}

	return session, nil
}

// QueryFromWMI executes a WMI query in the specified namespace and processes each result
// through the provided handler function. Stops processing if handler returns false or encounters an error.
func QueryFromWMI(namespace string, query *query.WmiQuery, handler InstanceHandler) error {
	session, err := NewWMISession(namespace)
	if err != nil {
		return err
	}

	defer session.Close()

	instances, err := session.QueryInstances(query.String())
	if err != nil {
		return fmt.Errorf("failed to query WMI class %s. error: %w", query.ClassName, err)
	}

	if len(instances) == 0 {
		return wmierrors.NotFound
	}

	var cont bool
	for _, instance := range instances {
		cont, err = handler(instance)
		if err != nil {
			err = fmt.Errorf("failed to query WMI class %s instance (%s). error: %w", query.ClassName, instance.String(), err)
		}
		if !cont {
			break
		}
	}

	return err
}

// QueryInstances retrieves all WMI instances matching the given query in the specified namespace.
func QueryInstances(namespace string, query *query.WmiQuery) ([]*cim.WmiInstance, error) {
	var instances []*cim.WmiInstance
	err := QueryFromWMI(namespace, query, func(instance *cim.WmiInstance) (bool, error) {
		instances = append(instances, instance)
		return true, nil
	})
	return instances, err
}

// InvokeCimMethod calls a static method on a specific WMI class with given input parameters,
// returning the method's return value, output parameters, and any error encountered.
func InvokeCimMethod(namespace, class, methodName string, inputParameters map[string]interface{}) (int, map[string]interface{}, error) {
	session, err := NewWMISession(namespace)
	if err != nil {
		return -1, nil, err
	}

	defer session.Close()

	rawResult, err := session.Session.CallMethod("Get", class)
	if err != nil {
		return -1, nil, err
	}

	classInst, err := cim.CreateWmiInstance(rawResult, session)
	if err != nil {
		return -1, nil, err
	}

	method, err := cim.NewWmiMethod(methodName, classInst)
	if err != nil {
		return -1, nil, err
	}

	var inParam cim.WmiMethodParamCollection
	for k, v := range inputParameters {
		inParam = append(inParam, &cim.WmiMethodParam{
			Name:  k,
			Value: v,
		})
	}

	var outParam cim.WmiMethodParamCollection
	var result *cim.WmiMethodResult
	result, err = method.Execute(inParam, outParam)
	if err != nil {
		return -1, nil, err
	}

	outputParameters := make(map[string]interface{})
	for _, v := range result.OutMethodParams {
		outputParameters[v.Name] = v.Value
	}

	return int(result.ReturnValue), outputParameters, nil
}

// IsNotFound returns true if it's a "not found" error.
func IsNotFound(err error) bool {
	return wmierrors.IsNotFound(err)
}

// IgnoreNotFound returns nil if the error is nil or a "not found" error,
// otherwise returns the original error.
func IgnoreNotFound(err error) error {
	if err == nil || IsNotFound(err) {
		return nil
	}
	return err
}

// WithCOMThread runs the given function `fn` on a locked OS thread
// with COM initialized using COINIT_MULTITHREADED.
//
// This is necessary for using COM/OLE APIs directly (e.g., via go-ole),
// because COM requires that initialization and usage occur on the same thread.
//
// It performs the following steps:
//   - Locks the current goroutine to its OS thread
//   - Calls ole.CoInitializeEx with COINIT_MULTITHREADED
//   - Executes the user-provided function
//   - Uninitializes COM
//   - Unlocks the thread
//
// If COM initialization fails, or if the user's function returns an error,
// that error is returned by WithCOMThread.
func WithCOMThread(fn func() error) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := ole.CoInitializeEx(0, ole.COINIT_MULTITHREADED); err != nil {
		var oleError *ole.OleError
		if errors.As(err, &oleError) && oleError != nil && oleError.Code() == uintptr(windows.S_FALSE) {
			klog.V(10).Infof("COM library has been already initialized for the calling thread, proceeding to the function with no error")
			err = nil
		}
		if err != nil {
			return err
		}
	} else {
		klog.V(10).Infof("COM library is initialized for the calling thread")
	}
	defer ole.CoUninitialize()

	return fn()
}
