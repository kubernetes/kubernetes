// Package hcn is a shim for the Host Compute Networking (HCN) service, which manages networking for Windows Server
// containers and Hyper-V containers. Previous to RS5, HCN was referred to as Host Networking Service (HNS).
package hcn

import (
	"fmt"

	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/sirupsen/logrus"
)

func checkForErrors(methodName string, hr error, resultBuffer *uint16) error {
	errorFound := false

	if hr != nil {
		errorFound = true
	}

	result := ""
	if resultBuffer != nil {
		result = interop.ConvertAndFreeCoTaskMemString(resultBuffer)
		if result != "" {
			errorFound = true
		}
	}

	if errorFound {
		returnError := hcserror.New(hr, methodName, result)
		logrus.Debugf(returnError.Error()) // HCN errors logged for debugging.
		return returnError
	}

	return nil
}

// NetworkNotFoundError results from a failed seach for a network by Id or Name
type NetworkNotFoundError struct {
	NetworkName string
	NetworkID   string
}

func (e NetworkNotFoundError) Error() string {
	if e.NetworkName == "" {
		return fmt.Sprintf("Network Name %s not found", e.NetworkName)
	}
	return fmt.Sprintf("Network Id %s not found", e.NetworkID)
}

// EndpointNotFoundError results from a failed seach for an endpoint by Id or Name
type EndpointNotFoundError struct {
	EndpointName string
	EndpointID   string
}

func (e EndpointNotFoundError) Error() string {
	if e.EndpointName == "" {
		return fmt.Sprintf("Endpoint Name %s not found", e.EndpointName)
	}
	return fmt.Sprintf("Endpoint Id %s not found", e.EndpointID)
}

// NamespaceNotFoundError results from a failed seach for a namsepace by Id
type NamespaceNotFoundError struct {
	NamespaceID string
}

func (e NamespaceNotFoundError) Error() string {
	return fmt.Sprintf("Namespace %s not found", e.NamespaceID)
}

// LoadBalancerNotFoundError results from a failed seach for a loadbalancer by Id
type LoadBalancerNotFoundError struct {
	LoadBalancerId string
}

func (e LoadBalancerNotFoundError) Error() string {
	return fmt.Sprintf("LoadBalancer %s not found", e.LoadBalancerId)
}

// IsNotFoundError returns a boolean indicating whether the error was caused by
// a resource not being found.
func IsNotFoundError(err error) bool {
	switch err.(type) {
	case NetworkNotFoundError:
		return true
	case EndpointNotFoundError:
		return true
	case NamespaceNotFoundError:
		return true
	case LoadBalancerNotFoundError:
		return true
	}
	return false
}
