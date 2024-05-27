//go:build windows

package hcn

import (
	"errors"
	"fmt"

	"github.com/sirupsen/logrus"
	"golang.org/x/sys/windows"

	"github.com/Microsoft/hcsshim/internal/hcs"
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/interop"
)

var (
	errInvalidNetworkID      = errors.New("invalid network ID")
	errInvalidEndpointID     = errors.New("invalid endpoint ID")
	errInvalidNamespaceID    = errors.New("invalid namespace ID")
	errInvalidLoadBalancerID = errors.New("invalid load balancer ID")
	errInvalidRouteID        = errors.New("invalid route ID")
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
		returnError := new(hr, methodName, result)
		logrus.Debugf(returnError.Error()) // HCN errors logged for debugging.
		return returnError
	}

	return nil
}

type ErrorCode uint32

// For common errors, define the error as it is in windows, so we can quickly determine it later
const (
	ERROR_NOT_FOUND                     = ErrorCode(windows.ERROR_NOT_FOUND)
	HCN_E_PORT_ALREADY_EXISTS ErrorCode = ErrorCode(windows.HCN_E_PORT_ALREADY_EXISTS)
)

type HcnError struct {
	*hcserror.HcsError
	code ErrorCode
}

func (e *HcnError) Error() string {
	return e.HcsError.Error()
}

func CheckErrorWithCode(err error, code ErrorCode) bool {
	var hcnError *HcnError
	if errors.As(err, &hcnError) {
		return hcnError.code == code
	}
	return false
}

func IsElementNotFoundError(err error) bool {
	return CheckErrorWithCode(err, ERROR_NOT_FOUND)
}

func IsPortAlreadyExistsError(err error) bool {
	return CheckErrorWithCode(err, HCN_E_PORT_ALREADY_EXISTS)
}

func new(hr error, title string, rest string) error {
	err := &HcnError{}
	hcsError := hcserror.New(hr, title, rest)
	err.HcsError = hcsError.(*hcserror.HcsError) //nolint:errorlint
	err.code = ErrorCode(hcserror.Win32FromError(hr))
	return err
}

//
// Note that the below errors are not errors returned by hcn itself
// we wish to separate them as they are shim usage error
//

// NetworkNotFoundError results from a failed search for a network by Id or Name
type NetworkNotFoundError struct {
	NetworkName string
	NetworkID   string
}

var _ error = NetworkNotFoundError{}

func (e NetworkNotFoundError) Error() string {
	if e.NetworkName != "" {
		return fmt.Sprintf("Network name %q not found", e.NetworkName)
	}
	return fmt.Sprintf("Network ID %q not found", e.NetworkID)
}

// EndpointNotFoundError results from a failed search for an endpoint by Id or Name
type EndpointNotFoundError struct {
	EndpointName string
	EndpointID   string
}

var _ error = EndpointNotFoundError{}

func (e EndpointNotFoundError) Error() string {
	if e.EndpointName != "" {
		return fmt.Sprintf("Endpoint name %q not found", e.EndpointName)
	}
	return fmt.Sprintf("Endpoint ID %q not found", e.EndpointID)
}

// NamespaceNotFoundError results from a failed search for a namsepace by Id
type NamespaceNotFoundError struct {
	NamespaceID string
}

var _ error = NamespaceNotFoundError{}

func (e NamespaceNotFoundError) Error() string {
	return fmt.Sprintf("Namespace ID %q not found", e.NamespaceID)
}

// LoadBalancerNotFoundError results from a failed search for a loadbalancer by Id
type LoadBalancerNotFoundError struct {
	LoadBalancerId string
}

var _ error = LoadBalancerNotFoundError{}

func (e LoadBalancerNotFoundError) Error() string {
	return fmt.Sprintf("LoadBalancer %q not found", e.LoadBalancerId)
}

// RouteNotFoundError results from a failed search for a route by Id
type RouteNotFoundError struct {
	RouteId string
}

var _ error = RouteNotFoundError{}

func (e RouteNotFoundError) Error() string {
	return fmt.Sprintf("SDN Route %q not found", e.RouteId)
}

// IsNotFoundError returns a boolean indicating whether the error was caused by
// a resource not being found.
func IsNotFoundError(err error) bool {
	// Calling [errors.As] in a loop over `[]error{NetworkNotFoundError{}, ...}` will not work,
	// since the loop variable will be an interface type (ie, `error`) and `errors.As(error, *error)` will
	// always succeed.
	// Unless golang adds loops over (or arrays of) types, we need to manually call `errors.As` for
	// each potential error type.
	//
	// Also, for T = NetworkNotFoundError and co, the error implementation is for T, not *T
	if e := (NetworkNotFoundError{}); errors.As(err, &e) {
		return true
	}
	if e := (EndpointNotFoundError{}); errors.As(err, &e) {
		return true
	}
	if e := (NamespaceNotFoundError{}); errors.As(err, &e) {
		return true
	}
	if e := (LoadBalancerNotFoundError{}); errors.As(err, &e) {
		return true
	}
	if e := (RouteNotFoundError{}); errors.As(err, &e) {
		return true
	}
	if e := (&hcserror.HcsError{}); errors.As(err, &e) {
		return errors.Is(e.Err, hcs.ErrElementNotFound)
	}

	return false
}
