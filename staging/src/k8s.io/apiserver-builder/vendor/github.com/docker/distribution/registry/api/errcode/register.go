package errcode

import (
	"fmt"
	"net/http"
	"sort"
	"sync"
)

var (
	errorCodeToDescriptors = map[ErrorCode]ErrorDescriptor{}
	idToDescriptors        = map[string]ErrorDescriptor{}
	groupToDescriptors     = map[string][]ErrorDescriptor{}
)

var (
	// ErrorCodeUnknown is a generic error that can be used as a last
	// resort if there is no situation-specific error message that can be used
	ErrorCodeUnknown = Register("errcode", ErrorDescriptor{
		Value:   "UNKNOWN",
		Message: "unknown error",
		Description: `Generic error returned when the error does not have an
			                                            API classification.`,
		HTTPStatusCode: http.StatusInternalServerError,
	})

	// ErrorCodeUnsupported is returned when an operation is not supported.
	ErrorCodeUnsupported = Register("errcode", ErrorDescriptor{
		Value:   "UNSUPPORTED",
		Message: "The operation is unsupported.",
		Description: `The operation was unsupported due to a missing
		implementation or invalid set of parameters.`,
		HTTPStatusCode: http.StatusMethodNotAllowed,
	})

	// ErrorCodeUnauthorized is returned if a request requires
	// authentication.
	ErrorCodeUnauthorized = Register("errcode", ErrorDescriptor{
		Value:   "UNAUTHORIZED",
		Message: "authentication required",
		Description: `The access controller was unable to authenticate
		the client. Often this will be accompanied by a
		Www-Authenticate HTTP response header indicating how to
		authenticate.`,
		HTTPStatusCode: http.StatusUnauthorized,
	})

	// ErrorCodeDenied is returned if a client does not have sufficient
	// permission to perform an action.
	ErrorCodeDenied = Register("errcode", ErrorDescriptor{
		Value:   "DENIED",
		Message: "requested access to the resource is denied",
		Description: `The access controller denied access for the
		operation on a resource.`,
		HTTPStatusCode: http.StatusForbidden,
	})

	// ErrorCodeUnavailable provides a common error to report unavialability
	// of a service or endpoint.
	ErrorCodeUnavailable = Register("errcode", ErrorDescriptor{
		Value:          "UNAVAILABLE",
		Message:        "service unavailable",
		Description:    "Returned when a service is not available",
		HTTPStatusCode: http.StatusServiceUnavailable,
	})
)

var nextCode = 1000
var registerLock sync.Mutex

// Register will make the passed-in error known to the environment and
// return a new ErrorCode
func Register(group string, descriptor ErrorDescriptor) ErrorCode {
	registerLock.Lock()
	defer registerLock.Unlock()

	descriptor.Code = ErrorCode(nextCode)

	if _, ok := idToDescriptors[descriptor.Value]; ok {
		panic(fmt.Sprintf("ErrorValue %q is already registered", descriptor.Value))
	}
	if _, ok := errorCodeToDescriptors[descriptor.Code]; ok {
		panic(fmt.Sprintf("ErrorCode %v is already registered", descriptor.Code))
	}

	groupToDescriptors[group] = append(groupToDescriptors[group], descriptor)
	errorCodeToDescriptors[descriptor.Code] = descriptor
	idToDescriptors[descriptor.Value] = descriptor

	nextCode++
	return descriptor.Code
}

type byValue []ErrorDescriptor

func (a byValue) Len() int           { return len(a) }
func (a byValue) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byValue) Less(i, j int) bool { return a[i].Value < a[j].Value }

// GetGroupNames returns the list of Error group names that are registered
func GetGroupNames() []string {
	keys := []string{}

	for k := range groupToDescriptors {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// GetErrorCodeGroup returns the named group of error descriptors
func GetErrorCodeGroup(name string) []ErrorDescriptor {
	desc := groupToDescriptors[name]
	sort.Sort(byValue(desc))
	return desc
}

// GetErrorAllDescriptors returns a slice of all ErrorDescriptors that are
// registered, irrespective of what group they're in
func GetErrorAllDescriptors() []ErrorDescriptor {
	result := []ErrorDescriptor{}

	for _, group := range GetGroupNames() {
		result = append(result, GetErrorCodeGroup(group)...)
	}
	sort.Sort(byValue(result))
	return result
}
