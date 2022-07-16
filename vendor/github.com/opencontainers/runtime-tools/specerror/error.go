// Package specerror implements runtime-spec-specific tooling for
// tracking RFC 2119 violations.
package specerror

import (
	"fmt"

	"github.com/hashicorp/go-multierror"
	rfc2119 "github.com/opencontainers/runtime-tools/error"
)

const referenceTemplate = "https://github.com/opencontainers/runtime-spec/blob/v%s/%s"

// Code represents the spec violation, enumerating both
// configuration violations and runtime violations.
type Code int64

const (
	// NonError represents that an input is not an error
	NonError Code = 0x1a001 + iota
	// NonRFCError represents that an error is not a rfc2119 error
	NonRFCError
)

type errorTemplate struct {
	Level     rfc2119.Level
	Reference func(version string) (reference string, err error)
}

// Error represents a runtime-spec violation.
type Error struct {
	// Err holds the RFC 2119 violation.
	Err rfc2119.Error

	// Code is a matchable holds a Code
	Code Code
}

// LevelErrors represents Errors filtered into fatal and warnings.
type LevelErrors struct {
	// Warnings holds Errors that were below a compliance-level threshold.
	Warnings []*Error

	// Error holds errors that were at or above a compliance-level
	// threshold, as well as errors that are not Errors.
	Error *multierror.Error
}

var ociErrors = map[Code]errorTemplate{}

func register(code Code, level rfc2119.Level, ref func(versiong string) (string, error)) {
	if _, ok := ociErrors[code]; ok {
		panic(fmt.Sprintf("should not regist a same code twice: %v", code))
	}

	ociErrors[code] = errorTemplate{Level: level, Reference: ref}
}

// Error returns the error message with specification reference.
func (err *Error) Error() string {
	return err.Err.Error()
}

// NewRFCError creates an rfc2119.Error referencing a spec violation.
//
// A version string (for the version of the spec that was violated)
// must be set to get a working URL.
func NewRFCError(code Code, err error, version string) (*rfc2119.Error, error) {
	template := ociErrors[code]
	reference, err2 := template.Reference(version)
	if err2 != nil {
		return nil, err2
	}
	return &rfc2119.Error{
		Level:     template.Level,
		Reference: reference,
		Err:       err,
	}, nil
}

// NewRFCErrorOrPanic creates an rfc2119.Error referencing a spec
// violation and panics on failure.  This is handy for situations
// where you can't be bothered to check NewRFCError for failure.
func NewRFCErrorOrPanic(code Code, err error, version string) *rfc2119.Error {
	rfcError, err2 := NewRFCError(code, err, version)
	if err2 != nil {
		panic(err2.Error())
	}
	return rfcError
}

// NewError creates an Error referencing a spec violation.  The error
// can be cast to an *Error for extracting structured information
// about the level of the violation and a reference to the violated
// spec condition.
//
// A version string (for the version of the spec that was violated)
// must be set to get a working URL.
func NewError(code Code, err error, version string) error {
	rfcError, err2 := NewRFCError(code, err, version)
	if err2 != nil {
		return err2
	}
	return &Error{
		Err:  *rfcError,
		Code: code,
	}
}

// FindError finds an error from a source error (multiple error) and
// returns the error code if found.
// If the source error is nil or empty, return NonError.
// If the source error is not a multiple error, return NonRFCError.
func FindError(err error, code Code) Code {
	if err == nil {
		return NonError
	}

	if merr, ok := err.(*multierror.Error); ok {
		if merr.ErrorOrNil() == nil {
			return NonError
		}
		for _, e := range merr.Errors {
			if rfcErr, ok := e.(*Error); ok {
				if rfcErr.Code == code {
					return code
				}
			}
		}
	}
	return NonRFCError
}

// SplitLevel removes RFC 2119 errors with a level less than 'level'
// from the source error.  If the source error is not a multierror, it
// is returned unchanged.
func SplitLevel(errIn error, level rfc2119.Level) (levelErrors LevelErrors, errOut error) {
	merr, ok := errIn.(*multierror.Error)
	if !ok {
		return levelErrors, errIn
	}
	for _, err := range merr.Errors {
		e, ok := err.(*Error)
		if ok && e.Err.Level < level {
			fmt.Println(e)
			levelErrors.Warnings = append(levelErrors.Warnings, e)
			continue
		}
		levelErrors.Error = multierror.Append(levelErrors.Error, err)
	}
	return levelErrors, nil
}
