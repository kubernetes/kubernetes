package parse

import (
	"errors"
)

// errMissingSpecificType represents an error when a generic type is not
// satisfied by a specific type.
type errMissingSpecificType struct {
	GenericType string
}

// Error gets a human readable string describing this error.
func (e errMissingSpecificType) Error() string {
	return "Missing specific type for '" + e.GenericType + "' generic type"
}

// errImports represents an error from goimports.
type errImports struct {
	Err error
}

// Error gets a human readable string describing this error.
func (e errImports) Error() string {
	return "Failed to goimports the generated code: " + e.Err.Error()
}

// errSource represents an error with the source file.
type errSource struct {
	Err error
}

// Error gets a human readable string describing this error.
func (e errSource) Error() string {
	return "Failed to parse source file: " + e.Err.Error()
}

type errBadTypeArgs struct {
	Message string
	Arg     string
}

func (e errBadTypeArgs) Error() string {
	return "\"" + e.Arg + "\" is bad: " + e.Message
}

var errMissingTypeInformation = errors.New("No type arguments were specified and no \"// +gogen\" tag was found in the source.")
