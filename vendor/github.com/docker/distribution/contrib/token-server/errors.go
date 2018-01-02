package main

import (
	"net/http"

	"github.com/docker/distribution/registry/api/errcode"
)

var (
	errGroup = "tokenserver"

	// ErrorBadTokenOption is returned when a token parameter is invalid
	ErrorBadTokenOption = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "BAD_TOKEN_OPTION",
		Message: "bad token option",
		Description: `This error may be returned when a request for a
		token contains an option which is not valid`,
		HTTPStatusCode: http.StatusBadRequest,
	})

	// ErrorMissingRequiredField is returned when a required form field is missing
	ErrorMissingRequiredField = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "MISSING_REQUIRED_FIELD",
		Message: "missing required field",
		Description: `This error may be returned when a request for a
		token does not contain a required form field`,
		HTTPStatusCode: http.StatusBadRequest,
	})

	// ErrorUnsupportedValue is returned when a form field has an unsupported value
	ErrorUnsupportedValue = errcode.Register(errGroup, errcode.ErrorDescriptor{
		Value:   "UNSUPPORTED_VALUE",
		Message: "unsupported value",
		Description: `This error may be returned when a request for a
		token contains a form field with an unsupported value`,
		HTTPStatusCode: http.StatusBadRequest,
	})
)
