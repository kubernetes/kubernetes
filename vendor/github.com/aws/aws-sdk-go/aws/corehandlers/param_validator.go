package corehandlers

import "github.com/aws/aws-sdk-go/aws/request"

// ValidateParametersHandler is a request handler to validate the input parameters.
// Validating parameters only has meaning if done prior to the request being sent.
var ValidateParametersHandler = request.NamedHandler{Name: "core.ValidateParametersHandler", Fn: func(r *request.Request) {
	if !r.ParamsFilled() {
		return
	}

	if v, ok := r.Params.(request.Validator); ok {
		if err := v.Validate(); err != nil {
			r.Error = err
		}
	}
}}
