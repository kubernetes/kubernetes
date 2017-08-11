package request

import (
	"bytes"
	"fmt"

	"github.com/aws/aws-sdk-go/aws/awserr"
)

const (
	// InvalidParameterErrCode is the error code for invalid parameters errors
	InvalidParameterErrCode = "InvalidParameter"
	// ParamRequiredErrCode is the error code for required parameter errors
	ParamRequiredErrCode = "ParamRequiredError"
	// ParamMinValueErrCode is the error code for fields with too low of a
	// number value.
	ParamMinValueErrCode = "ParamMinValueError"
	// ParamMinLenErrCode is the error code for fields without enough elements.
	ParamMinLenErrCode = "ParamMinLenError"
)

// Validator provides a way for types to perform validation logic on their
// input values that external code can use to determine if a type's values
// are valid.
type Validator interface {
	Validate() error
}

// An ErrInvalidParams provides wrapping of invalid parameter errors found when
// validating API operation input parameters.
type ErrInvalidParams struct {
	// Context is the base context of the invalid parameter group.
	Context string
	errs    []ErrInvalidParam
}

// Add adds a new invalid parameter error to the collection of invalid
// parameters. The context of the invalid parameter will be updated to reflect
// this collection.
func (e *ErrInvalidParams) Add(err ErrInvalidParam) {
	err.SetContext(e.Context)
	e.errs = append(e.errs, err)
}

// AddNested adds the invalid parameter errors from another ErrInvalidParams
// value into this collection. The nested errors will have their nested context
// updated and base context to reflect the merging.
//
// Use for nested validations errors.
func (e *ErrInvalidParams) AddNested(nestedCtx string, nested ErrInvalidParams) {
	for _, err := range nested.errs {
		err.SetContext(e.Context)
		err.AddNestedContext(nestedCtx)
		e.errs = append(e.errs, err)
	}
}

// Len returns the number of invalid parameter errors
func (e ErrInvalidParams) Len() int {
	return len(e.errs)
}

// Code returns the code of the error
func (e ErrInvalidParams) Code() string {
	return InvalidParameterErrCode
}

// Message returns the message of the error
func (e ErrInvalidParams) Message() string {
	return fmt.Sprintf("%d validation error(s) found.", len(e.errs))
}

// Error returns the string formatted form of the invalid parameters.
func (e ErrInvalidParams) Error() string {
	w := &bytes.Buffer{}
	fmt.Fprintf(w, "%s: %s\n", e.Code(), e.Message())

	for _, err := range e.errs {
		fmt.Fprintf(w, "- %s\n", err.Message())
	}

	return w.String()
}

// OrigErr returns the invalid parameters as a awserr.BatchedErrors value
func (e ErrInvalidParams) OrigErr() error {
	return awserr.NewBatchError(
		InvalidParameterErrCode, e.Message(), e.OrigErrs())
}

// OrigErrs returns a slice of the invalid parameters
func (e ErrInvalidParams) OrigErrs() []error {
	errs := make([]error, len(e.errs))
	for i := 0; i < len(errs); i++ {
		errs[i] = e.errs[i]
	}

	return errs
}

// An ErrInvalidParam represents an invalid parameter error type.
type ErrInvalidParam interface {
	awserr.Error

	// Field name the error occurred on.
	Field() string

	// SetContext updates the context of the error.
	SetContext(string)

	// AddNestedContext updates the error's context to include a nested level.
	AddNestedContext(string)
}

type errInvalidParam struct {
	context       string
	nestedContext string
	field         string
	code          string
	msg           string
}

// Code returns the error code for the type of invalid parameter.
func (e *errInvalidParam) Code() string {
	return e.code
}

// Message returns the reason the parameter was invalid, and its context.
func (e *errInvalidParam) Message() string {
	return fmt.Sprintf("%s, %s.", e.msg, e.Field())
}

// Error returns the string version of the invalid parameter error.
func (e *errInvalidParam) Error() string {
	return fmt.Sprintf("%s: %s", e.code, e.Message())
}

// OrigErr returns nil, Implemented for awserr.Error interface.
func (e *errInvalidParam) OrigErr() error {
	return nil
}

// Field Returns the field and context the error occurred.
func (e *errInvalidParam) Field() string {
	field := e.context
	if len(field) > 0 {
		field += "."
	}
	if len(e.nestedContext) > 0 {
		field += fmt.Sprintf("%s.", e.nestedContext)
	}
	field += e.field

	return field
}

// SetContext updates the base context of the error.
func (e *errInvalidParam) SetContext(ctx string) {
	e.context = ctx
}

// AddNestedContext prepends a context to the field's path.
func (e *errInvalidParam) AddNestedContext(ctx string) {
	if len(e.nestedContext) == 0 {
		e.nestedContext = ctx
	} else {
		e.nestedContext = fmt.Sprintf("%s.%s", ctx, e.nestedContext)
	}

}

// An ErrParamRequired represents an required parameter error.
type ErrParamRequired struct {
	errInvalidParam
}

// NewErrParamRequired creates a new required parameter error.
func NewErrParamRequired(field string) *ErrParamRequired {
	return &ErrParamRequired{
		errInvalidParam{
			code:  ParamRequiredErrCode,
			field: field,
			msg:   fmt.Sprintf("missing required field"),
		},
	}
}

// An ErrParamMinValue represents a minimum value parameter error.
type ErrParamMinValue struct {
	errInvalidParam
	min float64
}

// NewErrParamMinValue creates a new minimum value parameter error.
func NewErrParamMinValue(field string, min float64) *ErrParamMinValue {
	return &ErrParamMinValue{
		errInvalidParam: errInvalidParam{
			code:  ParamMinValueErrCode,
			field: field,
			msg:   fmt.Sprintf("minimum field value of %v", min),
		},
		min: min,
	}
}

// MinValue returns the field's require minimum value.
//
// float64 is returned for both int and float min values.
func (e *ErrParamMinValue) MinValue() float64 {
	return e.min
}

// An ErrParamMinLen represents a minimum length parameter error.
type ErrParamMinLen struct {
	errInvalidParam
	min int
}

// NewErrParamMinLen creates a new minimum length parameter error.
func NewErrParamMinLen(field string, min int) *ErrParamMinLen {
	return &ErrParamMinLen{
		errInvalidParam: errInvalidParam{
			code:  ParamMinLenErrCode,
			field: field,
			msg:   fmt.Sprintf("minimum field size of %v", min),
		},
		min: min,
	}
}

// MinLen returns the field's required minimum length.
func (e *ErrParamMinLen) MinLen() int {
	return e.min
}
