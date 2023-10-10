package apivalidation

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// Follows the structure of the schema representing validation state at
// each node
type Result struct {
	Errors       []*field.Error
	Warnings     []*field.Error
	UpdateErrors []*field.Error
	Path         *field.Path
}

func (r *Result) AddErrors(errs ...error) {
	for _, e := range errs {
		// This will not catch errors that are backed by nil pointer...
		if e == nil {
			continue
		}
		r.Errors = append(r.Errors, kubeOpenAPIErrorToFieldError(r.Path, e))
	}
}

func (r *Result) IsValid() bool {
	return len(r.Errors)+len(r.UpdateErrors) == 0
}

func (r *Result) Merge(other *Result) {
	r.Errors = append(r.Errors, other.Errors...)
	r.Warnings = append(r.Warnings, other.Warnings...)
	r.UpdateErrors = append(r.UpdateErrors, other.UpdateErrors...)
}

func NewResult() *Result {
	return &Result{}
}
