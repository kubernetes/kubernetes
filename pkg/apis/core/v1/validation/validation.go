/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package validation

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

const isNegativeErrorMsg string = `must be greater than or equal to 0`

var validLogStreams = sets.New[string](
	v1.LogStreamStdout,
	v1.LogStreamStderr,
	v1.LogStreamAll,
)

// ValidatePodLogOptions checks if options that are set are at the correct
// value. Any incorrect value will be returned to the ErrorList.
func ValidatePodLogOptions(opts *v1.PodLogOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	if opts.TailLines != nil && *opts.TailLines < 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("tailLines"), *opts.TailLines, isNegativeErrorMsg))
	}
	if opts.LimitBytes != nil && *opts.LimitBytes < 1 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("limitBytes"), *opts.LimitBytes, "must be greater than 0"))
	}
	switch {
	case opts.SinceSeconds != nil && opts.SinceTime != nil:
		allErrs = append(allErrs, field.Forbidden(field.NewPath(""), "at most one of `sinceTime` or `sinceSeconds` may be specified"))
	case opts.SinceSeconds != nil:
		if *opts.SinceSeconds < 1 {
			allErrs = append(allErrs, field.Invalid(field.NewPath("sinceSeconds"), *opts.SinceSeconds, "must be greater than 0"))
		}
	}
	// opts.Stream can be nil because defaulting might not apply if no URL params are provided.
	if opts.Stream != nil {
		if !validLogStreams.Has(*opts.Stream) {
			allErrs = append(allErrs, field.NotSupported(field.NewPath("stream"), *opts.Stream, validLogStreams.UnsortedList()))
		}
		if *opts.Stream != v1.LogStreamAll && opts.TailLines != nil {
			allErrs = append(allErrs, field.Forbidden(field.NewPath(""), "`tailLines` and specific `stream` are mutually exclusive for now"))
		}
	}
	return allErrs
}
