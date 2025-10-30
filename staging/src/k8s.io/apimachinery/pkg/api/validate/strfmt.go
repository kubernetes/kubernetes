/*
Copyright 2025 The Kubernetes Authors.

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

package validate

import (
	"context"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/api/validate/content"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

const (
	uuidErrorMessage              = "must be a lowercase UUID in 8-4-4-4-12 format"
	defaultResourceRequestsPrefix = "requests."
	// Default namespace prefix.
	resourceDefaultNamespacePrefix = "kubernetes.io/"
	resourceDeviceMaxLength        = 32
)

// ShortName verifies that the specified value is a valid "short name"
// (sometimes known as a "DNS label").
//   - must not be empty
//   - must be less than 64 characters long
//   - must start and end with lower-case alphanumeric characters
//   - must contain only lower-case alphanumeric characters or dashes
//
// All errors returned by this function will be "invalid" type errors. If the
// caller wants better errors, it must take responsibility for checking things
// like required/optional and max-length.
func ShortName[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	var allErrs field.ErrorList
	for _, msg := range content.IsDNS1123Label((string)(*value)) {
		allErrs = append(allErrs, field.Invalid(fldPath, *value, msg).WithOrigin("format=k8s-short-name"))
	}
	return allErrs
}

// LongName verifies that the specified value is a valid "long name"
// (sometimes known as a "DNS subdomain").
//   - must not be empty
//   - must be less than 254 characters long
//   - each element must start and end with lower-case alphanumeric characters
//   - each element must contain only lower-case alphanumeric characters or dashes
//
// All errors returned by this function will be "invalid" type errors. If the
// caller wants better errors, it must take responsibility for checking things
// like required/optional and max-length.
func LongName[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	var allErrs field.ErrorList
	for _, msg := range content.IsDNS1123Subdomain((string)(*value)) {
		allErrs = append(allErrs, field.Invalid(fldPath, *value, msg).WithOrigin("format=k8s-long-name"))
	}
	return allErrs
}

// LabelKey verifies that the specified value is a valid label key.
// A label key is composed of an optional prefix and a name, separated by a '/'.
// The name part is required and must:
//   - be 63 characters or less
//   - begin and end with an alphanumeric character ([a-z0-9A-Z])
//   - contain only alphanumeric characters, dashes (-), underscores (_), or dots (.)
//
// The prefix is optional and must:
//   - be a DNS subdomain
//   - be no more than 253 characters
func LabelKey[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	var allErrs field.ErrorList
	for _, msg := range content.IsLabelKey((string)(*value)) {
		allErrs = append(allErrs, field.Invalid(fldPath, *value, msg).WithOrigin("format=k8s-label-key"))
	}
	return allErrs
}

// LongNameCaseless verifies that the specified value is a valid "long name"
// (sometimes known as a "DNS subdomain"), but is case-insensitive.
//   - must not be empty
//   - must be less than 254 characters long
//   - each element must start and end with alphanumeric characters
//   - each element must contain only alphanumeric characters or dashes
//
// Deprecated: Case-insensitive names are not recommended as they can lead to ambiguity
// (e.g., 'Foo', 'FOO', and 'foo' would be allowed names for foo). Use LongName for strict, lowercase validation.
func LongNameCaseless[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	var allErrs field.ErrorList
	for _, msg := range content.IsDNS1123SubdomainCaseless((string)(*value)) {
		allErrs = append(allErrs, field.Invalid(fldPath, *value, msg).WithOrigin("format=k8s-long-name-caseless"))
	}
	return allErrs
}

// LabelValue verifies that the specified value is a valid label value.
//   - can be empty
//   - must be no more than 63 characters
//   - must start and end with alphanumeric characters
//   - must contain only alphanumeric characters, dashes, underscores, or dots
func LabelValue[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	var allErrs field.ErrorList
	for _, msg := range content.IsLabelValue((string)(*value)) {
		allErrs = append(allErrs, field.Invalid(fldPath, *value, msg).WithOrigin("format=k8s-label-value"))
	}
	return allErrs
}

// UUID verifies that the specified value is a valid UUID (RFC 4122).
//   - must be 36 characters long
//   - must be in the normalized form `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
//   - must use only lowercase hexadecimal characters
func UUID[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	val := (string)(*value)
	if len(val) != 36 {
		return field.ErrorList{field.Invalid(fldPath, val, uuidErrorMessage).WithOrigin("format=k8s-uuid")}
	}
	for idx := 0; idx < len(val); idx++ {
		character := val[idx]
		switch idx {
		case 8, 13, 18, 23:
			if character != '-' {
				return field.ErrorList{field.Invalid(fldPath, val, uuidErrorMessage).WithOrigin("format=k8s-uuid")}
			}
		default:
			// should be lower case hexadecimal.
			if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
				return field.ErrorList{field.Invalid(fldPath, val, uuidErrorMessage).WithOrigin("format=k8s-uuid")}
			}
		}
	}
	return nil
}

// ResourcePoolName verifies that the specified value is one or more valid "long name"
// parts separated by a '/' and no longer than 253 characters.
func ResourcePoolName[T ~string](ctx context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	val := (string)(*value)
	var allErrs field.ErrorList
	if len(val) > 253 {
		allErrs = append(allErrs, field.TooLong(fldPath, val, 253))
	}
	parts := strings.Split(val, "/")
	for i, part := range parts {
		if len(part) == 0 {
			allErrs = append(allErrs, field.Invalid(fldPath, val, fmt.Sprintf("segment %d: must not be empty", i)))
			continue
		}
		// Note that we are overwriting the origin from the underlying LongName validation.
		allErrs = append(allErrs, LongName(ctx, op, fldPath, &part, nil).PrefixDetail(fmt.Sprintf("segment %d: ", i))...)
	}
	return allErrs.WithOrigin("format=k8s-resource-pool-name")
}

// ExtendedResourceName verifies that the specified value is a valid extended resource name.
// An extended resource name is a domain-prefixed name that does not use the "kubernetes.io"
// or "requests." prefixes. Must be a valid label key when appended to "requests.", as in quota.
//
//   - must have slash domain and name.
//   - must not have the "kubernetes.io" domain
//   - must not have the "requests." prefix
//   - name must be 63 characters or less
//   - must be a valid label key when appended to "requests.", as in quota
//     -- must contain only alphanumeric characters, dashes, underscores, or dots
//     -- must end with an alphanumeric character
func ExtendedResourceName[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	val := string(*value)
	allErrs := field.ErrorList{}
	if !strings.Contains(val, "/") {
		allErrs = append(allErrs, field.Invalid(fldPath, val, "a name must be a domain-prefixed path, such as 'example.com/my-prop'"))
	} else if strings.Contains(val, resourceDefaultNamespacePrefix) {
		allErrs = append(allErrs, field.Invalid(fldPath, val, fmt.Sprintf("must not have %q domain", resourceDefaultNamespacePrefix)))
	}
	// Ensure extended resource is not type of quota.
	if strings.HasPrefix(val, defaultResourceRequestsPrefix) {
		allErrs = append(allErrs, field.Invalid(fldPath, val, fmt.Sprintf("must not have %q prefix", defaultResourceRequestsPrefix)))
	}

	// Ensure it satisfies the rules in IsLabelKey() after converted into quota resource name
	nameForQuota := fmt.Sprintf("%s%s", defaultResourceRequestsPrefix, val)
	for _, msg := range content.IsLabelKey(nameForQuota) {
		allErrs = append(allErrs, field.Invalid(fldPath, val, msg))
	}
	return allErrs.WithOrigin("format=k8s-extended-resource-name")
}

// resourcesQualifiedName verifies that the specified value is a valid Kubernetes resources
// qualified name.
//   - must not be empty
//   - must be composed of an optional prefix and a name, separated by a slash (e.g., "prefix/name")
//   - the prefix, if specified, must be a DNS subdomain
//   - the name part must be a C identifier
//   - the name part must be no more than 32 characters
func resourcesQualifiedName[T ~string](ctx context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	var allErrs field.ErrorList
	s := string(*value)
	parts := strings.Split(s, "/")
	// TODO: This validation and the corresponding handwritten validation validateQualifiedName in
	// pkg/apis/resource/validation/validation.go are not validating whether there are more than 1
	// slash. This should be fixed in both places.
	switch len(parts) {
	case 1:
		allErrs = append(allErrs, validateCIdentifier(parts[0], resourceDeviceMaxLength, fldPath)...)
	case 2:
		if len(parts[0]) == 0 {
			allErrs = append(allErrs, field.Invalid(fldPath, "", "prefix must not be empty"))
		} else {
			if len(parts[0]) > 63 {
				allErrs = append(allErrs, field.TooLong(fldPath, parts[0], 63))
			}
			allErrs = append(allErrs, LongName(ctx, op, fldPath, &parts[0], nil).PrefixDetail("prefix: ")...)
		}
		if len(parts[1]) == 0 {
			allErrs = append(allErrs, field.Invalid(fldPath, "", "name must not be empty"))
		} else {
			allErrs = append(allErrs, validateCIdentifier(parts[1], resourceDeviceMaxLength, fldPath)...)
		}
	}
	return allErrs
}

// ResourceFullyQualifiedName verifies that the specified value is a valid Kubernetes
// fully qualified name.
//   - must not be empty
//   - must be composed of a prefix and a name, separated by a slash (e.g., "prefix/name")
//   - the prefix must be a DNS subdomain
//   - the name part must be a C identifier
//   - the name part must be no more than 32 characters
func ResourceFullyQualifiedName[T ~string](ctx context.Context, op operation.Operation, fldPath *field.Path, value, _ *T) field.ErrorList {
	if value == nil {
		return nil
	}
	var allErrs field.ErrorList
	s := string(*value)
	allErrs = append(allErrs, resourcesQualifiedName(ctx, op, fldPath, &s, nil)...)
	if !strings.Contains(s, "/") {
		allErrs = append(allErrs, field.Invalid(fldPath, s, "a fully qualified name must be a domain and a name separated by a slash"))
	}
	return allErrs.WithOrigin("format=k8s-resource-fully-qualified-name")
}

func validateCIdentifier(id string, length int, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if len(id) > length {
		allErrs = append(allErrs, field.TooLong(fldPath, id, length))
	}
	for _, msg := range content.IsCIdentifier(id) {
		allErrs = append(allErrs, field.Invalid(fldPath, id, msg))
	}
	return allErrs
}
