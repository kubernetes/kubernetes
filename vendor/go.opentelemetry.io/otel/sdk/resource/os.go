// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import (
	"context"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.34.0"
)

type osDescriptionProvider func() (string, error)

var defaultOSDescriptionProvider osDescriptionProvider = platformOSDescription

var osDescription = defaultOSDescriptionProvider

func setDefaultOSDescriptionProvider() {
	setOSDescriptionProvider(defaultOSDescriptionProvider)
}

func setOSDescriptionProvider(osDescriptionProvider osDescriptionProvider) {
	osDescription = osDescriptionProvider
}

type (
	osTypeDetector        struct{}
	osDescriptionDetector struct{}
)

// Detect returns a *Resource that describes the operating system type the
// service is running on.
func (osTypeDetector) Detect(ctx context.Context) (*Resource, error) {
	osType := runtimeOS()

	osTypeAttribute := mapRuntimeOSToSemconvOSType(osType)

	return NewWithAttributes(
		semconv.SchemaURL,
		osTypeAttribute,
	), nil
}

// Detect returns a *Resource that describes the operating system the
// service is running on.
func (osDescriptionDetector) Detect(ctx context.Context) (*Resource, error) {
	description, err := osDescription()
	if err != nil {
		return nil, err
	}

	return NewWithAttributes(
		semconv.SchemaURL,
		semconv.OSDescription(description),
	), nil
}

// mapRuntimeOSToSemconvOSType translates the OS name as provided by the Go runtime
// into an OS type attribute with the corresponding value defined by the semantic
// conventions. In case the provided OS name isn't mapped, it's transformed to lowercase
// and used as the value for the returned OS type attribute.
func mapRuntimeOSToSemconvOSType(osType string) attribute.KeyValue {
	// the elements in this map are the intersection between
	// available GOOS values and defined semconv OS types
	osTypeAttributeMap := map[string]attribute.KeyValue{
		"aix":       semconv.OSTypeAIX,
		"darwin":    semconv.OSTypeDarwin,
		"dragonfly": semconv.OSTypeDragonflyBSD,
		"freebsd":   semconv.OSTypeFreeBSD,
		"linux":     semconv.OSTypeLinux,
		"netbsd":    semconv.OSTypeNetBSD,
		"openbsd":   semconv.OSTypeOpenBSD,
		"solaris":   semconv.OSTypeSolaris,
		"windows":   semconv.OSTypeWindows,
		"zos":       semconv.OSTypeZOS,
	}

	var osTypeAttribute attribute.KeyValue

	if attr, ok := osTypeAttributeMap[osType]; ok {
		osTypeAttribute = attr
	} else {
		osTypeAttribute = semconv.OSTypeKey.String(strings.ToLower(osType))
	}

	return osTypeAttribute
}
