// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"strings"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.37.0"
)

const (
	// resourceAttrKey is the environment variable name OpenTelemetry Resource information will be read from.
	resourceAttrKey = "OTEL_RESOURCE_ATTRIBUTES" //nolint:gosec // False positive G101: Potential hardcoded credentials

	// svcNameKey is the environment variable name that Service Name information will be read from.
	svcNameKey = "OTEL_SERVICE_NAME"
)

// errMissingValue is returned when a resource value is missing.
var errMissingValue = fmt.Errorf("%w: missing value", ErrPartialResource)

// fromEnv is a Detector that implements the Detector and collects
// resources from environment.  This Detector is included as a
// builtin.
type fromEnv struct{}

// compile time assertion that FromEnv implements Detector interface.
var _ Detector = fromEnv{}

// Detect collects resources from environment.
func (fromEnv) Detect(context.Context) (*Resource, error) {
	attrs := strings.TrimSpace(os.Getenv(resourceAttrKey))
	svcName := strings.TrimSpace(os.Getenv(svcNameKey))

	if attrs == "" && svcName == "" {
		return Empty(), nil
	}

	var res *Resource

	if svcName != "" {
		res = NewSchemaless(semconv.ServiceName(svcName))
	}

	r2, err := constructOTResources(attrs)

	// Ensure that the resource with the service name from OTEL_SERVICE_NAME
	// takes precedence, if it was defined.
	res, err2 := Merge(r2, res)

	if err == nil {
		err = err2
	} else if err2 != nil {
		err = fmt.Errorf("detecting resources: %s", []string{err.Error(), err2.Error()})
	}

	return res, err
}

func constructOTResources(s string) (*Resource, error) {
	if s == "" {
		return Empty(), nil
	}
	pairs := strings.Split(s, ",")
	var attrs []attribute.KeyValue
	var invalid []string
	for _, p := range pairs {
		k, v, found := strings.Cut(p, "=")
		if !found {
			invalid = append(invalid, p)
			continue
		}
		key := strings.TrimSpace(k)
		val, err := url.PathUnescape(strings.TrimSpace(v))
		if err != nil {
			// Retain original value if decoding fails, otherwise it will be
			// an empty string.
			val = v
			otel.Handle(err)
		}
		attrs = append(attrs, attribute.String(key, val))
	}
	var err error
	if len(invalid) > 0 {
		err = fmt.Errorf("%w: %v", errMissingValue, invalid)
	}
	return NewSchemaless(attrs...), err
}
