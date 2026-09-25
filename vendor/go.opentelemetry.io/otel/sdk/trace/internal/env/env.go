// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package env provides types and functionality for environment variable support
// in the OpenTelemetry SDK.
package env // import "go.opentelemetry.io/otel/sdk/trace/internal/env"

import (
	"os"
	"strconv"

	"go.opentelemetry.io/otel/internal/global"
)

// Environment variable names.
const (
	// BatchSpanProcessorScheduleDelayKey is the delay interval between two
	// consecutive exports (i.e. 5000).
	BatchSpanProcessorScheduleDelayKey = "OTEL_BSP_SCHEDULE_DELAY"
	// BatchSpanProcessorExportTimeoutKey is the maximum allowed time to
	// export data (i.e. 3000).
	BatchSpanProcessorExportTimeoutKey = "OTEL_BSP_EXPORT_TIMEOUT"
	// BatchSpanProcessorMaxQueueSizeKey is the maximum queue size (i.e. 2048).
	BatchSpanProcessorMaxQueueSizeKey = "OTEL_BSP_MAX_QUEUE_SIZE"
	// BatchSpanProcessorMaxExportBatchSizeKey is the maximum batch size (i.e.
	// 512). Note: it must be less than or equal to
	// BatchSpanProcessorMaxQueueSize.
	BatchSpanProcessorMaxExportBatchSizeKey = "OTEL_BSP_MAX_EXPORT_BATCH_SIZE"

	// AttributeValueLengthKey is the maximum allowed attribute value size.
	AttributeValueLengthKey = "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT"

	// AttributeCountKey is the maximum allowed span attribute count.
	AttributeCountKey = "OTEL_ATTRIBUTE_COUNT_LIMIT"

	// SpanAttributeValueLengthKey is the maximum allowed attribute value size
	// for a span.
	SpanAttributeValueLengthKey = "OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT"

	// SpanAttributeCountKey is the maximum allowed span attribute count for a
	// span.
	SpanAttributeCountKey = "OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT"

	// SpanEventCountKey is the maximum allowed span event count.
	SpanEventCountKey = "OTEL_SPAN_EVENT_COUNT_LIMIT"

	// SpanEventAttributeCountKey is the maximum allowed attribute per span
	// event count.
	SpanEventAttributeCountKey = "OTEL_EVENT_ATTRIBUTE_COUNT_LIMIT"

	// SpanLinkCountKey is the maximum allowed span link count.
	SpanLinkCountKey = "OTEL_SPAN_LINK_COUNT_LIMIT"

	// SpanLinkAttributeCountKey is the maximum allowed attribute per span
	// link count.
	SpanLinkAttributeCountKey = "OTEL_LINK_ATTRIBUTE_COUNT_LIMIT"
)

// firstInt returns the value of the first matching environment variable from
// keys. If the value is not an integer or no match is found, defaultValue is
// returned.
func firstInt(defaultValue int, keys ...string) int {
	for _, key := range keys {
		value := os.Getenv(key)
		if value == "" {
			continue
		}

		intValue, err := strconv.Atoi(value)
		if err != nil {
			global.Info("Got invalid value, number value expected.", key, value)
			return defaultValue
		}

		return intValue
	}

	return defaultValue
}

// IntEnvOr returns the int value of the environment variable with name key if
// it exists, it is not empty, and the value is an int. Otherwise, defaultValue is returned.
func IntEnvOr(key string, defaultValue int) int {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}

	intValue, err := strconv.Atoi(value)
	if err != nil {
		global.Info("Got invalid value, number value expected.", key, value)
		return defaultValue
	}

	return intValue
}

// BatchSpanProcessorScheduleDelay returns the environment variable value for
// the OTEL_BSP_SCHEDULE_DELAY key if it exists, otherwise defaultValue is
// returned.
func BatchSpanProcessorScheduleDelay(defaultValue int) int {
	return IntEnvOr(BatchSpanProcessorScheduleDelayKey, defaultValue)
}

// BatchSpanProcessorExportTimeout returns the environment variable value for
// the OTEL_BSP_EXPORT_TIMEOUT key if it exists, otherwise defaultValue is
// returned.
func BatchSpanProcessorExportTimeout(defaultValue int) int {
	return IntEnvOr(BatchSpanProcessorExportTimeoutKey, defaultValue)
}

// BatchSpanProcessorMaxQueueSize returns the environment variable value for
// the OTEL_BSP_MAX_QUEUE_SIZE key if it exists, otherwise defaultValue is
// returned.
func BatchSpanProcessorMaxQueueSize(defaultValue int) int {
	return IntEnvOr(BatchSpanProcessorMaxQueueSizeKey, defaultValue)
}

// BatchSpanProcessorMaxExportBatchSize returns the environment variable value for
// the OTEL_BSP_MAX_EXPORT_BATCH_SIZE key if it exists, otherwise defaultValue
// is returned.
func BatchSpanProcessorMaxExportBatchSize(defaultValue int) int {
	return IntEnvOr(BatchSpanProcessorMaxExportBatchSizeKey, defaultValue)
}

// SpanAttributeValueLength returns the environment variable value for the
// OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT key if it exists. Otherwise, the
// environment variable value for OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT is
// returned or defaultValue if that is not set.
func SpanAttributeValueLength(defaultValue int) int {
	return firstInt(defaultValue, SpanAttributeValueLengthKey, AttributeValueLengthKey)
}

// SpanAttributeCount returns the environment variable value for the
// OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT key if it exists. Otherwise, the
// environment variable value for OTEL_ATTRIBUTE_COUNT_LIMIT is returned or
// defaultValue if that is not set.
func SpanAttributeCount(defaultValue int) int {
	return firstInt(defaultValue, SpanAttributeCountKey, AttributeCountKey)
}

// SpanEventCount returns the environment variable value for the
// OTEL_SPAN_EVENT_COUNT_LIMIT key if it exists, otherwise defaultValue is
// returned.
func SpanEventCount(defaultValue int) int {
	return IntEnvOr(SpanEventCountKey, defaultValue)
}

// SpanEventAttributeCount returns the environment variable value for the
// OTEL_EVENT_ATTRIBUTE_COUNT_LIMIT key if it exists, otherwise defaultValue
// is returned.
func SpanEventAttributeCount(defaultValue int) int {
	return IntEnvOr(SpanEventAttributeCountKey, defaultValue)
}

// SpanLinkCount returns the environment variable value for the
// OTEL_SPAN_LINK_COUNT_LIMIT key if it exists, otherwise defaultValue is
// returned.
func SpanLinkCount(defaultValue int) int {
	return IntEnvOr(SpanLinkCountKey, defaultValue)
}

// SpanLinkAttributeCount returns the environment variable value for the
// OTEL_LINK_ATTRIBUTE_COUNT_LIMIT key if it exists, otherwise defaultValue is
// returned.
func SpanLinkAttributeCount(defaultValue int) int {
	return IntEnvOr(SpanLinkAttributeCountKey, defaultValue)
}
