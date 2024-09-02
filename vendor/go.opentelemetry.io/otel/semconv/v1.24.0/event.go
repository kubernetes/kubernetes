// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Code generated from semantic convention specification. DO NOT EDIT.

package semconv // import "go.opentelemetry.io/otel/semconv/v1.24.0"

import "go.opentelemetry.io/otel/attribute"

// This event represents an occurrence of a lifecycle transition on the iOS
// platform.
const (
	// IosStateKey is the attribute Key conforming to the "ios.state" semantic
	// conventions. It represents the this attribute represents the state the
	// application has transitioned into at the occurrence of the event.
	//
	// Type: Enum
	// RequirementLevel: Required
	// Stability: experimental
	// Note: The iOS lifecycle states are defined in the [UIApplicationDelegate
	// documentation](https://developer.apple.com/documentation/uikit/uiapplicationdelegate#1656902),
	// and from which the `OS terminology` column values are derived.
	IosStateKey = attribute.Key("ios.state")
)

var (
	// The app has become `active`. Associated with UIKit notification `applicationDidBecomeActive`
	IosStateActive = IosStateKey.String("active")
	// The app is now `inactive`. Associated with UIKit notification `applicationWillResignActive`
	IosStateInactive = IosStateKey.String("inactive")
	// The app is now in the background. This value is associated with UIKit notification `applicationDidEnterBackground`
	IosStateBackground = IosStateKey.String("background")
	// The app is now in the foreground. This value is associated with UIKit notification `applicationWillEnterForeground`
	IosStateForeground = IosStateKey.String("foreground")
	// The app is about to terminate. Associated with UIKit notification `applicationWillTerminate`
	IosStateTerminate = IosStateKey.String("terminate")
)

// This event represents an occurrence of a lifecycle transition on the Android
// platform.
const (
	// AndroidStateKey is the attribute Key conforming to the "android.state"
	// semantic conventions. It represents the this attribute represents the
	// state the application has transitioned into at the occurrence of the
	// event.
	//
	// Type: Enum
	// RequirementLevel: Required
	// Stability: experimental
	// Note: The Android lifecycle states are defined in [Activity lifecycle
	// callbacks](https://developer.android.com/guide/components/activities/activity-lifecycle#lc),
	// and from which the `OS identifiers` are derived.
	AndroidStateKey = attribute.Key("android.state")
)

var (
	// Any time before Activity.onResume() or, if the app has no Activity, Context.startService() has been called in the app for the first time
	AndroidStateCreated = AndroidStateKey.String("created")
	// Any time after Activity.onPause() or, if the app has no Activity, Context.stopService() has been called when the app was in the foreground state
	AndroidStateBackground = AndroidStateKey.String("background")
	// Any time after Activity.onResume() or, if the app has no Activity, Context.startService() has been called when the app was in either the created or background states
	AndroidStateForeground = AndroidStateKey.String("foreground")
)

// This semantic convention defines the attributes used to represent a feature
// flag evaluation as an event.
const (
	// FeatureFlagKeyKey is the attribute Key conforming to the
	// "feature_flag.key" semantic conventions. It represents the unique
	// identifier of the feature flag.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: experimental
	// Examples: 'logo-color'
	FeatureFlagKeyKey = attribute.Key("feature_flag.key")

	// FeatureFlagProviderNameKey is the attribute Key conforming to the
	// "feature_flag.provider_name" semantic conventions. It represents the
	// name of the service provider that performs the flag evaluation.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: experimental
	// Examples: 'Flag Manager'
	FeatureFlagProviderNameKey = attribute.Key("feature_flag.provider_name")

	// FeatureFlagVariantKey is the attribute Key conforming to the
	// "feature_flag.variant" semantic conventions. It represents the sHOULD be
	// a semantic identifier for a value. If one is unavailable, a stringified
	// version of the value can be used.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: experimental
	// Examples: 'red', 'true', 'on'
	// Note: A semantic identifier, commonly referred to as a variant, provides
	// a means
	// for referring to a value without including the value itself. This can
	// provide additional context for understanding the meaning behind a value.
	// For example, the variant `red` maybe be used for the value `#c05543`.
	//
	// A stringified version of the value can be used in situations where a
	// semantic identifier is unavailable. String representation of the value
	// should be determined by the implementer.
	FeatureFlagVariantKey = attribute.Key("feature_flag.variant")
)

// FeatureFlagKey returns an attribute KeyValue conforming to the
// "feature_flag.key" semantic conventions. It represents the unique identifier
// of the feature flag.
func FeatureFlagKey(val string) attribute.KeyValue {
	return FeatureFlagKeyKey.String(val)
}

// FeatureFlagProviderName returns an attribute KeyValue conforming to the
// "feature_flag.provider_name" semantic conventions. It represents the name of
// the service provider that performs the flag evaluation.
func FeatureFlagProviderName(val string) attribute.KeyValue {
	return FeatureFlagProviderNameKey.String(val)
}

// FeatureFlagVariant returns an attribute KeyValue conforming to the
// "feature_flag.variant" semantic conventions. It represents the sHOULD be a
// semantic identifier for a value. If one is unavailable, a stringified
// version of the value can be used.
func FeatureFlagVariant(val string) attribute.KeyValue {
	return FeatureFlagVariantKey.String(val)
}

// RPC received/sent message.
const (
	// MessageCompressedSizeKey is the attribute Key conforming to the
	// "message.compressed_size" semantic conventions. It represents the
	// compressed size of the message in bytes.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	MessageCompressedSizeKey = attribute.Key("message.compressed_size")

	// MessageIDKey is the attribute Key conforming to the "message.id"
	// semantic conventions. It represents the mUST be calculated as two
	// different counters starting from `1` one for sent messages and one for
	// received message.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	// Note: This way we guarantee that the values will be consistent between
	// different implementations.
	MessageIDKey = attribute.Key("message.id")

	// MessageTypeKey is the attribute Key conforming to the "message.type"
	// semantic conventions. It represents the whether this is a received or
	// sent message.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	MessageTypeKey = attribute.Key("message.type")

	// MessageUncompressedSizeKey is the attribute Key conforming to the
	// "message.uncompressed_size" semantic conventions. It represents the
	// uncompressed size of the message in bytes.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: experimental
	MessageUncompressedSizeKey = attribute.Key("message.uncompressed_size")
)

var (
	// sent
	MessageTypeSent = MessageTypeKey.String("SENT")
	// received
	MessageTypeReceived = MessageTypeKey.String("RECEIVED")
)

// MessageCompressedSize returns an attribute KeyValue conforming to the
// "message.compressed_size" semantic conventions. It represents the compressed
// size of the message in bytes.
func MessageCompressedSize(val int) attribute.KeyValue {
	return MessageCompressedSizeKey.Int(val)
}

// MessageID returns an attribute KeyValue conforming to the "message.id"
// semantic conventions. It represents the mUST be calculated as two different
// counters starting from `1` one for sent messages and one for received
// message.
func MessageID(val int) attribute.KeyValue {
	return MessageIDKey.Int(val)
}

// MessageUncompressedSize returns an attribute KeyValue conforming to the
// "message.uncompressed_size" semantic conventions. It represents the
// uncompressed size of the message in bytes.
func MessageUncompressedSize(val int) attribute.KeyValue {
	return MessageUncompressedSizeKey.Int(val)
}
