// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Code generated from semantic convention specification. DO NOT EDIT.

package semconv // import "go.opentelemetry.io/otel/semconv/v1.25.0"

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
