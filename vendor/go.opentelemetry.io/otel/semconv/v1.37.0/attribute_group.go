// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Code generated from semantic convention specification. DO NOT EDIT.

package semconv // import "go.opentelemetry.io/otel/semconv/v1.37.0"

import "go.opentelemetry.io/otel/attribute"

// Namespace: android
const (
	// AndroidAppStateKey is the attribute Key conforming to the "android.app.state"
	// semantic conventions. It represents the this attribute represents the state
	// of the application.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "created"
	// Note: The Android lifecycle states are defined in
	// [Activity lifecycle callbacks], and from which the `OS identifiers` are
	// derived.
	//
	// [Activity lifecycle callbacks]: https://developer.android.com/guide/components/activities/activity-lifecycle#lc
	AndroidAppStateKey = attribute.Key("android.app.state")

	// AndroidOSAPILevelKey is the attribute Key conforming to the
	// "android.os.api_level" semantic conventions. It represents the uniquely
	// identifies the framework API revision offered by a version (`os.version`) of
	// the android operating system. More information can be found in the
	// [Android API levels documentation].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "33", "32"
	//
	// [Android API levels documentation]: https://developer.android.com/guide/topics/manifest/uses-sdk-element#ApiLevels
	AndroidOSAPILevelKey = attribute.Key("android.os.api_level")
)

// AndroidOSAPILevel returns an attribute KeyValue conforming to the
// "android.os.api_level" semantic conventions. It represents the uniquely
// identifies the framework API revision offered by a version (`os.version`) of
// the android operating system. More information can be found in the
// [Android API levels documentation].
//
// [Android API levels documentation]: https://developer.android.com/guide/topics/manifest/uses-sdk-element#ApiLevels
func AndroidOSAPILevel(val string) attribute.KeyValue {
	return AndroidOSAPILevelKey.String(val)
}

// Enum values for android.app.state
var (
	// Any time before Activity.onResume() or, if the app has no Activity,
	// Context.startService() has been called in the app for the first time.
	//
	// Stability: development
	AndroidAppStateCreated = AndroidAppStateKey.String("created")
	// Any time after Activity.onPause() or, if the app has no Activity,
	// Context.stopService() has been called when the app was in the foreground
	// state.
	//
	// Stability: development
	AndroidAppStateBackground = AndroidAppStateKey.String("background")
	// Any time after Activity.onResume() or, if the app has no Activity,
	// Context.startService() has been called when the app was in either the created
	// or background states.
	//
	// Stability: development
	AndroidAppStateForeground = AndroidAppStateKey.String("foreground")
)

// Namespace: app
const (
	// AppBuildIDKey is the attribute Key conforming to the "app.build_id" semantic
	// conventions. It represents the unique identifier for a particular build or
	// compilation of the application.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "6cff0a7e-cefc-4668-96f5-1273d8b334d0",
	// "9f2b833506aa6973a92fde9733e6271f", "my-app-1.0.0-code-123"
	AppBuildIDKey = attribute.Key("app.build_id")

	// AppInstallationIDKey is the attribute Key conforming to the
	// "app.installation.id" semantic conventions. It represents a unique identifier
	// representing the installation of an application on a specific device.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2ab2916d-a51f-4ac8-80ee-45ac31a28092"
	// Note: Its value SHOULD persist across launches of the same application
	// installation, including through application upgrades.
	// It SHOULD change if the application is uninstalled or if all applications of
	// the vendor are uninstalled.
	// Additionally, users might be able to reset this value (e.g. by clearing
	// application data).
	// If an app is installed multiple times on the same device (e.g. in different
	// accounts on Android), each `app.installation.id` SHOULD have a different
	// value.
	// If multiple OpenTelemetry SDKs are used within the same application, they
	// SHOULD use the same value for `app.installation.id`.
	// Hardware IDs (e.g. serial number, IMEI, MAC address) MUST NOT be used as the
	// `app.installation.id`.
	//
	// For iOS, this value SHOULD be equal to the [vendor identifier].
	//
	// For Android, examples of `app.installation.id` implementations include:
	//
	//   - [Firebase Installation ID].
	//   - A globally unique UUID which is persisted across sessions in your
	//     application.
	//   - [App set ID].
	//   - [`Settings.getString(Settings.Secure.ANDROID_ID)`].
	//
	// More information about Android identifier best practices can be found in the
	// [Android user data IDs guide].
	//
	// [vendor identifier]: https://developer.apple.com/documentation/uikit/uidevice/identifierforvendor
	// [Firebase Installation ID]: https://firebase.google.com/docs/projects/manage-installations
	// [App set ID]: https://developer.android.com/identity/app-set-id
	// [`Settings.getString(Settings.Secure.ANDROID_ID)`]: https://developer.android.com/reference/android/provider/Settings.Secure#ANDROID_ID
	// [Android user data IDs guide]: https://developer.android.com/training/articles/user-data-ids
	AppInstallationIDKey = attribute.Key("app.installation.id")

	// AppJankFrameCountKey is the attribute Key conforming to the
	// "app.jank.frame_count" semantic conventions. It represents a number of frame
	// renders that experienced jank.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 9, 42
	// Note: Depending on platform limitations, the value provided MAY be
	// approximation.
	AppJankFrameCountKey = attribute.Key("app.jank.frame_count")

	// AppJankPeriodKey is the attribute Key conforming to the "app.jank.period"
	// semantic conventions. It represents the time period, in seconds, for which
	// this jank is being reported.
	//
	// Type: double
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1.0, 5.0, 10.24
	AppJankPeriodKey = attribute.Key("app.jank.period")

	// AppJankThresholdKey is the attribute Key conforming to the
	// "app.jank.threshold" semantic conventions. It represents the minimum
	// rendering threshold for this jank, in seconds.
	//
	// Type: double
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 0.016, 0.7, 1.024
	AppJankThresholdKey = attribute.Key("app.jank.threshold")

	// AppScreenCoordinateXKey is the attribute Key conforming to the
	// "app.screen.coordinate.x" semantic conventions. It represents the x
	// (horizontal) coordinate of a screen coordinate, in screen pixels.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 0, 131
	AppScreenCoordinateXKey = attribute.Key("app.screen.coordinate.x")

	// AppScreenCoordinateYKey is the attribute Key conforming to the
	// "app.screen.coordinate.y" semantic conventions. It represents the y
	// (vertical) component of a screen coordinate, in screen pixels.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 12, 99
	AppScreenCoordinateYKey = attribute.Key("app.screen.coordinate.y")

	// AppWidgetIDKey is the attribute Key conforming to the "app.widget.id"
	// semantic conventions. It represents an identifier that uniquely
	// differentiates this widget from other widgets in the same application.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "f9bc787d-ff05-48ad-90e1-fca1d46130b3", "submit_order_1829"
	// Note: A widget is an application component, typically an on-screen visual GUI
	// element.
	AppWidgetIDKey = attribute.Key("app.widget.id")

	// AppWidgetNameKey is the attribute Key conforming to the "app.widget.name"
	// semantic conventions. It represents the name of an application widget.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "submit", "attack", "Clear Cart"
	// Note: A widget is an application component, typically an on-screen visual GUI
	// element.
	AppWidgetNameKey = attribute.Key("app.widget.name")
)

// AppBuildID returns an attribute KeyValue conforming to the "app.build_id"
// semantic conventions. It represents the unique identifier for a particular
// build or compilation of the application.
func AppBuildID(val string) attribute.KeyValue {
	return AppBuildIDKey.String(val)
}

// AppInstallationID returns an attribute KeyValue conforming to the
// "app.installation.id" semantic conventions. It represents a unique identifier
// representing the installation of an application on a specific device.
func AppInstallationID(val string) attribute.KeyValue {
	return AppInstallationIDKey.String(val)
}

// AppJankFrameCount returns an attribute KeyValue conforming to the
// "app.jank.frame_count" semantic conventions. It represents a number of frame
// renders that experienced jank.
func AppJankFrameCount(val int) attribute.KeyValue {
	return AppJankFrameCountKey.Int(val)
}

// AppJankPeriod returns an attribute KeyValue conforming to the
// "app.jank.period" semantic conventions. It represents the time period, in
// seconds, for which this jank is being reported.
func AppJankPeriod(val float64) attribute.KeyValue {
	return AppJankPeriodKey.Float64(val)
}

// AppJankThreshold returns an attribute KeyValue conforming to the
// "app.jank.threshold" semantic conventions. It represents the minimum rendering
// threshold for this jank, in seconds.
func AppJankThreshold(val float64) attribute.KeyValue {
	return AppJankThresholdKey.Float64(val)
}

// AppScreenCoordinateX returns an attribute KeyValue conforming to the
// "app.screen.coordinate.x" semantic conventions. It represents the x
// (horizontal) coordinate of a screen coordinate, in screen pixels.
func AppScreenCoordinateX(val int) attribute.KeyValue {
	return AppScreenCoordinateXKey.Int(val)
}

// AppScreenCoordinateY returns an attribute KeyValue conforming to the
// "app.screen.coordinate.y" semantic conventions. It represents the y (vertical)
// component of a screen coordinate, in screen pixels.
func AppScreenCoordinateY(val int) attribute.KeyValue {
	return AppScreenCoordinateYKey.Int(val)
}

// AppWidgetID returns an attribute KeyValue conforming to the "app.widget.id"
// semantic conventions. It represents an identifier that uniquely differentiates
// this widget from other widgets in the same application.
func AppWidgetID(val string) attribute.KeyValue {
	return AppWidgetIDKey.String(val)
}

// AppWidgetName returns an attribute KeyValue conforming to the
// "app.widget.name" semantic conventions. It represents the name of an
// application widget.
func AppWidgetName(val string) attribute.KeyValue {
	return AppWidgetNameKey.String(val)
}

// Namespace: artifact
const (
	// ArtifactAttestationFilenameKey is the attribute Key conforming to the
	// "artifact.attestation.filename" semantic conventions. It represents the
	// provenance filename of the built attestation which directly relates to the
	// build artifact filename. This filename SHOULD accompany the artifact at
	// publish time. See the [SLSA Relationship] specification for more information.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "golang-binary-amd64-v0.1.0.attestation",
	// "docker-image-amd64-v0.1.0.intoto.json1", "release-1.tar.gz.attestation",
	// "file-name-package.tar.gz.intoto.json1"
	//
	// [SLSA Relationship]: https://slsa.dev/spec/v1.0/distributing-provenance#relationship-between-artifacts-and-attestations
	ArtifactAttestationFilenameKey = attribute.Key("artifact.attestation.filename")

	// ArtifactAttestationHashKey is the attribute Key conforming to the
	// "artifact.attestation.hash" semantic conventions. It represents the full
	// [hash value (see glossary)], of the built attestation. Some envelopes in the
	// [software attestation space] also refer to this as the **digest**.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "1b31dfcd5b7f9267bf2ff47651df1cfb9147b9e4df1f335accf65b4cda498408"
	//
	// [hash value (see glossary)]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-5.pdf
	// [software attestation space]: https://github.com/in-toto/attestation/tree/main/spec
	ArtifactAttestationHashKey = attribute.Key("artifact.attestation.hash")

	// ArtifactAttestationIDKey is the attribute Key conforming to the
	// "artifact.attestation.id" semantic conventions. It represents the id of the
	// build [software attestation].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "123"
	//
	// [software attestation]: https://slsa.dev/attestation-model
	ArtifactAttestationIDKey = attribute.Key("artifact.attestation.id")

	// ArtifactFilenameKey is the attribute Key conforming to the
	// "artifact.filename" semantic conventions. It represents the human readable
	// file name of the artifact, typically generated during build and release
	// processes. Often includes the package name and version in the file name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "golang-binary-amd64-v0.1.0", "docker-image-amd64-v0.1.0",
	// "release-1.tar.gz", "file-name-package.tar.gz"
	// Note: This file name can also act as the [Package Name]
	// in cases where the package ecosystem maps accordingly.
	// Additionally, the artifact [can be published]
	// for others, but that is not a guarantee.
	//
	// [Package Name]: https://slsa.dev/spec/v1.0/terminology#package-model
	// [can be published]: https://slsa.dev/spec/v1.0/terminology#software-supply-chain
	ArtifactFilenameKey = attribute.Key("artifact.filename")

	// ArtifactHashKey is the attribute Key conforming to the "artifact.hash"
	// semantic conventions. It represents the full [hash value (see glossary)],
	// often found in checksum.txt on a release of the artifact and used to verify
	// package integrity.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "9ff4c52759e2c4ac70b7d517bc7fcdc1cda631ca0045271ddd1b192544f8a3e9"
	// Note: The specific algorithm used to create the cryptographic hash value is
	// not defined. In situations where an artifact has multiple
	// cryptographic hashes, it is up to the implementer to choose which
	// hash value to set here; this should be the most secure hash algorithm
	// that is suitable for the situation and consistent with the
	// corresponding attestation. The implementer can then provide the other
	// hash values through an additional set of attribute extensions as they
	// deem necessary.
	//
	// [hash value (see glossary)]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-5.pdf
	ArtifactHashKey = attribute.Key("artifact.hash")

	// ArtifactPurlKey is the attribute Key conforming to the "artifact.purl"
	// semantic conventions. It represents the [Package URL] of the
	// [package artifact] provides a standard way to identify and locate the
	// packaged artifact.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "pkg:github/package-url/purl-spec@1209109710924",
	// "pkg:npm/foo@12.12.3"
	//
	// [Package URL]: https://github.com/package-url/purl-spec
	// [package artifact]: https://slsa.dev/spec/v1.0/terminology#package-model
	ArtifactPurlKey = attribute.Key("artifact.purl")

	// ArtifactVersionKey is the attribute Key conforming to the "artifact.version"
	// semantic conventions. It represents the version of the artifact.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "v0.1.0", "1.2.1", "122691-build"
	ArtifactVersionKey = attribute.Key("artifact.version")
)

// ArtifactAttestationFilename returns an attribute KeyValue conforming to the
// "artifact.attestation.filename" semantic conventions. It represents the
// provenance filename of the built attestation which directly relates to the
// build artifact filename. This filename SHOULD accompany the artifact at
// publish time. See the [SLSA Relationship] specification for more information.
//
// [SLSA Relationship]: https://slsa.dev/spec/v1.0/distributing-provenance#relationship-between-artifacts-and-attestations
func ArtifactAttestationFilename(val string) attribute.KeyValue {
	return ArtifactAttestationFilenameKey.String(val)
}

// ArtifactAttestationHash returns an attribute KeyValue conforming to the
// "artifact.attestation.hash" semantic conventions. It represents the full
// [hash value (see glossary)], of the built attestation. Some envelopes in the
// [software attestation space] also refer to this as the **digest**.
//
// [hash value (see glossary)]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-5.pdf
// [software attestation space]: https://github.com/in-toto/attestation/tree/main/spec
func ArtifactAttestationHash(val string) attribute.KeyValue {
	return ArtifactAttestationHashKey.String(val)
}

// ArtifactAttestationID returns an attribute KeyValue conforming to the
// "artifact.attestation.id" semantic conventions. It represents the id of the
// build [software attestation].
//
// [software attestation]: https://slsa.dev/attestation-model
func ArtifactAttestationID(val string) attribute.KeyValue {
	return ArtifactAttestationIDKey.String(val)
}

// ArtifactFilename returns an attribute KeyValue conforming to the
// "artifact.filename" semantic conventions. It represents the human readable
// file name of the artifact, typically generated during build and release
// processes. Often includes the package name and version in the file name.
func ArtifactFilename(val string) attribute.KeyValue {
	return ArtifactFilenameKey.String(val)
}

// ArtifactHash returns an attribute KeyValue conforming to the "artifact.hash"
// semantic conventions. It represents the full [hash value (see glossary)],
// often found in checksum.txt on a release of the artifact and used to verify
// package integrity.
//
// [hash value (see glossary)]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-5.pdf
func ArtifactHash(val string) attribute.KeyValue {
	return ArtifactHashKey.String(val)
}

// ArtifactPurl returns an attribute KeyValue conforming to the "artifact.purl"
// semantic conventions. It represents the [Package URL] of the
// [package artifact] provides a standard way to identify and locate the packaged
// artifact.
//
// [Package URL]: https://github.com/package-url/purl-spec
// [package artifact]: https://slsa.dev/spec/v1.0/terminology#package-model
func ArtifactPurl(val string) attribute.KeyValue {
	return ArtifactPurlKey.String(val)
}

// ArtifactVersion returns an attribute KeyValue conforming to the
// "artifact.version" semantic conventions. It represents the version of the
// artifact.
func ArtifactVersion(val string) attribute.KeyValue {
	return ArtifactVersionKey.String(val)
}

// Namespace: aws
const (
	// AWSBedrockGuardrailIDKey is the attribute Key conforming to the
	// "aws.bedrock.guardrail.id" semantic conventions. It represents the unique
	// identifier of the AWS Bedrock Guardrail. A [guardrail] helps safeguard and
	// prevent unwanted behavior from model responses or user messages.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "sgi5gkybzqak"
	//
	// [guardrail]: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html
	AWSBedrockGuardrailIDKey = attribute.Key("aws.bedrock.guardrail.id")

	// AWSBedrockKnowledgeBaseIDKey is the attribute Key conforming to the
	// "aws.bedrock.knowledge_base.id" semantic conventions. It represents the
	// unique identifier of the AWS Bedrock Knowledge base. A [knowledge base] is a
	// bank of information that can be queried by models to generate more relevant
	// responses and augment prompts.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "XFWUPB9PAW"
	//
	// [knowledge base]: https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html
	AWSBedrockKnowledgeBaseIDKey = attribute.Key("aws.bedrock.knowledge_base.id")

	// AWSDynamoDBAttributeDefinitionsKey is the attribute Key conforming to the
	// "aws.dynamodb.attribute_definitions" semantic conventions. It represents the
	// JSON-serialized value of each item in the `AttributeDefinitions` request
	// field.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "{ "AttributeName": "string", "AttributeType": "string" }"
	AWSDynamoDBAttributeDefinitionsKey = attribute.Key("aws.dynamodb.attribute_definitions")

	// AWSDynamoDBAttributesToGetKey is the attribute Key conforming to the
	// "aws.dynamodb.attributes_to_get" semantic conventions. It represents the
	// value of the `AttributesToGet` request parameter.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "lives", "id"
	AWSDynamoDBAttributesToGetKey = attribute.Key("aws.dynamodb.attributes_to_get")

	// AWSDynamoDBConsistentReadKey is the attribute Key conforming to the
	// "aws.dynamodb.consistent_read" semantic conventions. It represents the value
	// of the `ConsistentRead` request parameter.
	//
	// Type: boolean
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	AWSDynamoDBConsistentReadKey = attribute.Key("aws.dynamodb.consistent_read")

	// AWSDynamoDBConsumedCapacityKey is the attribute Key conforming to the
	// "aws.dynamodb.consumed_capacity" semantic conventions. It represents the
	// JSON-serialized value of each item in the `ConsumedCapacity` response field.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "{ "CapacityUnits": number, "GlobalSecondaryIndexes": { "string" :
	// { "CapacityUnits": number, "ReadCapacityUnits": number, "WriteCapacityUnits":
	// number } }, "LocalSecondaryIndexes": { "string" : { "CapacityUnits": number,
	// "ReadCapacityUnits": number, "WriteCapacityUnits": number } },
	// "ReadCapacityUnits": number, "Table": { "CapacityUnits": number,
	// "ReadCapacityUnits": number, "WriteCapacityUnits": number }, "TableName":
	// "string", "WriteCapacityUnits": number }"
	AWSDynamoDBConsumedCapacityKey = attribute.Key("aws.dynamodb.consumed_capacity")

	// AWSDynamoDBCountKey is the attribute Key conforming to the
	// "aws.dynamodb.count" semantic conventions. It represents the value of the
	// `Count` response parameter.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 10
	AWSDynamoDBCountKey = attribute.Key("aws.dynamodb.count")

	// AWSDynamoDBExclusiveStartTableKey is the attribute Key conforming to the
	// "aws.dynamodb.exclusive_start_table" semantic conventions. It represents the
	// value of the `ExclusiveStartTableName` request parameter.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Users", "CatsTable"
	AWSDynamoDBExclusiveStartTableKey = attribute.Key("aws.dynamodb.exclusive_start_table")

	// AWSDynamoDBGlobalSecondaryIndexUpdatesKey is the attribute Key conforming to
	// the "aws.dynamodb.global_secondary_index_updates" semantic conventions. It
	// represents the JSON-serialized value of each item in the
	// `GlobalSecondaryIndexUpdates` request field.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "{ "Create": { "IndexName": "string", "KeySchema": [ {
	// "AttributeName": "string", "KeyType": "string" } ], "Projection": {
	// "NonKeyAttributes": [ "string" ], "ProjectionType": "string" },
	// "ProvisionedThroughput": { "ReadCapacityUnits": number, "WriteCapacityUnits":
	// number } }"
	AWSDynamoDBGlobalSecondaryIndexUpdatesKey = attribute.Key("aws.dynamodb.global_secondary_index_updates")

	// AWSDynamoDBGlobalSecondaryIndexesKey is the attribute Key conforming to the
	// "aws.dynamodb.global_secondary_indexes" semantic conventions. It represents
	// the JSON-serialized value of each item of the `GlobalSecondaryIndexes`
	// request field.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "{ "IndexName": "string", "KeySchema": [ { "AttributeName":
	// "string", "KeyType": "string" } ], "Projection": { "NonKeyAttributes": [
	// "string" ], "ProjectionType": "string" }, "ProvisionedThroughput": {
	// "ReadCapacityUnits": number, "WriteCapacityUnits": number } }"
	AWSDynamoDBGlobalSecondaryIndexesKey = attribute.Key("aws.dynamodb.global_secondary_indexes")

	// AWSDynamoDBIndexNameKey is the attribute Key conforming to the
	// "aws.dynamodb.index_name" semantic conventions. It represents the value of
	// the `IndexName` request parameter.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "name_to_group"
	AWSDynamoDBIndexNameKey = attribute.Key("aws.dynamodb.index_name")

	// AWSDynamoDBItemCollectionMetricsKey is the attribute Key conforming to the
	// "aws.dynamodb.item_collection_metrics" semantic conventions. It represents
	// the JSON-serialized value of the `ItemCollectionMetrics` response field.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "{ "string" : [ { "ItemCollectionKey": { "string" : { "B": blob,
	// "BOOL": boolean, "BS": [ blob ], "L": [ "AttributeValue" ], "M": { "string" :
	// "AttributeValue" }, "N": "string", "NS": [ "string" ], "NULL": boolean, "S":
	// "string", "SS": [ "string" ] } }, "SizeEstimateRangeGB": [ number ] } ] }"
	AWSDynamoDBItemCollectionMetricsKey = attribute.Key("aws.dynamodb.item_collection_metrics")

	// AWSDynamoDBLimitKey is the attribute Key conforming to the
	// "aws.dynamodb.limit" semantic conventions. It represents the value of the
	// `Limit` request parameter.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 10
	AWSDynamoDBLimitKey = attribute.Key("aws.dynamodb.limit")

	// AWSDynamoDBLocalSecondaryIndexesKey is the attribute Key conforming to the
	// "aws.dynamodb.local_secondary_indexes" semantic conventions. It represents
	// the JSON-serialized value of each item of the `LocalSecondaryIndexes` request
	// field.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "{ "IndexArn": "string", "IndexName": "string", "IndexSizeBytes":
	// number, "ItemCount": number, "KeySchema": [ { "AttributeName": "string",
	// "KeyType": "string" } ], "Projection": { "NonKeyAttributes": [ "string" ],
	// "ProjectionType": "string" } }"
	AWSDynamoDBLocalSecondaryIndexesKey = attribute.Key("aws.dynamodb.local_secondary_indexes")

	// AWSDynamoDBProjectionKey is the attribute Key conforming to the
	// "aws.dynamodb.projection" semantic conventions. It represents the value of
	// the `ProjectionExpression` request parameter.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Title", "Title, Price, Color", "Title, Description, RelatedItems,
	// ProductReviews"
	AWSDynamoDBProjectionKey = attribute.Key("aws.dynamodb.projection")

	// AWSDynamoDBProvisionedReadCapacityKey is the attribute Key conforming to the
	// "aws.dynamodb.provisioned_read_capacity" semantic conventions. It represents
	// the value of the `ProvisionedThroughput.ReadCapacityUnits` request parameter.
	//
	// Type: double
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1.0, 2.0
	AWSDynamoDBProvisionedReadCapacityKey = attribute.Key("aws.dynamodb.provisioned_read_capacity")

	// AWSDynamoDBProvisionedWriteCapacityKey is the attribute Key conforming to the
	// "aws.dynamodb.provisioned_write_capacity" semantic conventions. It represents
	// the value of the `ProvisionedThroughput.WriteCapacityUnits` request
	// parameter.
	//
	// Type: double
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1.0, 2.0
	AWSDynamoDBProvisionedWriteCapacityKey = attribute.Key("aws.dynamodb.provisioned_write_capacity")

	// AWSDynamoDBScanForwardKey is the attribute Key conforming to the
	// "aws.dynamodb.scan_forward" semantic conventions. It represents the value of
	// the `ScanIndexForward` request parameter.
	//
	// Type: boolean
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	AWSDynamoDBScanForwardKey = attribute.Key("aws.dynamodb.scan_forward")

	// AWSDynamoDBScannedCountKey is the attribute Key conforming to the
	// "aws.dynamodb.scanned_count" semantic conventions. It represents the value of
	// the `ScannedCount` response parameter.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 50
	AWSDynamoDBScannedCountKey = attribute.Key("aws.dynamodb.scanned_count")

	// AWSDynamoDBSegmentKey is the attribute Key conforming to the
	// "aws.dynamodb.segment" semantic conventions. It represents the value of the
	// `Segment` request parameter.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 10
	AWSDynamoDBSegmentKey = attribute.Key("aws.dynamodb.segment")

	// AWSDynamoDBSelectKey is the attribute Key conforming to the
	// "aws.dynamodb.select" semantic conventions. It represents the value of the
	// `Select` request parameter.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "ALL_ATTRIBUTES", "COUNT"
	AWSDynamoDBSelectKey = attribute.Key("aws.dynamodb.select")

	// AWSDynamoDBTableCountKey is the attribute Key conforming to the
	// "aws.dynamodb.table_count" semantic conventions. It represents the number of
	// items in the `TableNames` response parameter.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 20
	AWSDynamoDBTableCountKey = attribute.Key("aws.dynamodb.table_count")

	// AWSDynamoDBTableNamesKey is the attribute Key conforming to the
	// "aws.dynamodb.table_names" semantic conventions. It represents the keys in
	// the `RequestItems` object field.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Users", "Cats"
	AWSDynamoDBTableNamesKey = attribute.Key("aws.dynamodb.table_names")

	// AWSDynamoDBTotalSegmentsKey is the attribute Key conforming to the
	// "aws.dynamodb.total_segments" semantic conventions. It represents the value
	// of the `TotalSegments` request parameter.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 100
	AWSDynamoDBTotalSegmentsKey = attribute.Key("aws.dynamodb.total_segments")

	// AWSECSClusterARNKey is the attribute Key conforming to the
	// "aws.ecs.cluster.arn" semantic conventions. It represents the ARN of an
	// [ECS cluster].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "arn:aws:ecs:us-west-2:123456789123:cluster/my-cluster"
	//
	// [ECS cluster]: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/clusters.html
	AWSECSClusterARNKey = attribute.Key("aws.ecs.cluster.arn")

	// AWSECSContainerARNKey is the attribute Key conforming to the
	// "aws.ecs.container.arn" semantic conventions. It represents the Amazon
	// Resource Name (ARN) of an [ECS container instance].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// "arn:aws:ecs:us-west-1:123456789123:container/32624152-9086-4f0e-acae-1a75b14fe4d9"
	//
	// [ECS container instance]: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_instances.html
	AWSECSContainerARNKey = attribute.Key("aws.ecs.container.arn")

	// AWSECSLaunchtypeKey is the attribute Key conforming to the
	// "aws.ecs.launchtype" semantic conventions. It represents the [launch type]
	// for an ECS task.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	//
	// [launch type]: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/launch_types.html
	AWSECSLaunchtypeKey = attribute.Key("aws.ecs.launchtype")

	// AWSECSTaskARNKey is the attribute Key conforming to the "aws.ecs.task.arn"
	// semantic conventions. It represents the ARN of a running [ECS task].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// "arn:aws:ecs:us-west-1:123456789123:task/10838bed-421f-43ef-870a-f43feacbbb5b",
	// "arn:aws:ecs:us-west-1:123456789123:task/my-cluster/task-id/23ebb8ac-c18f-46c6-8bbe-d55d0e37cfbd"
	//
	// [ECS task]: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-account-settings.html#ecs-resource-ids
	AWSECSTaskARNKey = attribute.Key("aws.ecs.task.arn")

	// AWSECSTaskFamilyKey is the attribute Key conforming to the
	// "aws.ecs.task.family" semantic conventions. It represents the family name of
	// the [ECS task definition] used to create the ECS task.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "opentelemetry-family"
	//
	// [ECS task definition]: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definitions.html
	AWSECSTaskFamilyKey = attribute.Key("aws.ecs.task.family")

	// AWSECSTaskIDKey is the attribute Key conforming to the "aws.ecs.task.id"
	// semantic conventions. It represents the ID of a running ECS task. The ID MUST
	// be extracted from `task.arn`.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "10838bed-421f-43ef-870a-f43feacbbb5b",
	// "23ebb8ac-c18f-46c6-8bbe-d55d0e37cfbd"
	AWSECSTaskIDKey = attribute.Key("aws.ecs.task.id")

	// AWSECSTaskRevisionKey is the attribute Key conforming to the
	// "aws.ecs.task.revision" semantic conventions. It represents the revision for
	// the task definition used to create the ECS task.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "8", "26"
	AWSECSTaskRevisionKey = attribute.Key("aws.ecs.task.revision")

	// AWSEKSClusterARNKey is the attribute Key conforming to the
	// "aws.eks.cluster.arn" semantic conventions. It represents the ARN of an EKS
	// cluster.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "arn:aws:ecs:us-west-2:123456789123:cluster/my-cluster"
	AWSEKSClusterARNKey = attribute.Key("aws.eks.cluster.arn")

	// AWSExtendedRequestIDKey is the attribute Key conforming to the
	// "aws.extended_request_id" semantic conventions. It represents the AWS
	// extended request ID as returned in the response header `x-amz-id-2`.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// "wzHcyEWfmOGDIE5QOhTAqFDoDWP3y8IUvpNINCwL9N4TEHbUw0/gZJ+VZTmCNCWR7fezEN3eCiQ="
	AWSExtendedRequestIDKey = attribute.Key("aws.extended_request_id")

	// AWSKinesisStreamNameKey is the attribute Key conforming to the
	// "aws.kinesis.stream_name" semantic conventions. It represents the name of the
	// AWS Kinesis [stream] the request refers to. Corresponds to the
	// `--stream-name` parameter of the Kinesis [describe-stream] operation.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "some-stream-name"
	//
	// [stream]: https://docs.aws.amazon.com/streams/latest/dev/introduction.html
	// [describe-stream]: https://docs.aws.amazon.com/cli/latest/reference/kinesis/describe-stream.html
	AWSKinesisStreamNameKey = attribute.Key("aws.kinesis.stream_name")

	// AWSLambdaInvokedARNKey is the attribute Key conforming to the
	// "aws.lambda.invoked_arn" semantic conventions. It represents the full invoked
	// ARN as provided on the `Context` passed to the function (
	// `Lambda-Runtime-Invoked-Function-Arn` header on the
	// `/runtime/invocation/next` applicable).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "arn:aws:lambda:us-east-1:123456:function:myfunction:myalias"
	// Note: This may be different from `cloud.resource_id` if an alias is involved.
	AWSLambdaInvokedARNKey = attribute.Key("aws.lambda.invoked_arn")

	// AWSLambdaResourceMappingIDKey is the attribute Key conforming to the
	// "aws.lambda.resource_mapping.id" semantic conventions. It represents the UUID
	// of the [AWS Lambda EvenSource Mapping]. An event source is mapped to a lambda
	// function. It's contents are read by Lambda and used to trigger a function.
	// This isn't available in the lambda execution context or the lambda runtime
	// environtment. This is going to be populated by the AWS SDK for each language
	// when that UUID is present. Some of these operations are
	// Create/Delete/Get/List/Update EventSourceMapping.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "587ad24b-03b9-4413-8202-bbd56b36e5b7"
	//
	// [AWS Lambda EvenSource Mapping]: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-lambda-eventsourcemapping.html
	AWSLambdaResourceMappingIDKey = attribute.Key("aws.lambda.resource_mapping.id")

	// AWSLogGroupARNsKey is the attribute Key conforming to the
	// "aws.log.group.arns" semantic conventions. It represents the Amazon Resource
	// Name(s) (ARN) of the AWS log group(s).
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "arn:aws:logs:us-west-1:123456789012:log-group:/aws/my/group:*"
	// Note: See the [log group ARN format documentation].
	//
	// [log group ARN format documentation]: https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/iam-access-control-overview-cwl.html#CWL_ARN_Format
	AWSLogGroupARNsKey = attribute.Key("aws.log.group.arns")

	// AWSLogGroupNamesKey is the attribute Key conforming to the
	// "aws.log.group.names" semantic conventions. It represents the name(s) of the
	// AWS log group(s) an application is writing to.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "/aws/lambda/my-function", "opentelemetry-service"
	// Note: Multiple log groups must be supported for cases like multi-container
	// applications, where a single application has sidecar containers, and each
	// write to their own log group.
	AWSLogGroupNamesKey = attribute.Key("aws.log.group.names")

	// AWSLogStreamARNsKey is the attribute Key conforming to the
	// "aws.log.stream.arns" semantic conventions. It represents the ARN(s) of the
	// AWS log stream(s).
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// "arn:aws:logs:us-west-1:123456789012:log-group:/aws/my/group:log-stream:logs/main/10838bed-421f-43ef-870a-f43feacbbb5b"
	// Note: See the [log stream ARN format documentation]. One log group can
	// contain several log streams, so these ARNs necessarily identify both a log
	// group and a log stream.
	//
	// [log stream ARN format documentation]: https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/iam-access-control-overview-cwl.html#CWL_ARN_Format
	AWSLogStreamARNsKey = attribute.Key("aws.log.stream.arns")

	// AWSLogStreamNamesKey is the attribute Key conforming to the
	// "aws.log.stream.names" semantic conventions. It represents the name(s) of the
	// AWS log stream(s) an application is writing to.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "logs/main/10838bed-421f-43ef-870a-f43feacbbb5b"
	AWSLogStreamNamesKey = attribute.Key("aws.log.stream.names")

	// AWSRequestIDKey is the attribute Key conforming to the "aws.request_id"
	// semantic conventions. It represents the AWS request ID as returned in the
	// response headers `x-amzn-requestid`, `x-amzn-request-id` or
	// `x-amz-request-id`.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "79b9da39-b7ae-508a-a6bc-864b2829c622", "C9ER4AJX75574TDJ"
	AWSRequestIDKey = attribute.Key("aws.request_id")

	// AWSS3BucketKey is the attribute Key conforming to the "aws.s3.bucket"
	// semantic conventions. It represents the S3 bucket name the request refers to.
	// Corresponds to the `--bucket` parameter of the [S3 API] operations.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "some-bucket-name"
	// Note: The `bucket` attribute is applicable to all S3 operations that
	// reference a bucket, i.e. that require the bucket name as a mandatory
	// parameter.
	// This applies to almost all S3 operations except `list-buckets`.
	//
	// [S3 API]: https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html
	AWSS3BucketKey = attribute.Key("aws.s3.bucket")

	// AWSS3CopySourceKey is the attribute Key conforming to the
	// "aws.s3.copy_source" semantic conventions. It represents the source object
	// (in the form `bucket`/`key`) for the copy operation.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "someFile.yml"
	// Note: The `copy_source` attribute applies to S3 copy operations and
	// corresponds to the `--copy-source` parameter
	// of the [copy-object operation within the S3 API].
	// This applies in particular to the following operations:
	//
	//   - [copy-object]
	//   - [upload-part-copy]
	//
	//
	// [copy-object operation within the S3 API]: https://docs.aws.amazon.com/cli/latest/reference/s3api/copy-object.html
	// [copy-object]: https://docs.aws.amazon.com/cli/latest/reference/s3api/copy-object.html
	// [upload-part-copy]: https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html
	AWSS3CopySourceKey = attribute.Key("aws.s3.copy_source")

	// AWSS3DeleteKey is the attribute Key conforming to the "aws.s3.delete"
	// semantic conventions. It represents the delete request container that
	// specifies the objects to be deleted.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// "Objects=[{Key=string,VersionId=string},{Key=string,VersionId=string}],Quiet=boolean"
	// Note: The `delete` attribute is only applicable to the [delete-object]
	// operation.
	// The `delete` attribute corresponds to the `--delete` parameter of the
	// [delete-objects operation within the S3 API].
	//
	// [delete-object]: https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-object.html
	// [delete-objects operation within the S3 API]: https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-objects.html
	AWSS3DeleteKey = attribute.Key("aws.s3.delete")

	// AWSS3KeyKey is the attribute Key conforming to the "aws.s3.key" semantic
	// conventions. It represents the S3 object key the request refers to.
	// Corresponds to the `--key` parameter of the [S3 API] operations.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "someFile.yml"
	// Note: The `key` attribute is applicable to all object-related S3 operations,
	// i.e. that require the object key as a mandatory parameter.
	// This applies in particular to the following operations:
	//
	//   - [copy-object]
	//   - [delete-object]
	//   - [get-object]
	//   - [head-object]
	//   - [put-object]
	//   - [restore-object]
	//   - [select-object-content]
	//   - [abort-multipart-upload]
	//   - [complete-multipart-upload]
	//   - [create-multipart-upload]
	//   - [list-parts]
	//   - [upload-part]
	//   - [upload-part-copy]
	//
	//
	// [S3 API]: https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html
	// [copy-object]: https://docs.aws.amazon.com/cli/latest/reference/s3api/copy-object.html
	// [delete-object]: https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-object.html
	// [get-object]: https://docs.aws.amazon.com/cli/latest/reference/s3api/get-object.html
	// [head-object]: https://docs.aws.amazon.com/cli/latest/reference/s3api/head-object.html
	// [put-object]: https://docs.aws.amazon.com/cli/latest/reference/s3api/put-object.html
	// [restore-object]: https://docs.aws.amazon.com/cli/latest/reference/s3api/restore-object.html
	// [select-object-content]: https://docs.aws.amazon.com/cli/latest/reference/s3api/select-object-content.html
	// [abort-multipart-upload]: https://docs.aws.amazon.com/cli/latest/reference/s3api/abort-multipart-upload.html
	// [complete-multipart-upload]: https://docs.aws.amazon.com/cli/latest/reference/s3api/complete-multipart-upload.html
	// [create-multipart-upload]: https://docs.aws.amazon.com/cli/latest/reference/s3api/create-multipart-upload.html
	// [list-parts]: https://docs.aws.amazon.com/cli/latest/reference/s3api/list-parts.html
	// [upload-part]: https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html
	// [upload-part-copy]: https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html
	AWSS3KeyKey = attribute.Key("aws.s3.key")

	// AWSS3PartNumberKey is the attribute Key conforming to the
	// "aws.s3.part_number" semantic conventions. It represents the part number of
	// the part being uploaded in a multipart-upload operation. This is a positive
	// integer between 1 and 10,000.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 3456
	// Note: The `part_number` attribute is only applicable to the [upload-part]
	// and [upload-part-copy] operations.
	// The `part_number` attribute corresponds to the `--part-number` parameter of
	// the
	// [upload-part operation within the S3 API].
	//
	// [upload-part]: https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html
	// [upload-part-copy]: https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html
	// [upload-part operation within the S3 API]: https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html
	AWSS3PartNumberKey = attribute.Key("aws.s3.part_number")

	// AWSS3UploadIDKey is the attribute Key conforming to the "aws.s3.upload_id"
	// semantic conventions. It represents the upload ID that identifies the
	// multipart upload.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "dfRtDYWFbkRONycy.Yxwh66Yjlx.cph0gtNBtJ"
	// Note: The `upload_id` attribute applies to S3 multipart-upload operations and
	// corresponds to the `--upload-id` parameter
	// of the [S3 API] multipart operations.
	// This applies in particular to the following operations:
	//
	//   - [abort-multipart-upload]
	//   - [complete-multipart-upload]
	//   - [list-parts]
	//   - [upload-part]
	//   - [upload-part-copy]
	//
	//
	// [S3 API]: https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html
	// [abort-multipart-upload]: https://docs.aws.amazon.com/cli/latest/reference/s3api/abort-multipart-upload.html
	// [complete-multipart-upload]: https://docs.aws.amazon.com/cli/latest/reference/s3api/complete-multipart-upload.html
	// [list-parts]: https://docs.aws.amazon.com/cli/latest/reference/s3api/list-parts.html
	// [upload-part]: https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html
	// [upload-part-copy]: https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html
	AWSS3UploadIDKey = attribute.Key("aws.s3.upload_id")

	// AWSSecretsmanagerSecretARNKey is the attribute Key conforming to the
	// "aws.secretsmanager.secret.arn" semantic conventions. It represents the ARN
	// of the Secret stored in the Secrets Mangger.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// "arn:aws:secretsmanager:us-east-1:123456789012:secret:SecretName-6RandomCharacters"
	AWSSecretsmanagerSecretARNKey = attribute.Key("aws.secretsmanager.secret.arn")

	// AWSSNSTopicARNKey is the attribute Key conforming to the "aws.sns.topic.arn"
	// semantic conventions. It represents the ARN of the AWS SNS Topic. An Amazon
	// SNS [topic] is a logical access point that acts as a communication channel.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "arn:aws:sns:us-east-1:123456789012:mystack-mytopic-NZJ5JSMVGFIE"
	//
	// [topic]: https://docs.aws.amazon.com/sns/latest/dg/sns-create-topic.html
	AWSSNSTopicARNKey = attribute.Key("aws.sns.topic.arn")

	// AWSSQSQueueURLKey is the attribute Key conforming to the "aws.sqs.queue.url"
	// semantic conventions. It represents the URL of the AWS SQS Queue. It's a
	// unique identifier for a queue in Amazon Simple Queue Service (SQS) and is
	// used to access the queue and perform actions on it.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "https://sqs.us-east-1.amazonaws.com/123456789012/MyQueue"
	AWSSQSQueueURLKey = attribute.Key("aws.sqs.queue.url")

	// AWSStepFunctionsActivityARNKey is the attribute Key conforming to the
	// "aws.step_functions.activity.arn" semantic conventions. It represents the ARN
	// of the AWS Step Functions Activity.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "arn:aws:states:us-east-1:123456789012:activity:get-greeting"
	AWSStepFunctionsActivityARNKey = attribute.Key("aws.step_functions.activity.arn")

	// AWSStepFunctionsStateMachineARNKey is the attribute Key conforming to the
	// "aws.step_functions.state_machine.arn" semantic conventions. It represents
	// the ARN of the AWS Step Functions State Machine.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// "arn:aws:states:us-east-1:123456789012:stateMachine:myStateMachine:1"
	AWSStepFunctionsStateMachineARNKey = attribute.Key("aws.step_functions.state_machine.arn")
)

// AWSBedrockGuardrailID returns an attribute KeyValue conforming to the
// "aws.bedrock.guardrail.id" semantic conventions. It represents the unique
// identifier of the AWS Bedrock Guardrail. A [guardrail] helps safeguard and
// prevent unwanted behavior from model responses or user messages.
//
// [guardrail]: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html
func AWSBedrockGuardrailID(val string) attribute.KeyValue {
	return AWSBedrockGuardrailIDKey.String(val)
}

// AWSBedrockKnowledgeBaseID returns an attribute KeyValue conforming to the
// "aws.bedrock.knowledge_base.id" semantic conventions. It represents the unique
// identifier of the AWS Bedrock Knowledge base. A [knowledge base] is a bank of
// information that can be queried by models to generate more relevant responses
// and augment prompts.
//
// [knowledge base]: https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html
func AWSBedrockKnowledgeBaseID(val string) attribute.KeyValue {
	return AWSBedrockKnowledgeBaseIDKey.String(val)
}

// AWSDynamoDBAttributeDefinitions returns an attribute KeyValue conforming to
// the "aws.dynamodb.attribute_definitions" semantic conventions. It represents
// the JSON-serialized value of each item in the `AttributeDefinitions` request
// field.
func AWSDynamoDBAttributeDefinitions(val ...string) attribute.KeyValue {
	return AWSDynamoDBAttributeDefinitionsKey.StringSlice(val)
}

// AWSDynamoDBAttributesToGet returns an attribute KeyValue conforming to the
// "aws.dynamodb.attributes_to_get" semantic conventions. It represents the value
// of the `AttributesToGet` request parameter.
func AWSDynamoDBAttributesToGet(val ...string) attribute.KeyValue {
	return AWSDynamoDBAttributesToGetKey.StringSlice(val)
}

// AWSDynamoDBConsistentRead returns an attribute KeyValue conforming to the
// "aws.dynamodb.consistent_read" semantic conventions. It represents the value
// of the `ConsistentRead` request parameter.
func AWSDynamoDBConsistentRead(val bool) attribute.KeyValue {
	return AWSDynamoDBConsistentReadKey.Bool(val)
}

// AWSDynamoDBConsumedCapacity returns an attribute KeyValue conforming to the
// "aws.dynamodb.consumed_capacity" semantic conventions. It represents the
// JSON-serialized value of each item in the `ConsumedCapacity` response field.
func AWSDynamoDBConsumedCapacity(val ...string) attribute.KeyValue {
	return AWSDynamoDBConsumedCapacityKey.StringSlice(val)
}

// AWSDynamoDBCount returns an attribute KeyValue conforming to the
// "aws.dynamodb.count" semantic conventions. It represents the value of the
// `Count` response parameter.
func AWSDynamoDBCount(val int) attribute.KeyValue {
	return AWSDynamoDBCountKey.Int(val)
}

// AWSDynamoDBExclusiveStartTable returns an attribute KeyValue conforming to the
// "aws.dynamodb.exclusive_start_table" semantic conventions. It represents the
// value of the `ExclusiveStartTableName` request parameter.
func AWSDynamoDBExclusiveStartTable(val string) attribute.KeyValue {
	return AWSDynamoDBExclusiveStartTableKey.String(val)
}

// AWSDynamoDBGlobalSecondaryIndexUpdates returns an attribute KeyValue
// conforming to the "aws.dynamodb.global_secondary_index_updates" semantic
// conventions. It represents the JSON-serialized value of each item in the
// `GlobalSecondaryIndexUpdates` request field.
func AWSDynamoDBGlobalSecondaryIndexUpdates(val ...string) attribute.KeyValue {
	return AWSDynamoDBGlobalSecondaryIndexUpdatesKey.StringSlice(val)
}

// AWSDynamoDBGlobalSecondaryIndexes returns an attribute KeyValue conforming to
// the "aws.dynamodb.global_secondary_indexes" semantic conventions. It
// represents the JSON-serialized value of each item of the
// `GlobalSecondaryIndexes` request field.
func AWSDynamoDBGlobalSecondaryIndexes(val ...string) attribute.KeyValue {
	return AWSDynamoDBGlobalSecondaryIndexesKey.StringSlice(val)
}

// AWSDynamoDBIndexName returns an attribute KeyValue conforming to the
// "aws.dynamodb.index_name" semantic conventions. It represents the value of the
// `IndexName` request parameter.
func AWSDynamoDBIndexName(val string) attribute.KeyValue {
	return AWSDynamoDBIndexNameKey.String(val)
}

// AWSDynamoDBItemCollectionMetrics returns an attribute KeyValue conforming to
// the "aws.dynamodb.item_collection_metrics" semantic conventions. It represents
// the JSON-serialized value of the `ItemCollectionMetrics` response field.
func AWSDynamoDBItemCollectionMetrics(val string) attribute.KeyValue {
	return AWSDynamoDBItemCollectionMetricsKey.String(val)
}

// AWSDynamoDBLimit returns an attribute KeyValue conforming to the
// "aws.dynamodb.limit" semantic conventions. It represents the value of the
// `Limit` request parameter.
func AWSDynamoDBLimit(val int) attribute.KeyValue {
	return AWSDynamoDBLimitKey.Int(val)
}

// AWSDynamoDBLocalSecondaryIndexes returns an attribute KeyValue conforming to
// the "aws.dynamodb.local_secondary_indexes" semantic conventions. It represents
// the JSON-serialized value of each item of the `LocalSecondaryIndexes` request
// field.
func AWSDynamoDBLocalSecondaryIndexes(val ...string) attribute.KeyValue {
	return AWSDynamoDBLocalSecondaryIndexesKey.StringSlice(val)
}

// AWSDynamoDBProjection returns an attribute KeyValue conforming to the
// "aws.dynamodb.projection" semantic conventions. It represents the value of the
// `ProjectionExpression` request parameter.
func AWSDynamoDBProjection(val string) attribute.KeyValue {
	return AWSDynamoDBProjectionKey.String(val)
}

// AWSDynamoDBProvisionedReadCapacity returns an attribute KeyValue conforming to
// the "aws.dynamodb.provisioned_read_capacity" semantic conventions. It
// represents the value of the `ProvisionedThroughput.ReadCapacityUnits` request
// parameter.
func AWSDynamoDBProvisionedReadCapacity(val float64) attribute.KeyValue {
	return AWSDynamoDBProvisionedReadCapacityKey.Float64(val)
}

// AWSDynamoDBProvisionedWriteCapacity returns an attribute KeyValue conforming
// to the "aws.dynamodb.provisioned_write_capacity" semantic conventions. It
// represents the value of the `ProvisionedThroughput.WriteCapacityUnits` request
// parameter.
func AWSDynamoDBProvisionedWriteCapacity(val float64) attribute.KeyValue {
	return AWSDynamoDBProvisionedWriteCapacityKey.Float64(val)
}

// AWSDynamoDBScanForward returns an attribute KeyValue conforming to the
// "aws.dynamodb.scan_forward" semantic conventions. It represents the value of
// the `ScanIndexForward` request parameter.
func AWSDynamoDBScanForward(val bool) attribute.KeyValue {
	return AWSDynamoDBScanForwardKey.Bool(val)
}

// AWSDynamoDBScannedCount returns an attribute KeyValue conforming to the
// "aws.dynamodb.scanned_count" semantic conventions. It represents the value of
// the `ScannedCount` response parameter.
func AWSDynamoDBScannedCount(val int) attribute.KeyValue {
	return AWSDynamoDBScannedCountKey.Int(val)
}

// AWSDynamoDBSegment returns an attribute KeyValue conforming to the
// "aws.dynamodb.segment" semantic conventions. It represents the value of the
// `Segment` request parameter.
func AWSDynamoDBSegment(val int) attribute.KeyValue {
	return AWSDynamoDBSegmentKey.Int(val)
}

// AWSDynamoDBSelect returns an attribute KeyValue conforming to the
// "aws.dynamodb.select" semantic conventions. It represents the value of the
// `Select` request parameter.
func AWSDynamoDBSelect(val string) attribute.KeyValue {
	return AWSDynamoDBSelectKey.String(val)
}

// AWSDynamoDBTableCount returns an attribute KeyValue conforming to the
// "aws.dynamodb.table_count" semantic conventions. It represents the number of
// items in the `TableNames` response parameter.
func AWSDynamoDBTableCount(val int) attribute.KeyValue {
	return AWSDynamoDBTableCountKey.Int(val)
}

// AWSDynamoDBTableNames returns an attribute KeyValue conforming to the
// "aws.dynamodb.table_names" semantic conventions. It represents the keys in the
// `RequestItems` object field.
func AWSDynamoDBTableNames(val ...string) attribute.KeyValue {
	return AWSDynamoDBTableNamesKey.StringSlice(val)
}

// AWSDynamoDBTotalSegments returns an attribute KeyValue conforming to the
// "aws.dynamodb.total_segments" semantic conventions. It represents the value of
// the `TotalSegments` request parameter.
func AWSDynamoDBTotalSegments(val int) attribute.KeyValue {
	return AWSDynamoDBTotalSegmentsKey.Int(val)
}

// AWSECSClusterARN returns an attribute KeyValue conforming to the
// "aws.ecs.cluster.arn" semantic conventions. It represents the ARN of an
// [ECS cluster].
//
// [ECS cluster]: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/clusters.html
func AWSECSClusterARN(val string) attribute.KeyValue {
	return AWSECSClusterARNKey.String(val)
}

// AWSECSContainerARN returns an attribute KeyValue conforming to the
// "aws.ecs.container.arn" semantic conventions. It represents the Amazon
// Resource Name (ARN) of an [ECS container instance].
//
// [ECS container instance]: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_instances.html
func AWSECSContainerARN(val string) attribute.KeyValue {
	return AWSECSContainerARNKey.String(val)
}

// AWSECSTaskARN returns an attribute KeyValue conforming to the
// "aws.ecs.task.arn" semantic conventions. It represents the ARN of a running
// [ECS task].
//
// [ECS task]: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-account-settings.html#ecs-resource-ids
func AWSECSTaskARN(val string) attribute.KeyValue {
	return AWSECSTaskARNKey.String(val)
}

// AWSECSTaskFamily returns an attribute KeyValue conforming to the
// "aws.ecs.task.family" semantic conventions. It represents the family name of
// the [ECS task definition] used to create the ECS task.
//
// [ECS task definition]: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definitions.html
func AWSECSTaskFamily(val string) attribute.KeyValue {
	return AWSECSTaskFamilyKey.String(val)
}

// AWSECSTaskID returns an attribute KeyValue conforming to the "aws.ecs.task.id"
// semantic conventions. It represents the ID of a running ECS task. The ID MUST
// be extracted from `task.arn`.
func AWSECSTaskID(val string) attribute.KeyValue {
	return AWSECSTaskIDKey.String(val)
}

// AWSECSTaskRevision returns an attribute KeyValue conforming to the
// "aws.ecs.task.revision" semantic conventions. It represents the revision for
// the task definition used to create the ECS task.
func AWSECSTaskRevision(val string) attribute.KeyValue {
	return AWSECSTaskRevisionKey.String(val)
}

// AWSEKSClusterARN returns an attribute KeyValue conforming to the
// "aws.eks.cluster.arn" semantic conventions. It represents the ARN of an EKS
// cluster.
func AWSEKSClusterARN(val string) attribute.KeyValue {
	return AWSEKSClusterARNKey.String(val)
}

// AWSExtendedRequestID returns an attribute KeyValue conforming to the
// "aws.extended_request_id" semantic conventions. It represents the AWS extended
// request ID as returned in the response header `x-amz-id-2`.
func AWSExtendedRequestID(val string) attribute.KeyValue {
	return AWSExtendedRequestIDKey.String(val)
}

// AWSKinesisStreamName returns an attribute KeyValue conforming to the
// "aws.kinesis.stream_name" semantic conventions. It represents the name of the
// AWS Kinesis [stream] the request refers to. Corresponds to the `--stream-name`
//  parameter of the Kinesis [describe-stream] operation.
//
// [stream]: https://docs.aws.amazon.com/streams/latest/dev/introduction.html
// [describe-stream]: https://docs.aws.amazon.com/cli/latest/reference/kinesis/describe-stream.html
func AWSKinesisStreamName(val string) attribute.KeyValue {
	return AWSKinesisStreamNameKey.String(val)
}

// AWSLambdaInvokedARN returns an attribute KeyValue conforming to the
// "aws.lambda.invoked_arn" semantic conventions. It represents the full invoked
// ARN as provided on the `Context` passed to the function (
// `Lambda-Runtime-Invoked-Function-Arn` header on the `/runtime/invocation/next`
//  applicable).
func AWSLambdaInvokedARN(val string) attribute.KeyValue {
	return AWSLambdaInvokedARNKey.String(val)
}

// AWSLambdaResourceMappingID returns an attribute KeyValue conforming to the
// "aws.lambda.resource_mapping.id" semantic conventions. It represents the UUID
// of the [AWS Lambda EvenSource Mapping]. An event source is mapped to a lambda
// function. It's contents are read by Lambda and used to trigger a function.
// This isn't available in the lambda execution context or the lambda runtime
// environtment. This is going to be populated by the AWS SDK for each language
// when that UUID is present. Some of these operations are
// Create/Delete/Get/List/Update EventSourceMapping.
//
// [AWS Lambda EvenSource Mapping]: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-lambda-eventsourcemapping.html
func AWSLambdaResourceMappingID(val string) attribute.KeyValue {
	return AWSLambdaResourceMappingIDKey.String(val)
}

// AWSLogGroupARNs returns an attribute KeyValue conforming to the
// "aws.log.group.arns" semantic conventions. It represents the Amazon Resource
// Name(s) (ARN) of the AWS log group(s).
func AWSLogGroupARNs(val ...string) attribute.KeyValue {
	return AWSLogGroupARNsKey.StringSlice(val)
}

// AWSLogGroupNames returns an attribute KeyValue conforming to the
// "aws.log.group.names" semantic conventions. It represents the name(s) of the
// AWS log group(s) an application is writing to.
func AWSLogGroupNames(val ...string) attribute.KeyValue {
	return AWSLogGroupNamesKey.StringSlice(val)
}

// AWSLogStreamARNs returns an attribute KeyValue conforming to the
// "aws.log.stream.arns" semantic conventions. It represents the ARN(s) of the
// AWS log stream(s).
func AWSLogStreamARNs(val ...string) attribute.KeyValue {
	return AWSLogStreamARNsKey.StringSlice(val)
}

// AWSLogStreamNames returns an attribute KeyValue conforming to the
// "aws.log.stream.names" semantic conventions. It represents the name(s) of the
// AWS log stream(s) an application is writing to.
func AWSLogStreamNames(val ...string) attribute.KeyValue {
	return AWSLogStreamNamesKey.StringSlice(val)
}

// AWSRequestID returns an attribute KeyValue conforming to the "aws.request_id"
// semantic conventions. It represents the AWS request ID as returned in the
// response headers `x-amzn-requestid`, `x-amzn-request-id` or `x-amz-request-id`
// .
func AWSRequestID(val string) attribute.KeyValue {
	return AWSRequestIDKey.String(val)
}

// AWSS3Bucket returns an attribute KeyValue conforming to the "aws.s3.bucket"
// semantic conventions. It represents the S3 bucket name the request refers to.
// Corresponds to the `--bucket` parameter of the [S3 API] operations.
//
// [S3 API]: https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html
func AWSS3Bucket(val string) attribute.KeyValue {
	return AWSS3BucketKey.String(val)
}

// AWSS3CopySource returns an attribute KeyValue conforming to the
// "aws.s3.copy_source" semantic conventions. It represents the source object (in
// the form `bucket`/`key`) for the copy operation.
func AWSS3CopySource(val string) attribute.KeyValue {
	return AWSS3CopySourceKey.String(val)
}

// AWSS3Delete returns an attribute KeyValue conforming to the "aws.s3.delete"
// semantic conventions. It represents the delete request container that
// specifies the objects to be deleted.
func AWSS3Delete(val string) attribute.KeyValue {
	return AWSS3DeleteKey.String(val)
}

// AWSS3Key returns an attribute KeyValue conforming to the "aws.s3.key" semantic
// conventions. It represents the S3 object key the request refers to.
// Corresponds to the `--key` parameter of the [S3 API] operations.
//
// [S3 API]: https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html
func AWSS3Key(val string) attribute.KeyValue {
	return AWSS3KeyKey.String(val)
}

// AWSS3PartNumber returns an attribute KeyValue conforming to the
// "aws.s3.part_number" semantic conventions. It represents the part number of
// the part being uploaded in a multipart-upload operation. This is a positive
// integer between 1 and 10,000.
func AWSS3PartNumber(val int) attribute.KeyValue {
	return AWSS3PartNumberKey.Int(val)
}

// AWSS3UploadID returns an attribute KeyValue conforming to the
// "aws.s3.upload_id" semantic conventions. It represents the upload ID that
// identifies the multipart upload.
func AWSS3UploadID(val string) attribute.KeyValue {
	return AWSS3UploadIDKey.String(val)
}

// AWSSecretsmanagerSecretARN returns an attribute KeyValue conforming to the
// "aws.secretsmanager.secret.arn" semantic conventions. It represents the ARN of
// the Secret stored in the Secrets Mangger.
func AWSSecretsmanagerSecretARN(val string) attribute.KeyValue {
	return AWSSecretsmanagerSecretARNKey.String(val)
}

// AWSSNSTopicARN returns an attribute KeyValue conforming to the
// "aws.sns.topic.arn" semantic conventions. It represents the ARN of the AWS SNS
// Topic. An Amazon SNS [topic] is a logical access point that acts as a
// communication channel.
//
// [topic]: https://docs.aws.amazon.com/sns/latest/dg/sns-create-topic.html
func AWSSNSTopicARN(val string) attribute.KeyValue {
	return AWSSNSTopicARNKey.String(val)
}

// AWSSQSQueueURL returns an attribute KeyValue conforming to the
// "aws.sqs.queue.url" semantic conventions. It represents the URL of the AWS SQS
// Queue. It's a unique identifier for a queue in Amazon Simple Queue Service
// (SQS) and is used to access the queue and perform actions on it.
func AWSSQSQueueURL(val string) attribute.KeyValue {
	return AWSSQSQueueURLKey.String(val)
}

// AWSStepFunctionsActivityARN returns an attribute KeyValue conforming to the
// "aws.step_functions.activity.arn" semantic conventions. It represents the ARN
// of the AWS Step Functions Activity.
func AWSStepFunctionsActivityARN(val string) attribute.KeyValue {
	return AWSStepFunctionsActivityARNKey.String(val)
}

// AWSStepFunctionsStateMachineARN returns an attribute KeyValue conforming to
// the "aws.step_functions.state_machine.arn" semantic conventions. It represents
// the ARN of the AWS Step Functions State Machine.
func AWSStepFunctionsStateMachineARN(val string) attribute.KeyValue {
	return AWSStepFunctionsStateMachineARNKey.String(val)
}

// Enum values for aws.ecs.launchtype
var (
	// Amazon EC2
	// Stability: development
	AWSECSLaunchtypeEC2 = AWSECSLaunchtypeKey.String("ec2")
	// Amazon Fargate
	// Stability: development
	AWSECSLaunchtypeFargate = AWSECSLaunchtypeKey.String("fargate")
)

// Namespace: azure
const (
	// AzureClientIDKey is the attribute Key conforming to the "azure.client.id"
	// semantic conventions. It represents the unique identifier of the client
	// instance.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "3ba4827d-4422-483f-b59f-85b74211c11d", "storage-client-1"
	AzureClientIDKey = attribute.Key("azure.client.id")

	// AzureCosmosDBConnectionModeKey is the attribute Key conforming to the
	// "azure.cosmosdb.connection.mode" semantic conventions. It represents the
	// cosmos client connection mode.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	AzureCosmosDBConnectionModeKey = attribute.Key("azure.cosmosdb.connection.mode")

	// AzureCosmosDBConsistencyLevelKey is the attribute Key conforming to the
	// "azure.cosmosdb.consistency.level" semantic conventions. It represents the
	// account or request [consistency level].
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Eventual", "ConsistentPrefix", "BoundedStaleness", "Strong",
	// "Session"
	//
	// [consistency level]: https://learn.microsoft.com/azure/cosmos-db/consistency-levels
	AzureCosmosDBConsistencyLevelKey = attribute.Key("azure.cosmosdb.consistency.level")

	// AzureCosmosDBOperationContactedRegionsKey is the attribute Key conforming to
	// the "azure.cosmosdb.operation.contacted_regions" semantic conventions. It
	// represents the list of regions contacted during operation in the order that
	// they were contacted. If there is more than one region listed, it indicates
	// that the operation was performed on multiple regions i.e. cross-regional
	// call.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "North Central US", "Australia East", "Australia Southeast"
	// Note: Region name matches the format of `displayName` in [Azure Location API]
	//
	// [Azure Location API]: https://learn.microsoft.com/rest/api/subscription/subscriptions/list-locations?view=rest-subscription-2021-10-01&tabs=HTTP#location
	AzureCosmosDBOperationContactedRegionsKey = attribute.Key("azure.cosmosdb.operation.contacted_regions")

	// AzureCosmosDBOperationRequestChargeKey is the attribute Key conforming to the
	// "azure.cosmosdb.operation.request_charge" semantic conventions. It represents
	// the number of request units consumed by the operation.
	//
	// Type: double
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 46.18, 1.0
	AzureCosmosDBOperationRequestChargeKey = attribute.Key("azure.cosmosdb.operation.request_charge")

	// AzureCosmosDBRequestBodySizeKey is the attribute Key conforming to the
	// "azure.cosmosdb.request.body.size" semantic conventions. It represents the
	// request payload size in bytes.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	AzureCosmosDBRequestBodySizeKey = attribute.Key("azure.cosmosdb.request.body.size")

	// AzureCosmosDBResponseSubStatusCodeKey is the attribute Key conforming to the
	// "azure.cosmosdb.response.sub_status_code" semantic conventions. It represents
	// the cosmos DB sub status code.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1000, 1002
	AzureCosmosDBResponseSubStatusCodeKey = attribute.Key("azure.cosmosdb.response.sub_status_code")

	// AzureResourceProviderNamespaceKey is the attribute Key conforming to the
	// "azure.resource_provider.namespace" semantic conventions. It represents the
	// [Azure Resource Provider Namespace] as recognized by the client.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Microsoft.Storage", "Microsoft.KeyVault", "Microsoft.ServiceBus"
	//
	// [Azure Resource Provider Namespace]: https://learn.microsoft.com/azure/azure-resource-manager/management/azure-services-resource-providers
	AzureResourceProviderNamespaceKey = attribute.Key("azure.resource_provider.namespace")

	// AzureServiceRequestIDKey is the attribute Key conforming to the
	// "azure.service.request.id" semantic conventions. It represents the unique
	// identifier of the service request. It's generated by the Azure service and
	// returned with the response.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "00000000-0000-0000-0000-000000000000"
	AzureServiceRequestIDKey = attribute.Key("azure.service.request.id")
)

// AzureClientID returns an attribute KeyValue conforming to the
// "azure.client.id" semantic conventions. It represents the unique identifier of
// the client instance.
func AzureClientID(val string) attribute.KeyValue {
	return AzureClientIDKey.String(val)
}

// AzureCosmosDBOperationContactedRegions returns an attribute KeyValue
// conforming to the "azure.cosmosdb.operation.contacted_regions" semantic
// conventions. It represents the list of regions contacted during operation in
// the order that they were contacted. If there is more than one region listed,
// it indicates that the operation was performed on multiple regions i.e.
// cross-regional call.
func AzureCosmosDBOperationContactedRegions(val ...string) attribute.KeyValue {
	return AzureCosmosDBOperationContactedRegionsKey.StringSlice(val)
}

// AzureCosmosDBOperationRequestCharge returns an attribute KeyValue conforming
// to the "azure.cosmosdb.operation.request_charge" semantic conventions. It
// represents the number of request units consumed by the operation.
func AzureCosmosDBOperationRequestCharge(val float64) attribute.KeyValue {
	return AzureCosmosDBOperationRequestChargeKey.Float64(val)
}

// AzureCosmosDBRequestBodySize returns an attribute KeyValue conforming to the
// "azure.cosmosdb.request.body.size" semantic conventions. It represents the
// request payload size in bytes.
func AzureCosmosDBRequestBodySize(val int) attribute.KeyValue {
	return AzureCosmosDBRequestBodySizeKey.Int(val)
}

// AzureCosmosDBResponseSubStatusCode returns an attribute KeyValue conforming to
// the "azure.cosmosdb.response.sub_status_code" semantic conventions. It
// represents the cosmos DB sub status code.
func AzureCosmosDBResponseSubStatusCode(val int) attribute.KeyValue {
	return AzureCosmosDBResponseSubStatusCodeKey.Int(val)
}

// AzureResourceProviderNamespace returns an attribute KeyValue conforming to the
// "azure.resource_provider.namespace" semantic conventions. It represents the
// [Azure Resource Provider Namespace] as recognized by the client.
//
// [Azure Resource Provider Namespace]: https://learn.microsoft.com/azure/azure-resource-manager/management/azure-services-resource-providers
func AzureResourceProviderNamespace(val string) attribute.KeyValue {
	return AzureResourceProviderNamespaceKey.String(val)
}

// AzureServiceRequestID returns an attribute KeyValue conforming to the
// "azure.service.request.id" semantic conventions. It represents the unique
// identifier of the service request. It's generated by the Azure service and
// returned with the response.
func AzureServiceRequestID(val string) attribute.KeyValue {
	return AzureServiceRequestIDKey.String(val)
}

// Enum values for azure.cosmosdb.connection.mode
var (
	// Gateway (HTTP) connection.
	// Stability: development
	AzureCosmosDBConnectionModeGateway = AzureCosmosDBConnectionModeKey.String("gateway")
	// Direct connection.
	// Stability: development
	AzureCosmosDBConnectionModeDirect = AzureCosmosDBConnectionModeKey.String("direct")
)

// Enum values for azure.cosmosdb.consistency.level
var (
	// Strong
	// Stability: development
	AzureCosmosDBConsistencyLevelStrong = AzureCosmosDBConsistencyLevelKey.String("Strong")
	// Bounded Staleness
	// Stability: development
	AzureCosmosDBConsistencyLevelBoundedStaleness = AzureCosmosDBConsistencyLevelKey.String("BoundedStaleness")
	// Session
	// Stability: development
	AzureCosmosDBConsistencyLevelSession = AzureCosmosDBConsistencyLevelKey.String("Session")
	// Eventual
	// Stability: development
	AzureCosmosDBConsistencyLevelEventual = AzureCosmosDBConsistencyLevelKey.String("Eventual")
	// Consistent Prefix
	// Stability: development
	AzureCosmosDBConsistencyLevelConsistentPrefix = AzureCosmosDBConsistencyLevelKey.String("ConsistentPrefix")
)

// Namespace: browser
const (
	// BrowserBrandsKey is the attribute Key conforming to the "browser.brands"
	// semantic conventions. It represents the array of brand name and version
	// separated by a space.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: " Not A;Brand 99", "Chromium 99", "Chrome 99"
	// Note: This value is intended to be taken from the [UA client hints API] (
	// `navigator.userAgentData.brands`).
	//
	// [UA client hints API]: https://wicg.github.io/ua-client-hints/#interface
	BrowserBrandsKey = attribute.Key("browser.brands")

	// BrowserLanguageKey is the attribute Key conforming to the "browser.language"
	// semantic conventions. It represents the preferred language of the user using
	// the browser.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "en", "en-US", "fr", "fr-FR"
	// Note: This value is intended to be taken from the Navigator API
	// `navigator.language`.
	BrowserLanguageKey = attribute.Key("browser.language")

	// BrowserMobileKey is the attribute Key conforming to the "browser.mobile"
	// semantic conventions. It represents a boolean that is true if the browser is
	// running on a mobile device.
	//
	// Type: boolean
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: This value is intended to be taken from the [UA client hints API] (
	// `navigator.userAgentData.mobile`). If unavailable, this attribute SHOULD be
	// left unset.
	//
	// [UA client hints API]: https://wicg.github.io/ua-client-hints/#interface
	BrowserMobileKey = attribute.Key("browser.mobile")

	// BrowserPlatformKey is the attribute Key conforming to the "browser.platform"
	// semantic conventions. It represents the platform on which the browser is
	// running.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Windows", "macOS", "Android"
	// Note: This value is intended to be taken from the [UA client hints API] (
	// `navigator.userAgentData.platform`). If unavailable, the legacy
	// `navigator.platform` API SHOULD NOT be used instead and this attribute SHOULD
	// be left unset in order for the values to be consistent.
	// The list of possible values is defined in the
	// [W3C User-Agent Client Hints specification]. Note that some (but not all) of
	// these values can overlap with values in the
	// [`os.type` and `os.name` attributes]. However, for consistency, the values in
	// the `browser.platform` attribute should capture the exact value that the user
	// agent provides.
	//
	// [UA client hints API]: https://wicg.github.io/ua-client-hints/#interface
	// [W3C User-Agent Client Hints specification]: https://wicg.github.io/ua-client-hints/#sec-ch-ua-platform
	// [`os.type` and `os.name` attributes]: ./os.md
	BrowserPlatformKey = attribute.Key("browser.platform")
)

// BrowserBrands returns an attribute KeyValue conforming to the "browser.brands"
// semantic conventions. It represents the array of brand name and version
// separated by a space.
func BrowserBrands(val ...string) attribute.KeyValue {
	return BrowserBrandsKey.StringSlice(val)
}

// BrowserLanguage returns an attribute KeyValue conforming to the
// "browser.language" semantic conventions. It represents the preferred language
// of the user using the browser.
func BrowserLanguage(val string) attribute.KeyValue {
	return BrowserLanguageKey.String(val)
}

// BrowserMobile returns an attribute KeyValue conforming to the "browser.mobile"
// semantic conventions. It represents a boolean that is true if the browser is
// running on a mobile device.
func BrowserMobile(val bool) attribute.KeyValue {
	return BrowserMobileKey.Bool(val)
}

// BrowserPlatform returns an attribute KeyValue conforming to the
// "browser.platform" semantic conventions. It represents the platform on which
// the browser is running.
func BrowserPlatform(val string) attribute.KeyValue {
	return BrowserPlatformKey.String(val)
}

// Namespace: cassandra
const (
	// CassandraConsistencyLevelKey is the attribute Key conforming to the
	// "cassandra.consistency.level" semantic conventions. It represents the
	// consistency level of the query. Based on consistency values from [CQL].
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	//
	// [CQL]: https://docs.datastax.com/en/cassandra-oss/3.0/cassandra/dml/dmlConfigConsistency.html
	CassandraConsistencyLevelKey = attribute.Key("cassandra.consistency.level")

	// CassandraCoordinatorDCKey is the attribute Key conforming to the
	// "cassandra.coordinator.dc" semantic conventions. It represents the data
	// center of the coordinating node for a query.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: us-west-2
	CassandraCoordinatorDCKey = attribute.Key("cassandra.coordinator.dc")

	// CassandraCoordinatorIDKey is the attribute Key conforming to the
	// "cassandra.coordinator.id" semantic conventions. It represents the ID of the
	// coordinating node for a query.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: be13faa2-8574-4d71-926d-27f16cf8a7af
	CassandraCoordinatorIDKey = attribute.Key("cassandra.coordinator.id")

	// CassandraPageSizeKey is the attribute Key conforming to the
	// "cassandra.page.size" semantic conventions. It represents the fetch size used
	// for paging, i.e. how many rows will be returned at once.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 5000
	CassandraPageSizeKey = attribute.Key("cassandra.page.size")

	// CassandraQueryIdempotentKey is the attribute Key conforming to the
	// "cassandra.query.idempotent" semantic conventions. It represents the whether
	// or not the query is idempotent.
	//
	// Type: boolean
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	CassandraQueryIdempotentKey = attribute.Key("cassandra.query.idempotent")

	// CassandraSpeculativeExecutionCountKey is the attribute Key conforming to the
	// "cassandra.speculative_execution.count" semantic conventions. It represents
	// the number of times a query was speculatively executed. Not set or `0` if the
	// query was not executed speculatively.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 0, 2
	CassandraSpeculativeExecutionCountKey = attribute.Key("cassandra.speculative_execution.count")
)

// CassandraCoordinatorDC returns an attribute KeyValue conforming to the
// "cassandra.coordinator.dc" semantic conventions. It represents the data center
// of the coordinating node for a query.
func CassandraCoordinatorDC(val string) attribute.KeyValue {
	return CassandraCoordinatorDCKey.String(val)
}

// CassandraCoordinatorID returns an attribute KeyValue conforming to the
// "cassandra.coordinator.id" semantic conventions. It represents the ID of the
// coordinating node for a query.
func CassandraCoordinatorID(val string) attribute.KeyValue {
	return CassandraCoordinatorIDKey.String(val)
}

// CassandraPageSize returns an attribute KeyValue conforming to the
// "cassandra.page.size" semantic conventions. It represents the fetch size used
// for paging, i.e. how many rows will be returned at once.
func CassandraPageSize(val int) attribute.KeyValue {
	return CassandraPageSizeKey.Int(val)
}

// CassandraQueryIdempotent returns an attribute KeyValue conforming to the
// "cassandra.query.idempotent" semantic conventions. It represents the whether
// or not the query is idempotent.
func CassandraQueryIdempotent(val bool) attribute.KeyValue {
	return CassandraQueryIdempotentKey.Bool(val)
}

// CassandraSpeculativeExecutionCount returns an attribute KeyValue conforming to
// the "cassandra.speculative_execution.count" semantic conventions. It
// represents the number of times a query was speculatively executed. Not set or
// `0` if the query was not executed speculatively.
func CassandraSpeculativeExecutionCount(val int) attribute.KeyValue {
	return CassandraSpeculativeExecutionCountKey.Int(val)
}

// Enum values for cassandra.consistency.level
var (
	// All
	// Stability: development
	CassandraConsistencyLevelAll = CassandraConsistencyLevelKey.String("all")
	// Each Quorum
	// Stability: development
	CassandraConsistencyLevelEachQuorum = CassandraConsistencyLevelKey.String("each_quorum")
	// Quorum
	// Stability: development
	CassandraConsistencyLevelQuorum = CassandraConsistencyLevelKey.String("quorum")
	// Local Quorum
	// Stability: development
	CassandraConsistencyLevelLocalQuorum = CassandraConsistencyLevelKey.String("local_quorum")
	// One
	// Stability: development
	CassandraConsistencyLevelOne = CassandraConsistencyLevelKey.String("one")
	// Two
	// Stability: development
	CassandraConsistencyLevelTwo = CassandraConsistencyLevelKey.String("two")
	// Three
	// Stability: development
	CassandraConsistencyLevelThree = CassandraConsistencyLevelKey.String("three")
	// Local One
	// Stability: development
	CassandraConsistencyLevelLocalOne = CassandraConsistencyLevelKey.String("local_one")
	// Any
	// Stability: development
	CassandraConsistencyLevelAny = CassandraConsistencyLevelKey.String("any")
	// Serial
	// Stability: development
	CassandraConsistencyLevelSerial = CassandraConsistencyLevelKey.String("serial")
	// Local Serial
	// Stability: development
	CassandraConsistencyLevelLocalSerial = CassandraConsistencyLevelKey.String("local_serial")
)

// Namespace: cicd
const (
	// CICDPipelineActionNameKey is the attribute Key conforming to the
	// "cicd.pipeline.action.name" semantic conventions. It represents the kind of
	// action a pipeline run is performing.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "BUILD", "RUN", "SYNC"
	CICDPipelineActionNameKey = attribute.Key("cicd.pipeline.action.name")

	// CICDPipelineNameKey is the attribute Key conforming to the
	// "cicd.pipeline.name" semantic conventions. It represents the human readable
	// name of the pipeline within a CI/CD system.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Build and Test", "Lint", "Deploy Go Project",
	// "deploy_to_environment"
	CICDPipelineNameKey = attribute.Key("cicd.pipeline.name")

	// CICDPipelineResultKey is the attribute Key conforming to the
	// "cicd.pipeline.result" semantic conventions. It represents the result of a
	// pipeline run.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "success", "failure", "timeout", "skipped"
	CICDPipelineResultKey = attribute.Key("cicd.pipeline.result")

	// CICDPipelineRunIDKey is the attribute Key conforming to the
	// "cicd.pipeline.run.id" semantic conventions. It represents the unique
	// identifier of a pipeline run within a CI/CD system.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "120912"
	CICDPipelineRunIDKey = attribute.Key("cicd.pipeline.run.id")

	// CICDPipelineRunStateKey is the attribute Key conforming to the
	// "cicd.pipeline.run.state" semantic conventions. It represents the pipeline
	// run goes through these states during its lifecycle.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "pending", "executing", "finalizing"
	CICDPipelineRunStateKey = attribute.Key("cicd.pipeline.run.state")

	// CICDPipelineRunURLFullKey is the attribute Key conforming to the
	// "cicd.pipeline.run.url.full" semantic conventions. It represents the [URL] of
	// the pipeline run, providing the complete address in order to locate and
	// identify the pipeline run.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// "https://github.com/open-telemetry/semantic-conventions/actions/runs/9753949763?pr=1075"
	//
	// [URL]: https://wikipedia.org/wiki/URL
	CICDPipelineRunURLFullKey = attribute.Key("cicd.pipeline.run.url.full")

	// CICDPipelineTaskNameKey is the attribute Key conforming to the
	// "cicd.pipeline.task.name" semantic conventions. It represents the human
	// readable name of a task within a pipeline. Task here most closely aligns with
	// a [computing process] in a pipeline. Other terms for tasks include commands,
	// steps, and procedures.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Run GoLang Linter", "Go Build", "go-test", "deploy_binary"
	//
	// [computing process]: https://wikipedia.org/wiki/Pipeline_(computing)
	CICDPipelineTaskNameKey = attribute.Key("cicd.pipeline.task.name")

	// CICDPipelineTaskRunIDKey is the attribute Key conforming to the
	// "cicd.pipeline.task.run.id" semantic conventions. It represents the unique
	// identifier of a task run within a pipeline.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "12097"
	CICDPipelineTaskRunIDKey = attribute.Key("cicd.pipeline.task.run.id")

	// CICDPipelineTaskRunResultKey is the attribute Key conforming to the
	// "cicd.pipeline.task.run.result" semantic conventions. It represents the
	// result of a task run.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "success", "failure", "timeout", "skipped"
	CICDPipelineTaskRunResultKey = attribute.Key("cicd.pipeline.task.run.result")

	// CICDPipelineTaskRunURLFullKey is the attribute Key conforming to the
	// "cicd.pipeline.task.run.url.full" semantic conventions. It represents the
	// [URL] of the pipeline task run, providing the complete address in order to
	// locate and identify the pipeline task run.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// "https://github.com/open-telemetry/semantic-conventions/actions/runs/9753949763/job/26920038674?pr=1075"
	//
	// [URL]: https://wikipedia.org/wiki/URL
	CICDPipelineTaskRunURLFullKey = attribute.Key("cicd.pipeline.task.run.url.full")

	// CICDPipelineTaskTypeKey is the attribute Key conforming to the
	// "cicd.pipeline.task.type" semantic conventions. It represents the type of the
	// task within a pipeline.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "build", "test", "deploy"
	CICDPipelineTaskTypeKey = attribute.Key("cicd.pipeline.task.type")

	// CICDSystemComponentKey is the attribute Key conforming to the
	// "cicd.system.component" semantic conventions. It represents the name of a
	// component of the CICD system.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "controller", "scheduler", "agent"
	CICDSystemComponentKey = attribute.Key("cicd.system.component")

	// CICDWorkerIDKey is the attribute Key conforming to the "cicd.worker.id"
	// semantic conventions. It represents the unique identifier of a worker within
	// a CICD system.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "abc123", "10.0.1.2", "controller"
	CICDWorkerIDKey = attribute.Key("cicd.worker.id")

	// CICDWorkerNameKey is the attribute Key conforming to the "cicd.worker.name"
	// semantic conventions. It represents the name of a worker within a CICD
	// system.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "agent-abc", "controller", "Ubuntu LTS"
	CICDWorkerNameKey = attribute.Key("cicd.worker.name")

	// CICDWorkerStateKey is the attribute Key conforming to the "cicd.worker.state"
	// semantic conventions. It represents the state of a CICD worker / agent.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "idle", "busy", "down"
	CICDWorkerStateKey = attribute.Key("cicd.worker.state")

	// CICDWorkerURLFullKey is the attribute Key conforming to the
	// "cicd.worker.url.full" semantic conventions. It represents the [URL] of the
	// worker, providing the complete address in order to locate and identify the
	// worker.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "https://cicd.example.org/worker/abc123"
	//
	// [URL]: https://wikipedia.org/wiki/URL
	CICDWorkerURLFullKey = attribute.Key("cicd.worker.url.full")
)

// CICDPipelineName returns an attribute KeyValue conforming to the
// "cicd.pipeline.name" semantic conventions. It represents the human readable
// name of the pipeline within a CI/CD system.
func CICDPipelineName(val string) attribute.KeyValue {
	return CICDPipelineNameKey.String(val)
}

// CICDPipelineRunID returns an attribute KeyValue conforming to the
// "cicd.pipeline.run.id" semantic conventions. It represents the unique
// identifier of a pipeline run within a CI/CD system.
func CICDPipelineRunID(val string) attribute.KeyValue {
	return CICDPipelineRunIDKey.String(val)
}

// CICDPipelineRunURLFull returns an attribute KeyValue conforming to the
// "cicd.pipeline.run.url.full" semantic conventions. It represents the [URL] of
// the pipeline run, providing the complete address in order to locate and
// identify the pipeline run.
//
// [URL]: https://wikipedia.org/wiki/URL
func CICDPipelineRunURLFull(val string) attribute.KeyValue {
	return CICDPipelineRunURLFullKey.String(val)
}

// CICDPipelineTaskName returns an attribute KeyValue conforming to the
// "cicd.pipeline.task.name" semantic conventions. It represents the human
// readable name of a task within a pipeline. Task here most closely aligns with
// a [computing process] in a pipeline. Other terms for tasks include commands,
// steps, and procedures.
//
// [computing process]: https://wikipedia.org/wiki/Pipeline_(computing)
func CICDPipelineTaskName(val string) attribute.KeyValue {
	return CICDPipelineTaskNameKey.String(val)
}

// CICDPipelineTaskRunID returns an attribute KeyValue conforming to the
// "cicd.pipeline.task.run.id" semantic conventions. It represents the unique
// identifier of a task run within a pipeline.
func CICDPipelineTaskRunID(val string) attribute.KeyValue {
	return CICDPipelineTaskRunIDKey.String(val)
}

// CICDPipelineTaskRunURLFull returns an attribute KeyValue conforming to the
// "cicd.pipeline.task.run.url.full" semantic conventions. It represents the
// [URL] of the pipeline task run, providing the complete address in order to
// locate and identify the pipeline task run.
//
// [URL]: https://wikipedia.org/wiki/URL
func CICDPipelineTaskRunURLFull(val string) attribute.KeyValue {
	return CICDPipelineTaskRunURLFullKey.String(val)
}

// CICDSystemComponent returns an attribute KeyValue conforming to the
// "cicd.system.component" semantic conventions. It represents the name of a
// component of the CICD system.
func CICDSystemComponent(val string) attribute.KeyValue {
	return CICDSystemComponentKey.String(val)
}

// CICDWorkerID returns an attribute KeyValue conforming to the "cicd.worker.id"
// semantic conventions. It represents the unique identifier of a worker within a
// CICD system.
func CICDWorkerID(val string) attribute.KeyValue {
	return CICDWorkerIDKey.String(val)
}

// CICDWorkerName returns an attribute KeyValue conforming to the
// "cicd.worker.name" semantic conventions. It represents the name of a worker
// within a CICD system.
func CICDWorkerName(val string) attribute.KeyValue {
	return CICDWorkerNameKey.String(val)
}

// CICDWorkerURLFull returns an attribute KeyValue conforming to the
// "cicd.worker.url.full" semantic conventions. It represents the [URL] of the
// worker, providing the complete address in order to locate and identify the
// worker.
//
// [URL]: https://wikipedia.org/wiki/URL
func CICDWorkerURLFull(val string) attribute.KeyValue {
	return CICDWorkerURLFullKey.String(val)
}

// Enum values for cicd.pipeline.action.name
var (
	// The pipeline run is executing a build.
	// Stability: development
	CICDPipelineActionNameBuild = CICDPipelineActionNameKey.String("BUILD")
	// The pipeline run is executing.
	// Stability: development
	CICDPipelineActionNameRun = CICDPipelineActionNameKey.String("RUN")
	// The pipeline run is executing a sync.
	// Stability: development
	CICDPipelineActionNameSync = CICDPipelineActionNameKey.String("SYNC")
)

// Enum values for cicd.pipeline.result
var (
	// The pipeline run finished successfully.
	// Stability: development
	CICDPipelineResultSuccess = CICDPipelineResultKey.String("success")
	// The pipeline run did not finish successfully, eg. due to a compile error or a
	// failing test. Such failures are usually detected by non-zero exit codes of
	// the tools executed in the pipeline run.
	// Stability: development
	CICDPipelineResultFailure = CICDPipelineResultKey.String("failure")
	// The pipeline run failed due to an error in the CICD system, eg. due to the
	// worker being killed.
	// Stability: development
	CICDPipelineResultError = CICDPipelineResultKey.String("error")
	// A timeout caused the pipeline run to be interrupted.
	// Stability: development
	CICDPipelineResultTimeout = CICDPipelineResultKey.String("timeout")
	// The pipeline run was cancelled, eg. by a user manually cancelling the
	// pipeline run.
	// Stability: development
	CICDPipelineResultCancellation = CICDPipelineResultKey.String("cancellation")
	// The pipeline run was skipped, eg. due to a precondition not being met.
	// Stability: development
	CICDPipelineResultSkip = CICDPipelineResultKey.String("skip")
)

// Enum values for cicd.pipeline.run.state
var (
	// The run pending state spans from the event triggering the pipeline run until
	// the execution of the run starts (eg. time spent in a queue, provisioning
	// agents, creating run resources).
	//
	// Stability: development
	CICDPipelineRunStatePending = CICDPipelineRunStateKey.String("pending")
	// The executing state spans the execution of any run tasks (eg. build, test).
	// Stability: development
	CICDPipelineRunStateExecuting = CICDPipelineRunStateKey.String("executing")
	// The finalizing state spans from when the run has finished executing (eg.
	// cleanup of run resources).
	// Stability: development
	CICDPipelineRunStateFinalizing = CICDPipelineRunStateKey.String("finalizing")
)

// Enum values for cicd.pipeline.task.run.result
var (
	// The task run finished successfully.
	// Stability: development
	CICDPipelineTaskRunResultSuccess = CICDPipelineTaskRunResultKey.String("success")
	// The task run did not finish successfully, eg. due to a compile error or a
	// failing test. Such failures are usually detected by non-zero exit codes of
	// the tools executed in the task run.
	// Stability: development
	CICDPipelineTaskRunResultFailure = CICDPipelineTaskRunResultKey.String("failure")
	// The task run failed due to an error in the CICD system, eg. due to the worker
	// being killed.
	// Stability: development
	CICDPipelineTaskRunResultError = CICDPipelineTaskRunResultKey.String("error")
	// A timeout caused the task run to be interrupted.
	// Stability: development
	CICDPipelineTaskRunResultTimeout = CICDPipelineTaskRunResultKey.String("timeout")
	// The task run was cancelled, eg. by a user manually cancelling the task run.
	// Stability: development
	CICDPipelineTaskRunResultCancellation = CICDPipelineTaskRunResultKey.String("cancellation")
	// The task run was skipped, eg. due to a precondition not being met.
	// Stability: development
	CICDPipelineTaskRunResultSkip = CICDPipelineTaskRunResultKey.String("skip")
)

// Enum values for cicd.pipeline.task.type
var (
	// build
	// Stability: development
	CICDPipelineTaskTypeBuild = CICDPipelineTaskTypeKey.String("build")
	// test
	// Stability: development
	CICDPipelineTaskTypeTest = CICDPipelineTaskTypeKey.String("test")
	// deploy
	// Stability: development
	CICDPipelineTaskTypeDeploy = CICDPipelineTaskTypeKey.String("deploy")
)

// Enum values for cicd.worker.state
var (
	// The worker is not performing work for the CICD system. It is available to the
	// CICD system to perform work on (online / idle).
	// Stability: development
	CICDWorkerStateAvailable = CICDWorkerStateKey.String("available")
	// The worker is performing work for the CICD system.
	// Stability: development
	CICDWorkerStateBusy = CICDWorkerStateKey.String("busy")
	// The worker is not available to the CICD system (disconnected / down).
	// Stability: development
	CICDWorkerStateOffline = CICDWorkerStateKey.String("offline")
)

// Namespace: client
const (
	// ClientAddressKey is the attribute Key conforming to the "client.address"
	// semantic conventions. It represents the client address - domain name if
	// available without reverse DNS lookup; otherwise, IP address or Unix domain
	// socket name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "client.example.com", "10.1.2.80", "/tmp/my.sock"
	// Note: When observed from the server side, and when communicating through an
	// intermediary, `client.address` SHOULD represent the client address behind any
	// intermediaries, for example proxies, if it's available.
	ClientAddressKey = attribute.Key("client.address")

	// ClientPortKey is the attribute Key conforming to the "client.port" semantic
	// conventions. It represents the client port number.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: 65123
	// Note: When observed from the server side, and when communicating through an
	// intermediary, `client.port` SHOULD represent the client port behind any
	// intermediaries, for example proxies, if it's available.
	ClientPortKey = attribute.Key("client.port")
)

// ClientAddress returns an attribute KeyValue conforming to the "client.address"
// semantic conventions. It represents the client address - domain name if
// available without reverse DNS lookup; otherwise, IP address or Unix domain
// socket name.
func ClientAddress(val string) attribute.KeyValue {
	return ClientAddressKey.String(val)
}

// ClientPort returns an attribute KeyValue conforming to the "client.port"
// semantic conventions. It represents the client port number.
func ClientPort(val int) attribute.KeyValue {
	return ClientPortKey.Int(val)
}

// Namespace: cloud
const (
	// CloudAccountIDKey is the attribute Key conforming to the "cloud.account.id"
	// semantic conventions. It represents the cloud account ID the resource is
	// assigned to.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "111111111111", "opentelemetry"
	CloudAccountIDKey = attribute.Key("cloud.account.id")

	// CloudAvailabilityZoneKey is the attribute Key conforming to the
	// "cloud.availability_zone" semantic conventions. It represents the cloud
	// regions often have multiple, isolated locations known as zones to increase
	// availability. Availability zone represents the zone where the resource is
	// running.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "us-east-1c"
	// Note: Availability zones are called "zones" on Alibaba Cloud and Google
	// Cloud.
	CloudAvailabilityZoneKey = attribute.Key("cloud.availability_zone")

	// CloudPlatformKey is the attribute Key conforming to the "cloud.platform"
	// semantic conventions. It represents the cloud platform in use.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: The prefix of the service SHOULD match the one specified in
	// `cloud.provider`.
	CloudPlatformKey = attribute.Key("cloud.platform")

	// CloudProviderKey is the attribute Key conforming to the "cloud.provider"
	// semantic conventions. It represents the name of the cloud provider.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	CloudProviderKey = attribute.Key("cloud.provider")

	// CloudRegionKey is the attribute Key conforming to the "cloud.region" semantic
	// conventions. It represents the geographical region within a cloud provider.
	// When associated with a resource, this attribute specifies the region where
	// the resource operates. When calling services or APIs deployed on a cloud,
	// this attribute identifies the region where the called destination is
	// deployed.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "us-central1", "us-east-1"
	// Note: Refer to your provider's docs to see the available regions, for example
	// [Alibaba Cloud regions], [AWS regions], [Azure regions],
	// [Google Cloud regions], or [Tencent Cloud regions].
	//
	// [Alibaba Cloud regions]: https://www.alibabacloud.com/help/doc-detail/40654.htm
	// [AWS regions]: https://aws.amazon.com/about-aws/global-infrastructure/regions_az/
	// [Azure regions]: https://azure.microsoft.com/global-infrastructure/geographies/
	// [Google Cloud regions]: https://cloud.google.com/about/locations
	// [Tencent Cloud regions]: https://www.tencentcloud.com/document/product/213/6091
	CloudRegionKey = attribute.Key("cloud.region")

	// CloudResourceIDKey is the attribute Key conforming to the "cloud.resource_id"
	// semantic conventions. It represents the cloud provider-specific native
	// identifier of the monitored cloud resource (e.g. an [ARN] on AWS, a
	// [fully qualified resource ID] on Azure, a [full resource name] on GCP).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "arn:aws:lambda:REGION:ACCOUNT_ID:function:my-function",
	// "//run.googleapis.com/projects/PROJECT_ID/locations/LOCATION_ID/services/SERVICE_ID",
	// "/subscriptions/<SUBSCRIPTION_GUID>/resourceGroups/<RG>
	// /providers/Microsoft.Web/sites/<FUNCAPP>/functions/<FUNC>"
	// Note: On some cloud providers, it may not be possible to determine the full
	// ID at startup,
	// so it may be necessary to set `cloud.resource_id` as a span attribute
	// instead.
	//
	// The exact value to use for `cloud.resource_id` depends on the cloud provider.
	// The following well-known definitions MUST be used if you set this attribute
	// and they apply:
	//
	//   - **AWS Lambda:** The function [ARN].
	//     Take care not to use the "invoked ARN" directly but replace any
	//     [alias suffix]
	//     with the resolved function version, as the same runtime instance may be
	//     invocable with
	//     multiple different aliases.
	//   - **GCP:** The [URI of the resource]
	//   - **Azure:** The [Fully Qualified Resource ID] of the invoked function,
	//     *not* the function app, having the form
	//
	//     `/subscriptions/<SUBSCRIPTION_GUID>/resourceGroups/<RG>/providers/Microsoft.Web/sites/<FUNCAPP>/functions/<FUNC>`
	//     .
	//     This means that a span attribute MUST be used, as an Azure function app
	//     can host multiple functions that would usually share
	//     a TracerProvider.
	//
	//
	// [ARN]: https://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html
	// [fully qualified resource ID]: https://learn.microsoft.com/rest/api/resources/resources/get-by-id
	// [full resource name]: https://google.aip.dev/122#full-resource-names
	// [ARN]: https://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html
	// [alias suffix]: https://docs.aws.amazon.com/lambda/latest/dg/configuration-aliases.html
	// [URI of the resource]: https://cloud.google.com/iam/docs/full-resource-names
	// [Fully Qualified Resource ID]: https://learn.microsoft.com/rest/api/resources/resources/get-by-id
	CloudResourceIDKey = attribute.Key("cloud.resource_id")
)

// CloudAccountID returns an attribute KeyValue conforming to the
// "cloud.account.id" semantic conventions. It represents the cloud account ID
// the resource is assigned to.
func CloudAccountID(val string) attribute.KeyValue {
	return CloudAccountIDKey.String(val)
}

// CloudAvailabilityZone returns an attribute KeyValue conforming to the
// "cloud.availability_zone" semantic conventions. It represents the cloud
// regions often have multiple, isolated locations known as zones to increase
// availability. Availability zone represents the zone where the resource is
// running.
func CloudAvailabilityZone(val string) attribute.KeyValue {
	return CloudAvailabilityZoneKey.String(val)
}

// CloudRegion returns an attribute KeyValue conforming to the "cloud.region"
// semantic conventions. It represents the geographical region within a cloud
// provider. When associated with a resource, this attribute specifies the region
// where the resource operates. When calling services or APIs deployed on a
// cloud, this attribute identifies the region where the called destination is
// deployed.
func CloudRegion(val string) attribute.KeyValue {
	return CloudRegionKey.String(val)
}

// CloudResourceID returns an attribute KeyValue conforming to the
// "cloud.resource_id" semantic conventions. It represents the cloud
// provider-specific native identifier of the monitored cloud resource (e.g. an
// [ARN] on AWS, a [fully qualified resource ID] on Azure, a [full resource name]
//  on GCP).
//
// [ARN]: https://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html
// [fully qualified resource ID]: https://learn.microsoft.com/rest/api/resources/resources/get-by-id
// [full resource name]: https://google.aip.dev/122#full-resource-names
func CloudResourceID(val string) attribute.KeyValue {
	return CloudResourceIDKey.String(val)
}

// Enum values for cloud.platform
var (
	// Alibaba Cloud Elastic Compute Service
	// Stability: development
	CloudPlatformAlibabaCloudECS = CloudPlatformKey.String("alibaba_cloud_ecs")
	// Alibaba Cloud Function Compute
	// Stability: development
	CloudPlatformAlibabaCloudFC = CloudPlatformKey.String("alibaba_cloud_fc")
	// Red Hat OpenShift on Alibaba Cloud
	// Stability: development
	CloudPlatformAlibabaCloudOpenShift = CloudPlatformKey.String("alibaba_cloud_openshift")
	// AWS Elastic Compute Cloud
	// Stability: development
	CloudPlatformAWSEC2 = CloudPlatformKey.String("aws_ec2")
	// AWS Elastic Container Service
	// Stability: development
	CloudPlatformAWSECS = CloudPlatformKey.String("aws_ecs")
	// AWS Elastic Kubernetes Service
	// Stability: development
	CloudPlatformAWSEKS = CloudPlatformKey.String("aws_eks")
	// AWS Lambda
	// Stability: development
	CloudPlatformAWSLambda = CloudPlatformKey.String("aws_lambda")
	// AWS Elastic Beanstalk
	// Stability: development
	CloudPlatformAWSElasticBeanstalk = CloudPlatformKey.String("aws_elastic_beanstalk")
	// AWS App Runner
	// Stability: development
	CloudPlatformAWSAppRunner = CloudPlatformKey.String("aws_app_runner")
	// Red Hat OpenShift on AWS (ROSA)
	// Stability: development
	CloudPlatformAWSOpenShift = CloudPlatformKey.String("aws_openshift")
	// Azure Virtual Machines
	// Stability: development
	CloudPlatformAzureVM = CloudPlatformKey.String("azure.vm")
	// Azure Container Apps
	// Stability: development
	CloudPlatformAzureContainerApps = CloudPlatformKey.String("azure.container_apps")
	// Azure Container Instances
	// Stability: development
	CloudPlatformAzureContainerInstances = CloudPlatformKey.String("azure.container_instances")
	// Azure Kubernetes Service
	// Stability: development
	CloudPlatformAzureAKS = CloudPlatformKey.String("azure.aks")
	// Azure Functions
	// Stability: development
	CloudPlatformAzureFunctions = CloudPlatformKey.String("azure.functions")
	// Azure App Service
	// Stability: development
	CloudPlatformAzureAppService = CloudPlatformKey.String("azure.app_service")
	// Azure Red Hat OpenShift
	// Stability: development
	CloudPlatformAzureOpenShift = CloudPlatformKey.String("azure.openshift")
	// Google Bare Metal Solution (BMS)
	// Stability: development
	CloudPlatformGCPBareMetalSolution = CloudPlatformKey.String("gcp_bare_metal_solution")
	// Google Cloud Compute Engine (GCE)
	// Stability: development
	CloudPlatformGCPComputeEngine = CloudPlatformKey.String("gcp_compute_engine")
	// Google Cloud Run
	// Stability: development
	CloudPlatformGCPCloudRun = CloudPlatformKey.String("gcp_cloud_run")
	// Google Cloud Kubernetes Engine (GKE)
	// Stability: development
	CloudPlatformGCPKubernetesEngine = CloudPlatformKey.String("gcp_kubernetes_engine")
	// Google Cloud Functions (GCF)
	// Stability: development
	CloudPlatformGCPCloudFunctions = CloudPlatformKey.String("gcp_cloud_functions")
	// Google Cloud App Engine (GAE)
	// Stability: development
	CloudPlatformGCPAppEngine = CloudPlatformKey.String("gcp_app_engine")
	// Red Hat OpenShift on Google Cloud
	// Stability: development
	CloudPlatformGCPOpenShift = CloudPlatformKey.String("gcp_openshift")
	// Red Hat OpenShift on IBM Cloud
	// Stability: development
	CloudPlatformIBMCloudOpenShift = CloudPlatformKey.String("ibm_cloud_openshift")
	// Compute on Oracle Cloud Infrastructure (OCI)
	// Stability: development
	CloudPlatformOracleCloudCompute = CloudPlatformKey.String("oracle_cloud_compute")
	// Kubernetes Engine (OKE) on Oracle Cloud Infrastructure (OCI)
	// Stability: development
	CloudPlatformOracleCloudOKE = CloudPlatformKey.String("oracle_cloud_oke")
	// Tencent Cloud Cloud Virtual Machine (CVM)
	// Stability: development
	CloudPlatformTencentCloudCVM = CloudPlatformKey.String("tencent_cloud_cvm")
	// Tencent Cloud Elastic Kubernetes Service (EKS)
	// Stability: development
	CloudPlatformTencentCloudEKS = CloudPlatformKey.String("tencent_cloud_eks")
	// Tencent Cloud Serverless Cloud Function (SCF)
	// Stability: development
	CloudPlatformTencentCloudSCF = CloudPlatformKey.String("tencent_cloud_scf")
)

// Enum values for cloud.provider
var (
	// Alibaba Cloud
	// Stability: development
	CloudProviderAlibabaCloud = CloudProviderKey.String("alibaba_cloud")
	// Amazon Web Services
	// Stability: development
	CloudProviderAWS = CloudProviderKey.String("aws")
	// Microsoft Azure
	// Stability: development
	CloudProviderAzure = CloudProviderKey.String("azure")
	// Google Cloud Platform
	// Stability: development
	CloudProviderGCP = CloudProviderKey.String("gcp")
	// Heroku Platform as a Service
	// Stability: development
	CloudProviderHeroku = CloudProviderKey.String("heroku")
	// IBM Cloud
	// Stability: development
	CloudProviderIBMCloud = CloudProviderKey.String("ibm_cloud")
	// Oracle Cloud Infrastructure (OCI)
	// Stability: development
	CloudProviderOracleCloud = CloudProviderKey.String("oracle_cloud")
	// Tencent Cloud
	// Stability: development
	CloudProviderTencentCloud = CloudProviderKey.String("tencent_cloud")
)

// Namespace: cloudevents
const (
	// CloudEventsEventIDKey is the attribute Key conforming to the
	// "cloudevents.event_id" semantic conventions. It represents the [event_id]
	// uniquely identifies the event.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "123e4567-e89b-12d3-a456-426614174000", "0001"
	//
	// [event_id]: https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#id
	CloudEventsEventIDKey = attribute.Key("cloudevents.event_id")

	// CloudEventsEventSourceKey is the attribute Key conforming to the
	// "cloudevents.event_source" semantic conventions. It represents the [source]
	// identifies the context in which an event happened.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "https://github.com/cloudevents", "/cloudevents/spec/pull/123",
	// "my-service"
	//
	// [source]: https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#source-1
	CloudEventsEventSourceKey = attribute.Key("cloudevents.event_source")

	// CloudEventsEventSpecVersionKey is the attribute Key conforming to the
	// "cloudevents.event_spec_version" semantic conventions. It represents the
	// [version of the CloudEvents specification] which the event uses.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1.0
	//
	// [version of the CloudEvents specification]: https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#specversion
	CloudEventsEventSpecVersionKey = attribute.Key("cloudevents.event_spec_version")

	// CloudEventsEventSubjectKey is the attribute Key conforming to the
	// "cloudevents.event_subject" semantic conventions. It represents the [subject]
	//  of the event in the context of the event producer (identified by source).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: mynewfile.jpg
	//
	// [subject]: https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#subject
	CloudEventsEventSubjectKey = attribute.Key("cloudevents.event_subject")

	// CloudEventsEventTypeKey is the attribute Key conforming to the
	// "cloudevents.event_type" semantic conventions. It represents the [event_type]
	//  contains a value describing the type of event related to the originating
	// occurrence.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "com.github.pull_request.opened", "com.example.object.deleted.v2"
	//
	// [event_type]: https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#type
	CloudEventsEventTypeKey = attribute.Key("cloudevents.event_type")
)

// CloudEventsEventID returns an attribute KeyValue conforming to the
// "cloudevents.event_id" semantic conventions. It represents the [event_id]
// uniquely identifies the event.
//
// [event_id]: https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#id
func CloudEventsEventID(val string) attribute.KeyValue {
	return CloudEventsEventIDKey.String(val)
}

// CloudEventsEventSource returns an attribute KeyValue conforming to the
// "cloudevents.event_source" semantic conventions. It represents the [source]
// identifies the context in which an event happened.
//
// [source]: https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#source-1
func CloudEventsEventSource(val string) attribute.KeyValue {
	return CloudEventsEventSourceKey.String(val)
}

// CloudEventsEventSpecVersion returns an attribute KeyValue conforming to the
// "cloudevents.event_spec_version" semantic conventions. It represents the
// [version of the CloudEvents specification] which the event uses.
//
// [version of the CloudEvents specification]: https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#specversion
func CloudEventsEventSpecVersion(val string) attribute.KeyValue {
	return CloudEventsEventSpecVersionKey.String(val)
}

// CloudEventsEventSubject returns an attribute KeyValue conforming to the
// "cloudevents.event_subject" semantic conventions. It represents the [subject]
// of the event in the context of the event producer (identified by source).
//
// [subject]: https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#subject
func CloudEventsEventSubject(val string) attribute.KeyValue {
	return CloudEventsEventSubjectKey.String(val)
}

// CloudEventsEventType returns an attribute KeyValue conforming to the
// "cloudevents.event_type" semantic conventions. It represents the [event_type]
// contains a value describing the type of event related to the originating
// occurrence.
//
// [event_type]: https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md#type
func CloudEventsEventType(val string) attribute.KeyValue {
	return CloudEventsEventTypeKey.String(val)
}

// Namespace: cloudfoundry
const (
	// CloudFoundryAppIDKey is the attribute Key conforming to the
	// "cloudfoundry.app.id" semantic conventions. It represents the guid of the
	// application.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "218fc5a9-a5f1-4b54-aa05-46717d0ab26d"
	// Note: Application instrumentation should use the value from environment
	// variable `VCAP_APPLICATION.application_id`. This is the same value as
	// reported by `cf app <app-name> --guid`.
	CloudFoundryAppIDKey = attribute.Key("cloudfoundry.app.id")

	// CloudFoundryAppInstanceIDKey is the attribute Key conforming to the
	// "cloudfoundry.app.instance.id" semantic conventions. It represents the index
	// of the application instance. 0 when just one instance is active.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "0", "1"
	// Note: CloudFoundry defines the `instance_id` in the [Loggregator v2 envelope]
	// .
	// It is used for logs and metrics emitted by CloudFoundry. It is
	// supposed to contain the application instance index for applications
	// deployed on the runtime.
	//
	// Application instrumentation should use the value from environment
	// variable `CF_INSTANCE_INDEX`.
	//
	// [Loggregator v2 envelope]: https://github.com/cloudfoundry/loggregator-api#v2-envelope
	CloudFoundryAppInstanceIDKey = attribute.Key("cloudfoundry.app.instance.id")

	// CloudFoundryAppNameKey is the attribute Key conforming to the
	// "cloudfoundry.app.name" semantic conventions. It represents the name of the
	// application.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "my-app-name"
	// Note: Application instrumentation should use the value from environment
	// variable `VCAP_APPLICATION.application_name`. This is the same value
	// as reported by `cf apps`.
	CloudFoundryAppNameKey = attribute.Key("cloudfoundry.app.name")

	// CloudFoundryOrgIDKey is the attribute Key conforming to the
	// "cloudfoundry.org.id" semantic conventions. It represents the guid of the
	// CloudFoundry org the application is running in.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "218fc5a9-a5f1-4b54-aa05-46717d0ab26d"
	// Note: Application instrumentation should use the value from environment
	// variable `VCAP_APPLICATION.org_id`. This is the same value as
	// reported by `cf org <org-name> --guid`.
	CloudFoundryOrgIDKey = attribute.Key("cloudfoundry.org.id")

	// CloudFoundryOrgNameKey is the attribute Key conforming to the
	// "cloudfoundry.org.name" semantic conventions. It represents the name of the
	// CloudFoundry organization the app is running in.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "my-org-name"
	// Note: Application instrumentation should use the value from environment
	// variable `VCAP_APPLICATION.org_name`. This is the same value as
	// reported by `cf orgs`.
	CloudFoundryOrgNameKey = attribute.Key("cloudfoundry.org.name")

	// CloudFoundryProcessIDKey is the attribute Key conforming to the
	// "cloudfoundry.process.id" semantic conventions. It represents the UID
	// identifying the process.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "218fc5a9-a5f1-4b54-aa05-46717d0ab26d"
	// Note: Application instrumentation should use the value from environment
	// variable `VCAP_APPLICATION.process_id`. It is supposed to be equal to
	// `VCAP_APPLICATION.app_id` for applications deployed to the runtime.
	// For system components, this could be the actual PID.
	CloudFoundryProcessIDKey = attribute.Key("cloudfoundry.process.id")

	// CloudFoundryProcessTypeKey is the attribute Key conforming to the
	// "cloudfoundry.process.type" semantic conventions. It represents the type of
	// process.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "web"
	// Note: CloudFoundry applications can consist of multiple jobs. Usually the
	// main process will be of type `web`. There can be additional background
	// tasks or side-cars with different process types.
	CloudFoundryProcessTypeKey = attribute.Key("cloudfoundry.process.type")

	// CloudFoundrySpaceIDKey is the attribute Key conforming to the
	// "cloudfoundry.space.id" semantic conventions. It represents the guid of the
	// CloudFoundry space the application is running in.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "218fc5a9-a5f1-4b54-aa05-46717d0ab26d"
	// Note: Application instrumentation should use the value from environment
	// variable `VCAP_APPLICATION.space_id`. This is the same value as
	// reported by `cf space <space-name> --guid`.
	CloudFoundrySpaceIDKey = attribute.Key("cloudfoundry.space.id")

	// CloudFoundrySpaceNameKey is the attribute Key conforming to the
	// "cloudfoundry.space.name" semantic conventions. It represents the name of the
	// CloudFoundry space the application is running in.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "my-space-name"
	// Note: Application instrumentation should use the value from environment
	// variable `VCAP_APPLICATION.space_name`. This is the same value as
	// reported by `cf spaces`.
	CloudFoundrySpaceNameKey = attribute.Key("cloudfoundry.space.name")

	// CloudFoundrySystemIDKey is the attribute Key conforming to the
	// "cloudfoundry.system.id" semantic conventions. It represents a guid or
	// another name describing the event source.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "cf/gorouter"
	// Note: CloudFoundry defines the `source_id` in the [Loggregator v2 envelope].
	// It is used for logs and metrics emitted by CloudFoundry. It is
	// supposed to contain the component name, e.g. "gorouter", for
	// CloudFoundry components.
	//
	// When system components are instrumented, values from the
	// [Bosh spec]
	// should be used. The `system.id` should be set to
	// `spec.deployment/spec.name`.
	//
	// [Loggregator v2 envelope]: https://github.com/cloudfoundry/loggregator-api#v2-envelope
	// [Bosh spec]: https://bosh.io/docs/jobs/#properties-spec
	CloudFoundrySystemIDKey = attribute.Key("cloudfoundry.system.id")

	// CloudFoundrySystemInstanceIDKey is the attribute Key conforming to the
	// "cloudfoundry.system.instance.id" semantic conventions. It represents a guid
	// describing the concrete instance of the event source.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "218fc5a9-a5f1-4b54-aa05-46717d0ab26d"
	// Note: CloudFoundry defines the `instance_id` in the [Loggregator v2 envelope]
	// .
	// It is used for logs and metrics emitted by CloudFoundry. It is
	// supposed to contain the vm id for CloudFoundry components.
	//
	// When system components are instrumented, values from the
	// [Bosh spec]
	// should be used. The `system.instance.id` should be set to `spec.id`.
	//
	// [Loggregator v2 envelope]: https://github.com/cloudfoundry/loggregator-api#v2-envelope
	// [Bosh spec]: https://bosh.io/docs/jobs/#properties-spec
	CloudFoundrySystemInstanceIDKey = attribute.Key("cloudfoundry.system.instance.id")
)

// CloudFoundryAppID returns an attribute KeyValue conforming to the
// "cloudfoundry.app.id" semantic conventions. It represents the guid of the
// application.
func CloudFoundryAppID(val string) attribute.KeyValue {
	return CloudFoundryAppIDKey.String(val)
}

// CloudFoundryAppInstanceID returns an attribute KeyValue conforming to the
// "cloudfoundry.app.instance.id" semantic conventions. It represents the index
// of the application instance. 0 when just one instance is active.
func CloudFoundryAppInstanceID(val string) attribute.KeyValue {
	return CloudFoundryAppInstanceIDKey.String(val)
}

// CloudFoundryAppName returns an attribute KeyValue conforming to the
// "cloudfoundry.app.name" semantic conventions. It represents the name of the
// application.
func CloudFoundryAppName(val string) attribute.KeyValue {
	return CloudFoundryAppNameKey.String(val)
}

// CloudFoundryOrgID returns an attribute KeyValue conforming to the
// "cloudfoundry.org.id" semantic conventions. It represents the guid of the
// CloudFoundry org the application is running in.
func CloudFoundryOrgID(val string) attribute.KeyValue {
	return CloudFoundryOrgIDKey.String(val)
}

// CloudFoundryOrgName returns an attribute KeyValue conforming to the
// "cloudfoundry.org.name" semantic conventions. It represents the name of the
// CloudFoundry organization the app is running in.
func CloudFoundryOrgName(val string) attribute.KeyValue {
	return CloudFoundryOrgNameKey.String(val)
}

// CloudFoundryProcessID returns an attribute KeyValue conforming to the
// "cloudfoundry.process.id" semantic conventions. It represents the UID
// identifying the process.
func CloudFoundryProcessID(val string) attribute.KeyValue {
	return CloudFoundryProcessIDKey.String(val)
}

// CloudFoundryProcessType returns an attribute KeyValue conforming to the
// "cloudfoundry.process.type" semantic conventions. It represents the type of
// process.
func CloudFoundryProcessType(val string) attribute.KeyValue {
	return CloudFoundryProcessTypeKey.String(val)
}

// CloudFoundrySpaceID returns an attribute KeyValue conforming to the
// "cloudfoundry.space.id" semantic conventions. It represents the guid of the
// CloudFoundry space the application is running in.
func CloudFoundrySpaceID(val string) attribute.KeyValue {
	return CloudFoundrySpaceIDKey.String(val)
}

// CloudFoundrySpaceName returns an attribute KeyValue conforming to the
// "cloudfoundry.space.name" semantic conventions. It represents the name of the
// CloudFoundry space the application is running in.
func CloudFoundrySpaceName(val string) attribute.KeyValue {
	return CloudFoundrySpaceNameKey.String(val)
}

// CloudFoundrySystemID returns an attribute KeyValue conforming to the
// "cloudfoundry.system.id" semantic conventions. It represents a guid or another
// name describing the event source.
func CloudFoundrySystemID(val string) attribute.KeyValue {
	return CloudFoundrySystemIDKey.String(val)
}

// CloudFoundrySystemInstanceID returns an attribute KeyValue conforming to the
// "cloudfoundry.system.instance.id" semantic conventions. It represents a guid
// describing the concrete instance of the event source.
func CloudFoundrySystemInstanceID(val string) attribute.KeyValue {
	return CloudFoundrySystemInstanceIDKey.String(val)
}

// Namespace: code
const (
	// CodeColumnNumberKey is the attribute Key conforming to the
	// "code.column.number" semantic conventions. It represents the column number in
	// `code.file.path` best representing the operation. It SHOULD point within the
	// code unit named in `code.function.name`. This attribute MUST NOT be used on
	// the Profile signal since the data is already captured in 'message Line'. This
	// constraint is imposed to prevent redundancy and maintain data integrity.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Stable
	CodeColumnNumberKey = attribute.Key("code.column.number")

	// CodeFilePathKey is the attribute Key conforming to the "code.file.path"
	// semantic conventions. It represents the source code file name that identifies
	// the code unit as uniquely as possible (preferably an absolute file path).
	// This attribute MUST NOT be used on the Profile signal since the data is
	// already captured in 'message Function'. This constraint is imposed to prevent
	// redundancy and maintain data integrity.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: /usr/local/MyApplication/content_root/app/index.php
	CodeFilePathKey = attribute.Key("code.file.path")

	// CodeFunctionNameKey is the attribute Key conforming to the
	// "code.function.name" semantic conventions. It represents the method or
	// function fully-qualified name without arguments. The value should fit the
	// natural representation of the language runtime, which is also likely the same
	// used within `code.stacktrace` attribute value. This attribute MUST NOT be
	// used on the Profile signal since the data is already captured in 'message
	// Function'. This constraint is imposed to prevent redundancy and maintain data
	// integrity.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "com.example.MyHttpService.serveRequest",
	// "GuzzleHttp\Client::transfer", "fopen"
	// Note: Values and format depends on each language runtime, thus it is
	// impossible to provide an exhaustive list of examples.
	// The values are usually the same (or prefixes of) the ones found in native
	// stack trace representation stored in
	// `code.stacktrace` without information on arguments.
	//
	// Examples:
	//
	//   - Java method: `com.example.MyHttpService.serveRequest`
	//   - Java anonymous class method: `com.mycompany.Main$1.myMethod`
	//   - Java lambda method:
	//     `com.mycompany.Main$$Lambda/0x0000748ae4149c00.myMethod`
	//   - PHP function: `GuzzleHttp\Client::transfer`
	//   - Go function: `github.com/my/repo/pkg.foo.func5`
	//   - Elixir: `OpenTelemetry.Ctx.new`
	//   - Erlang: `opentelemetry_ctx:new`
	//   - Rust: `playground::my_module::my_cool_func`
	//   - C function: `fopen`
	CodeFunctionNameKey = attribute.Key("code.function.name")

	// CodeLineNumberKey is the attribute Key conforming to the "code.line.number"
	// semantic conventions. It represents the line number in `code.file.path` best
	// representing the operation. It SHOULD point within the code unit named in
	// `code.function.name`. This attribute MUST NOT be used on the Profile signal
	// since the data is already captured in 'message Line'. This constraint is
	// imposed to prevent redundancy and maintain data integrity.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Stable
	CodeLineNumberKey = attribute.Key("code.line.number")

	// CodeStacktraceKey is the attribute Key conforming to the "code.stacktrace"
	// semantic conventions. It represents a stacktrace as a string in the natural
	// representation for the language runtime. The representation is identical to
	// [`exception.stacktrace`]. This attribute MUST NOT be used on the Profile
	// signal since the data is already captured in 'message Location'. This
	// constraint is imposed to prevent redundancy and maintain data integrity.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: at com.example.GenerateTrace.methodB(GenerateTrace.java:13)\n at
	// com.example.GenerateTrace.methodA(GenerateTrace.java:9)\n at
	// com.example.GenerateTrace.main(GenerateTrace.java:5)
	//
	// [`exception.stacktrace`]: /docs/exceptions/exceptions-spans.md#stacktrace-representation
	CodeStacktraceKey = attribute.Key("code.stacktrace")
)

// CodeColumnNumber returns an attribute KeyValue conforming to the
// "code.column.number" semantic conventions. It represents the column number in
// `code.file.path` best representing the operation. It SHOULD point within the
// code unit named in `code.function.name`. This attribute MUST NOT be used on
// the Profile signal since the data is already captured in 'message Line'. This
// constraint is imposed to prevent redundancy and maintain data integrity.
func CodeColumnNumber(val int) attribute.KeyValue {
	return CodeColumnNumberKey.Int(val)
}

// CodeFilePath returns an attribute KeyValue conforming to the "code.file.path"
// semantic conventions. It represents the source code file name that identifies
// the code unit as uniquely as possible (preferably an absolute file path). This
// attribute MUST NOT be used on the Profile signal since the data is already
// captured in 'message Function'. This constraint is imposed to prevent
// redundancy and maintain data integrity.
func CodeFilePath(val string) attribute.KeyValue {
	return CodeFilePathKey.String(val)
}

// CodeFunctionName returns an attribute KeyValue conforming to the
// "code.function.name" semantic conventions. It represents the method or
// function fully-qualified name without arguments. The value should fit the
// natural representation of the language runtime, which is also likely the same
// used within `code.stacktrace` attribute value. This attribute MUST NOT be used
// on the Profile signal since the data is already captured in 'message
// Function'. This constraint is imposed to prevent redundancy and maintain data
// integrity.
func CodeFunctionName(val string) attribute.KeyValue {
	return CodeFunctionNameKey.String(val)
}

// CodeLineNumber returns an attribute KeyValue conforming to the
// "code.line.number" semantic conventions. It represents the line number in
// `code.file.path` best representing the operation. It SHOULD point within the
// code unit named in `code.function.name`. This attribute MUST NOT be used on
// the Profile signal since the data is already captured in 'message Line'. This
// constraint is imposed to prevent redundancy and maintain data integrity.
func CodeLineNumber(val int) attribute.KeyValue {
	return CodeLineNumberKey.Int(val)
}

// CodeStacktrace returns an attribute KeyValue conforming to the
// "code.stacktrace" semantic conventions. It represents a stacktrace as a string
// in the natural representation for the language runtime. The representation is
// identical to [`exception.stacktrace`]. This attribute MUST NOT be used on the
// Profile signal since the data is already captured in 'message Location'. This
// constraint is imposed to prevent redundancy and maintain data integrity.
//
// [`exception.stacktrace`]: /docs/exceptions/exceptions-spans.md#stacktrace-representation
func CodeStacktrace(val string) attribute.KeyValue {
	return CodeStacktraceKey.String(val)
}

// Namespace: container
const (
	// ContainerCommandKey is the attribute Key conforming to the
	// "container.command" semantic conventions. It represents the command used to
	// run the container (i.e. the command name).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "otelcontribcol"
	// Note: If using embedded credentials or sensitive data, it is recommended to
	// remove them to prevent potential leakage.
	ContainerCommandKey = attribute.Key("container.command")

	// ContainerCommandArgsKey is the attribute Key conforming to the
	// "container.command_args" semantic conventions. It represents the all the
	// command arguments (including the command/executable itself) run by the
	// container.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "otelcontribcol", "--config", "config.yaml"
	ContainerCommandArgsKey = attribute.Key("container.command_args")

	// ContainerCommandLineKey is the attribute Key conforming to the
	// "container.command_line" semantic conventions. It represents the full command
	// run by the container as a single string representing the full command.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "otelcontribcol --config config.yaml"
	ContainerCommandLineKey = attribute.Key("container.command_line")

	// ContainerCSIPluginNameKey is the attribute Key conforming to the
	// "container.csi.plugin.name" semantic conventions. It represents the name of
	// the CSI ([Container Storage Interface]) plugin used by the volume.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "pd.csi.storage.gke.io"
	// Note: This can sometimes be referred to as a "driver" in CSI implementations.
	// This should represent the `name` field of the GetPluginInfo RPC.
	//
	// [Container Storage Interface]: https://github.com/container-storage-interface/spec
	ContainerCSIPluginNameKey = attribute.Key("container.csi.plugin.name")

	// ContainerCSIVolumeIDKey is the attribute Key conforming to the
	// "container.csi.volume.id" semantic conventions. It represents the unique
	// volume ID returned by the CSI ([Container Storage Interface]) plugin.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "projects/my-gcp-project/zones/my-gcp-zone/disks/my-gcp-disk"
	// Note: This can sometimes be referred to as a "volume handle" in CSI
	// implementations. This should represent the `Volume.volume_id` field in CSI
	// spec.
	//
	// [Container Storage Interface]: https://github.com/container-storage-interface/spec
	ContainerCSIVolumeIDKey = attribute.Key("container.csi.volume.id")

	// ContainerIDKey is the attribute Key conforming to the "container.id" semantic
	// conventions. It represents the container ID. Usually a UUID, as for example
	// used to [identify Docker containers]. The UUID might be abbreviated.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "a3bf90e006b2"
	//
	// [identify Docker containers]: https://docs.docker.com/engine/containers/run/#container-identification
	ContainerIDKey = attribute.Key("container.id")

	// ContainerImageIDKey is the attribute Key conforming to the
	// "container.image.id" semantic conventions. It represents the runtime specific
	// image identifier. Usually a hash algorithm followed by a UUID.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// "sha256:19c92d0a00d1b66d897bceaa7319bee0dd38a10a851c60bcec9474aa3f01e50f"
	// Note: Docker defines a sha256 of the image id; `container.image.id`
	// corresponds to the `Image` field from the Docker container inspect [API]
	// endpoint.
	// K8s defines a link to the container registry repository with digest
	// `"imageID": "registry.azurecr.io /namespace/service/dockerfile@sha256:bdeabd40c3a8a492eaf9e8e44d0ebbb84bac7ee25ac0cf8a7159d25f62555625"`
	// .
	// The ID is assigned by the container runtime and can vary in different
	// environments. Consider using `oci.manifest.digest` if it is important to
	// identify the same image in different environments/runtimes.
	//
	// [API]: https://docs.docker.com/engine/api/v1.43/#tag/Container/operation/ContainerInspect
	ContainerImageIDKey = attribute.Key("container.image.id")

	// ContainerImageNameKey is the attribute Key conforming to the
	// "container.image.name" semantic conventions. It represents the name of the
	// image the container was built on.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "gcr.io/opentelemetry/operator"
	ContainerImageNameKey = attribute.Key("container.image.name")

	// ContainerImageRepoDigestsKey is the attribute Key conforming to the
	// "container.image.repo_digests" semantic conventions. It represents the repo
	// digests of the container image as provided by the container runtime.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// "example@sha256:afcc7f1ac1b49db317a7196c902e61c6c3c4607d63599ee1a82d702d249a0ccb",
	// "internal.registry.example.com:5000/example@sha256:b69959407d21e8a062e0416bf13405bb2b71ed7a84dde4158ebafacfa06f5578"
	// Note: [Docker] and [CRI] report those under the `RepoDigests` field.
	//
	// [Docker]: https://docs.docker.com/engine/api/v1.43/#tag/Image/operation/ImageInspect
	// [CRI]: https://github.com/kubernetes/cri-api/blob/c75ef5b473bbe2d0a4fc92f82235efd665ea8e9f/pkg/apis/runtime/v1/api.proto#L1237-L1238
	ContainerImageRepoDigestsKey = attribute.Key("container.image.repo_digests")

	// ContainerImageTagsKey is the attribute Key conforming to the
	// "container.image.tags" semantic conventions. It represents the container
	// image tags. An example can be found in [Docker Image Inspect]. Should be only
	// the `<tag>` section of the full name for example from
	// `registry.example.com/my-org/my-image:<tag>`.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "v1.27.1", "3.5.7-0"
	//
	// [Docker Image Inspect]: https://docs.docker.com/engine/api/v1.43/#tag/Image/operation/ImageInspect
	ContainerImageTagsKey = attribute.Key("container.image.tags")

	// ContainerNameKey is the attribute Key conforming to the "container.name"
	// semantic conventions. It represents the container name used by container
	// runtime.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "opentelemetry-autoconf"
	ContainerNameKey = attribute.Key("container.name")

	// ContainerRuntimeDescriptionKey is the attribute Key conforming to the
	// "container.runtime.description" semantic conventions. It represents a
	// description about the runtime which could include, for example details about
	// the CRI/API version being used or other customisations.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "docker://19.3.1 - CRI: 1.22.0"
	ContainerRuntimeDescriptionKey = attribute.Key("container.runtime.description")

	// ContainerRuntimeNameKey is the attribute Key conforming to the
	// "container.runtime.name" semantic conventions. It represents the container
	// runtime managing this container.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "docker", "containerd", "rkt"
	ContainerRuntimeNameKey = attribute.Key("container.runtime.name")

	// ContainerRuntimeVersionKey is the attribute Key conforming to the
	// "container.runtime.version" semantic conventions. It represents the version
	// of the runtime of this process, as returned by the runtime without
	// modification.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1.0.0
	ContainerRuntimeVersionKey = attribute.Key("container.runtime.version")
)

// ContainerCommand returns an attribute KeyValue conforming to the
// "container.command" semantic conventions. It represents the command used to
// run the container (i.e. the command name).
func ContainerCommand(val string) attribute.KeyValue {
	return ContainerCommandKey.String(val)
}

// ContainerCommandArgs returns an attribute KeyValue conforming to the
// "container.command_args" semantic conventions. It represents the all the
// command arguments (including the command/executable itself) run by the
// container.
func ContainerCommandArgs(val ...string) attribute.KeyValue {
	return ContainerCommandArgsKey.StringSlice(val)
}

// ContainerCommandLine returns an attribute KeyValue conforming to the
// "container.command_line" semantic conventions. It represents the full command
// run by the container as a single string representing the full command.
func ContainerCommandLine(val string) attribute.KeyValue {
	return ContainerCommandLineKey.String(val)
}

// ContainerCSIPluginName returns an attribute KeyValue conforming to the
// "container.csi.plugin.name" semantic conventions. It represents the name of
// the CSI ([Container Storage Interface]) plugin used by the volume.
//
// [Container Storage Interface]: https://github.com/container-storage-interface/spec
func ContainerCSIPluginName(val string) attribute.KeyValue {
	return ContainerCSIPluginNameKey.String(val)
}

// ContainerCSIVolumeID returns an attribute KeyValue conforming to the
// "container.csi.volume.id" semantic conventions. It represents the unique
// volume ID returned by the CSI ([Container Storage Interface]) plugin.
//
// [Container Storage Interface]: https://github.com/container-storage-interface/spec
func ContainerCSIVolumeID(val string) attribute.KeyValue {
	return ContainerCSIVolumeIDKey.String(val)
}

// ContainerID returns an attribute KeyValue conforming to the "container.id"
// semantic conventions. It represents the container ID. Usually a UUID, as for
// example used to [identify Docker containers]. The UUID might be abbreviated.
//
// [identify Docker containers]: https://docs.docker.com/engine/containers/run/#container-identification
func ContainerID(val string) attribute.KeyValue {
	return ContainerIDKey.String(val)
}

// ContainerImageID returns an attribute KeyValue conforming to the
// "container.image.id" semantic conventions. It represents the runtime specific
// image identifier. Usually a hash algorithm followed by a UUID.
func ContainerImageID(val string) attribute.KeyValue {
	return ContainerImageIDKey.String(val)
}

// ContainerImageName returns an attribute KeyValue conforming to the
// "container.image.name" semantic conventions. It represents the name of the
// image the container was built on.
func ContainerImageName(val string) attribute.KeyValue {
	return ContainerImageNameKey.String(val)
}

// ContainerImageRepoDigests returns an attribute KeyValue conforming to the
// "container.image.repo_digests" semantic conventions. It represents the repo
// digests of the container image as provided by the container runtime.
func ContainerImageRepoDigests(val ...string) attribute.KeyValue {
	return ContainerImageRepoDigestsKey.StringSlice(val)
}

// ContainerImageTags returns an attribute KeyValue conforming to the
// "container.image.tags" semantic conventions. It represents the container image
// tags. An example can be found in [Docker Image Inspect]. Should be only the
// `<tag>` section of the full name for example from
// `registry.example.com/my-org/my-image:<tag>`.
//
// [Docker Image Inspect]: https://docs.docker.com/engine/api/v1.43/#tag/Image/operation/ImageInspect
func ContainerImageTags(val ...string) attribute.KeyValue {
	return ContainerImageTagsKey.StringSlice(val)
}

// ContainerLabel returns an attribute KeyValue conforming to the
// "container.label" semantic conventions. It represents the container labels,
// `<key>` being the label name, the value being the label value.
func ContainerLabel(key string, val string) attribute.KeyValue {
	return attribute.String("container.label."+key, val)
}

// ContainerName returns an attribute KeyValue conforming to the "container.name"
// semantic conventions. It represents the container name used by container
// runtime.
func ContainerName(val string) attribute.KeyValue {
	return ContainerNameKey.String(val)
}

// ContainerRuntimeDescription returns an attribute KeyValue conforming to the
// "container.runtime.description" semantic conventions. It represents a
// description about the runtime which could include, for example details about
// the CRI/API version being used or other customisations.
func ContainerRuntimeDescription(val string) attribute.KeyValue {
	return ContainerRuntimeDescriptionKey.String(val)
}

// ContainerRuntimeName returns an attribute KeyValue conforming to the
// "container.runtime.name" semantic conventions. It represents the container
// runtime managing this container.
func ContainerRuntimeName(val string) attribute.KeyValue {
	return ContainerRuntimeNameKey.String(val)
}

// ContainerRuntimeVersion returns an attribute KeyValue conforming to the
// "container.runtime.version" semantic conventions. It represents the version of
// the runtime of this process, as returned by the runtime without modification.
func ContainerRuntimeVersion(val string) attribute.KeyValue {
	return ContainerRuntimeVersionKey.String(val)
}

// Namespace: cpu
const (
	// CPULogicalNumberKey is the attribute Key conforming to the
	// "cpu.logical_number" semantic conventions. It represents the logical CPU
	// number [0..n-1].
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1
	CPULogicalNumberKey = attribute.Key("cpu.logical_number")

	// CPUModeKey is the attribute Key conforming to the "cpu.mode" semantic
	// conventions. It represents the mode of the CPU.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "user", "system"
	CPUModeKey = attribute.Key("cpu.mode")
)

// CPULogicalNumber returns an attribute KeyValue conforming to the
// "cpu.logical_number" semantic conventions. It represents the logical CPU
// number [0..n-1].
func CPULogicalNumber(val int) attribute.KeyValue {
	return CPULogicalNumberKey.Int(val)
}

// Enum values for cpu.mode
var (
	// User
	// Stability: development
	CPUModeUser = CPUModeKey.String("user")
	// System
	// Stability: development
	CPUModeSystem = CPUModeKey.String("system")
	// Nice
	// Stability: development
	CPUModeNice = CPUModeKey.String("nice")
	// Idle
	// Stability: development
	CPUModeIdle = CPUModeKey.String("idle")
	// IO Wait
	// Stability: development
	CPUModeIOWait = CPUModeKey.String("iowait")
	// Interrupt
	// Stability: development
	CPUModeInterrupt = CPUModeKey.String("interrupt")
	// Steal
	// Stability: development
	CPUModeSteal = CPUModeKey.String("steal")
	// Kernel
	// Stability: development
	CPUModeKernel = CPUModeKey.String("kernel")
)

// Namespace: db
const (
	// DBClientConnectionPoolNameKey is the attribute Key conforming to the
	// "db.client.connection.pool.name" semantic conventions. It represents the name
	// of the connection pool; unique within the instrumented application. In case
	// the connection pool implementation doesn't provide a name, instrumentation
	// SHOULD use a combination of parameters that would make the name unique, for
	// example, combining attributes `server.address`, `server.port`, and
	// `db.namespace`, formatted as `server.address:server.port/db.namespace`.
	// Instrumentations that generate connection pool name following different
	// patterns SHOULD document it.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "myDataSource"
	DBClientConnectionPoolNameKey = attribute.Key("db.client.connection.pool.name")

	// DBClientConnectionStateKey is the attribute Key conforming to the
	// "db.client.connection.state" semantic conventions. It represents the state of
	// a connection in the pool.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "idle"
	DBClientConnectionStateKey = attribute.Key("db.client.connection.state")

	// DBCollectionNameKey is the attribute Key conforming to the
	// "db.collection.name" semantic conventions. It represents the name of a
	// collection (table, container) within the database.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "public.users", "customers"
	// Note: It is RECOMMENDED to capture the value as provided by the application
	// without attempting to do any case normalization.
	//
	// The collection name SHOULD NOT be extracted from `db.query.text`,
	// when the database system supports query text with multiple collections
	// in non-batch operations.
	//
	// For batch operations, if the individual operations are known to have the same
	// collection name then that collection name SHOULD be used.
	DBCollectionNameKey = attribute.Key("db.collection.name")

	// DBNamespaceKey is the attribute Key conforming to the "db.namespace" semantic
	// conventions. It represents the name of the database, fully qualified within
	// the server address and port.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "customers", "test.users"
	// Note: If a database system has multiple namespace components, they SHOULD be
	// concatenated from the most general to the most specific namespace component,
	// using `|` as a separator between the components. Any missing components (and
	// their associated separators) SHOULD be omitted.
	// Semantic conventions for individual database systems SHOULD document what
	// `db.namespace` means in the context of that system.
	// It is RECOMMENDED to capture the value as provided by the application without
	// attempting to do any case normalization.
	DBNamespaceKey = attribute.Key("db.namespace")

	// DBOperationBatchSizeKey is the attribute Key conforming to the
	// "db.operation.batch.size" semantic conventions. It represents the number of
	// queries included in a batch operation.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: 2, 3, 4
	// Note: Operations are only considered batches when they contain two or more
	// operations, and so `db.operation.batch.size` SHOULD never be `1`.
	DBOperationBatchSizeKey = attribute.Key("db.operation.batch.size")

	// DBOperationNameKey is the attribute Key conforming to the "db.operation.name"
	// semantic conventions. It represents the name of the operation or command
	// being executed.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "findAndModify", "HMSET", "SELECT"
	// Note: It is RECOMMENDED to capture the value as provided by the application
	// without attempting to do any case normalization.
	//
	// The operation name SHOULD NOT be extracted from `db.query.text`,
	// when the database system supports query text with multiple operations
	// in non-batch operations.
	//
	// If spaces can occur in the operation name, multiple consecutive spaces
	// SHOULD be normalized to a single space.
	//
	// For batch operations, if the individual operations are known to have the same
	// operation name
	// then that operation name SHOULD be used prepended by `BATCH `,
	// otherwise `db.operation.name` SHOULD be `BATCH` or some other database
	// system specific term if more applicable.
	DBOperationNameKey = attribute.Key("db.operation.name")

	// DBQuerySummaryKey is the attribute Key conforming to the "db.query.summary"
	// semantic conventions. It represents the low cardinality summary of a database
	// query.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "SELECT wuser_table", "INSERT shipping_details SELECT orders", "get
	// user by id"
	// Note: The query summary describes a class of database queries and is useful
	// as a grouping key, especially when analyzing telemetry for database
	// calls involving complex queries.
	//
	// Summary may be available to the instrumentation through
	// instrumentation hooks or other means. If it is not available,
	// instrumentations
	// that support query parsing SHOULD generate a summary following
	// [Generating query summary]
	// section.
	//
	// [Generating query summary]: /docs/database/database-spans.md#generating-a-summary-of-the-query
	DBQuerySummaryKey = attribute.Key("db.query.summary")

	// DBQueryTextKey is the attribute Key conforming to the "db.query.text"
	// semantic conventions. It represents the database query being executed.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "SELECT * FROM wuser_table where username = ?", "SET mykey ?"
	// Note: For sanitization see [Sanitization of `db.query.text`].
	// For batch operations, if the individual operations are known to have the same
	// query text then that query text SHOULD be used, otherwise all of the
	// individual query texts SHOULD be concatenated with separator `; ` or some
	// other database system specific separator if more applicable.
	// Parameterized query text SHOULD NOT be sanitized. Even though parameterized
	// query text can potentially have sensitive data, by using a parameterized
	// query the user is giving a strong signal that any sensitive data will be
	// passed as parameter values, and the benefit to observability of capturing the
	// static part of the query text by default outweighs the risk.
	//
	// [Sanitization of `db.query.text`]: /docs/database/database-spans.md#sanitization-of-dbquerytext
	DBQueryTextKey = attribute.Key("db.query.text")

	// DBResponseReturnedRowsKey is the attribute Key conforming to the
	// "db.response.returned_rows" semantic conventions. It represents the number of
	// rows returned by the operation.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 10, 30, 1000
	DBResponseReturnedRowsKey = attribute.Key("db.response.returned_rows")

	// DBResponseStatusCodeKey is the attribute Key conforming to the
	// "db.response.status_code" semantic conventions. It represents the database
	// response status code.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "102", "ORA-17002", "08P01", "404"
	// Note: The status code returned by the database. Usually it represents an
	// error code, but may also represent partial success, warning, or differentiate
	// between various types of successful outcomes.
	// Semantic conventions for individual database systems SHOULD document what
	// `db.response.status_code` means in the context of that system.
	DBResponseStatusCodeKey = attribute.Key("db.response.status_code")

	// DBStoredProcedureNameKey is the attribute Key conforming to the
	// "db.stored_procedure.name" semantic conventions. It represents the name of a
	// stored procedure within the database.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "GetCustomer"
	// Note: It is RECOMMENDED to capture the value as provided by the application
	// without attempting to do any case normalization.
	//
	// For batch operations, if the individual operations are known to have the same
	// stored procedure name then that stored procedure name SHOULD be used.
	DBStoredProcedureNameKey = attribute.Key("db.stored_procedure.name")

	// DBSystemNameKey is the attribute Key conforming to the "db.system.name"
	// semantic conventions. It represents the database management system (DBMS)
	// product as identified by the client instrumentation.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples:
	// Note: The actual DBMS may differ from the one identified by the client. For
	// example, when using PostgreSQL client libraries to connect to a CockroachDB,
	// the `db.system.name` is set to `postgresql` based on the instrumentation's
	// best knowledge.
	DBSystemNameKey = attribute.Key("db.system.name")
)

// DBClientConnectionPoolName returns an attribute KeyValue conforming to the
// "db.client.connection.pool.name" semantic conventions. It represents the name
// of the connection pool; unique within the instrumented application. In case
// the connection pool implementation doesn't provide a name, instrumentation
// SHOULD use a combination of parameters that would make the name unique, for
// example, combining attributes `server.address`, `server.port`, and
// `db.namespace`, formatted as `server.address:server.port/db.namespace`.
// Instrumentations that generate connection pool name following different
// patterns SHOULD document it.
func DBClientConnectionPoolName(val string) attribute.KeyValue {
	return DBClientConnectionPoolNameKey.String(val)
}

// DBCollectionName returns an attribute KeyValue conforming to the
// "db.collection.name" semantic conventions. It represents the name of a
// collection (table, container) within the database.
func DBCollectionName(val string) attribute.KeyValue {
	return DBCollectionNameKey.String(val)
}

// DBNamespace returns an attribute KeyValue conforming to the "db.namespace"
// semantic conventions. It represents the name of the database, fully qualified
// within the server address and port.
func DBNamespace(val string) attribute.KeyValue {
	return DBNamespaceKey.String(val)
}

// DBOperationBatchSize returns an attribute KeyValue conforming to the
// "db.operation.batch.size" semantic conventions. It represents the number of
// queries included in a batch operation.
func DBOperationBatchSize(val int) attribute.KeyValue {
	return DBOperationBatchSizeKey.Int(val)
}

// DBOperationName returns an attribute KeyValue conforming to the
// "db.operation.name" semantic conventions. It represents the name of the
// operation or command being executed.
func DBOperationName(val string) attribute.KeyValue {
	return DBOperationNameKey.String(val)
}

// DBOperationParameter returns an attribute KeyValue conforming to the
// "db.operation.parameter" semantic conventions. It represents a database
// operation parameter, with `<key>` being the parameter name, and the attribute
// value being a string representation of the parameter value.
func DBOperationParameter(key string, val string) attribute.KeyValue {
	return attribute.String("db.operation.parameter."+key, val)
}

// DBQueryParameter returns an attribute KeyValue conforming to the
// "db.query.parameter" semantic conventions. It represents a database query
// parameter, with `<key>` being the parameter name, and the attribute value
// being a string representation of the parameter value.
func DBQueryParameter(key string, val string) attribute.KeyValue {
	return attribute.String("db.query.parameter."+key, val)
}

// DBQuerySummary returns an attribute KeyValue conforming to the
// "db.query.summary" semantic conventions. It represents the low cardinality
// summary of a database query.
func DBQuerySummary(val string) attribute.KeyValue {
	return DBQuerySummaryKey.String(val)
}

// DBQueryText returns an attribute KeyValue conforming to the "db.query.text"
// semantic conventions. It represents the database query being executed.
func DBQueryText(val string) attribute.KeyValue {
	return DBQueryTextKey.String(val)
}

// DBResponseReturnedRows returns an attribute KeyValue conforming to the
// "db.response.returned_rows" semantic conventions. It represents the number of
// rows returned by the operation.
func DBResponseReturnedRows(val int) attribute.KeyValue {
	return DBResponseReturnedRowsKey.Int(val)
}

// DBResponseStatusCode returns an attribute KeyValue conforming to the
// "db.response.status_code" semantic conventions. It represents the database
// response status code.
func DBResponseStatusCode(val string) attribute.KeyValue {
	return DBResponseStatusCodeKey.String(val)
}

// DBStoredProcedureName returns an attribute KeyValue conforming to the
// "db.stored_procedure.name" semantic conventions. It represents the name of a
// stored procedure within the database.
func DBStoredProcedureName(val string) attribute.KeyValue {
	return DBStoredProcedureNameKey.String(val)
}

// Enum values for db.client.connection.state
var (
	// idle
	// Stability: development
	DBClientConnectionStateIdle = DBClientConnectionStateKey.String("idle")
	// used
	// Stability: development
	DBClientConnectionStateUsed = DBClientConnectionStateKey.String("used")
)

// Enum values for db.system.name
var (
	// Some other SQL database. Fallback only.
	// Stability: development
	DBSystemNameOtherSQL = DBSystemNameKey.String("other_sql")
	// [Adabas (Adaptable Database System)]
	// Stability: development
	//
	// [Adabas (Adaptable Database System)]: https://documentation.softwareag.com/?pf=adabas
	DBSystemNameSoftwareagAdabas = DBSystemNameKey.String("softwareag.adabas")
	// [Actian Ingres]
	// Stability: development
	//
	// [Actian Ingres]: https://www.actian.com/databases/ingres/
	DBSystemNameActianIngres = DBSystemNameKey.String("actian.ingres")
	// [Amazon DynamoDB]
	// Stability: development
	//
	// [Amazon DynamoDB]: https://aws.amazon.com/pm/dynamodb/
	DBSystemNameAWSDynamoDB = DBSystemNameKey.String("aws.dynamodb")
	// [Amazon Redshift]
	// Stability: development
	//
	// [Amazon Redshift]: https://aws.amazon.com/redshift/
	DBSystemNameAWSRedshift = DBSystemNameKey.String("aws.redshift")
	// [Azure Cosmos DB]
	// Stability: development
	//
	// [Azure Cosmos DB]: https://learn.microsoft.com/azure/cosmos-db
	DBSystemNameAzureCosmosDB = DBSystemNameKey.String("azure.cosmosdb")
	// [InterSystems Caché]
	// Stability: development
	//
	// [InterSystems Caché]: https://www.intersystems.com/products/cache/
	DBSystemNameIntersystemsCache = DBSystemNameKey.String("intersystems.cache")
	// [Apache Cassandra]
	// Stability: development
	//
	// [Apache Cassandra]: https://cassandra.apache.org/
	DBSystemNameCassandra = DBSystemNameKey.String("cassandra")
	// [ClickHouse]
	// Stability: development
	//
	// [ClickHouse]: https://clickhouse.com/
	DBSystemNameClickHouse = DBSystemNameKey.String("clickhouse")
	// [CockroachDB]
	// Stability: development
	//
	// [CockroachDB]: https://www.cockroachlabs.com/
	DBSystemNameCockroachDB = DBSystemNameKey.String("cockroachdb")
	// [Couchbase]
	// Stability: development
	//
	// [Couchbase]: https://www.couchbase.com/
	DBSystemNameCouchbase = DBSystemNameKey.String("couchbase")
	// [Apache CouchDB]
	// Stability: development
	//
	// [Apache CouchDB]: https://couchdb.apache.org/
	DBSystemNameCouchDB = DBSystemNameKey.String("couchdb")
	// [Apache Derby]
	// Stability: development
	//
	// [Apache Derby]: https://db.apache.org/derby/
	DBSystemNameDerby = DBSystemNameKey.String("derby")
	// [Elasticsearch]
	// Stability: development
	//
	// [Elasticsearch]: https://www.elastic.co/elasticsearch
	DBSystemNameElasticsearch = DBSystemNameKey.String("elasticsearch")
	// [Firebird]
	// Stability: development
	//
	// [Firebird]: https://www.firebirdsql.org/
	DBSystemNameFirebirdSQL = DBSystemNameKey.String("firebirdsql")
	// [Google Cloud Spanner]
	// Stability: development
	//
	// [Google Cloud Spanner]: https://cloud.google.com/spanner
	DBSystemNameGCPSpanner = DBSystemNameKey.String("gcp.spanner")
	// [Apache Geode]
	// Stability: development
	//
	// [Apache Geode]: https://geode.apache.org/
	DBSystemNameGeode = DBSystemNameKey.String("geode")
	// [H2 Database]
	// Stability: development
	//
	// [H2 Database]: https://h2database.com/
	DBSystemNameH2database = DBSystemNameKey.String("h2database")
	// [Apache HBase]
	// Stability: development
	//
	// [Apache HBase]: https://hbase.apache.org/
	DBSystemNameHBase = DBSystemNameKey.String("hbase")
	// [Apache Hive]
	// Stability: development
	//
	// [Apache Hive]: https://hive.apache.org/
	DBSystemNameHive = DBSystemNameKey.String("hive")
	// [HyperSQL Database]
	// Stability: development
	//
	// [HyperSQL Database]: https://hsqldb.org/
	DBSystemNameHSQLDB = DBSystemNameKey.String("hsqldb")
	// [IBM Db2]
	// Stability: development
	//
	// [IBM Db2]: https://www.ibm.com/db2
	DBSystemNameIBMDB2 = DBSystemNameKey.String("ibm.db2")
	// [IBM Informix]
	// Stability: development
	//
	// [IBM Informix]: https://www.ibm.com/products/informix
	DBSystemNameIBMInformix = DBSystemNameKey.String("ibm.informix")
	// [IBM Netezza]
	// Stability: development
	//
	// [IBM Netezza]: https://www.ibm.com/products/netezza
	DBSystemNameIBMNetezza = DBSystemNameKey.String("ibm.netezza")
	// [InfluxDB]
	// Stability: development
	//
	// [InfluxDB]: https://www.influxdata.com/
	DBSystemNameInfluxDB = DBSystemNameKey.String("influxdb")
	// [Instant]
	// Stability: development
	//
	// [Instant]: https://www.instantdb.com/
	DBSystemNameInstantDB = DBSystemNameKey.String("instantdb")
	// [MariaDB]
	// Stability: stable
	//
	// [MariaDB]: https://mariadb.org/
	DBSystemNameMariaDB = DBSystemNameKey.String("mariadb")
	// [Memcached]
	// Stability: development
	//
	// [Memcached]: https://memcached.org/
	DBSystemNameMemcached = DBSystemNameKey.String("memcached")
	// [MongoDB]
	// Stability: development
	//
	// [MongoDB]: https://www.mongodb.com/
	DBSystemNameMongoDB = DBSystemNameKey.String("mongodb")
	// [Microsoft SQL Server]
	// Stability: stable
	//
	// [Microsoft SQL Server]: https://www.microsoft.com/sql-server
	DBSystemNameMicrosoftSQLServer = DBSystemNameKey.String("microsoft.sql_server")
	// [MySQL]
	// Stability: stable
	//
	// [MySQL]: https://www.mysql.com/
	DBSystemNameMySQL = DBSystemNameKey.String("mysql")
	// [Neo4j]
	// Stability: development
	//
	// [Neo4j]: https://neo4j.com/
	DBSystemNameNeo4j = DBSystemNameKey.String("neo4j")
	// [OpenSearch]
	// Stability: development
	//
	// [OpenSearch]: https://opensearch.org/
	DBSystemNameOpenSearch = DBSystemNameKey.String("opensearch")
	// [Oracle Database]
	// Stability: development
	//
	// [Oracle Database]: https://www.oracle.com/database/
	DBSystemNameOracleDB = DBSystemNameKey.String("oracle.db")
	// [PostgreSQL]
	// Stability: stable
	//
	// [PostgreSQL]: https://www.postgresql.org/
	DBSystemNamePostgreSQL = DBSystemNameKey.String("postgresql")
	// [Redis]
	// Stability: development
	//
	// [Redis]: https://redis.io/
	DBSystemNameRedis = DBSystemNameKey.String("redis")
	// [SAP HANA]
	// Stability: development
	//
	// [SAP HANA]: https://www.sap.com/products/technology-platform/hana/what-is-sap-hana.html
	DBSystemNameSAPHANA = DBSystemNameKey.String("sap.hana")
	// [SAP MaxDB]
	// Stability: development
	//
	// [SAP MaxDB]: https://maxdb.sap.com/
	DBSystemNameSAPMaxDB = DBSystemNameKey.String("sap.maxdb")
	// [SQLite]
	// Stability: development
	//
	// [SQLite]: https://www.sqlite.org/
	DBSystemNameSQLite = DBSystemNameKey.String("sqlite")
	// [Teradata]
	// Stability: development
	//
	// [Teradata]: https://www.teradata.com/
	DBSystemNameTeradata = DBSystemNameKey.String("teradata")
	// [Trino]
	// Stability: development
	//
	// [Trino]: https://trino.io/
	DBSystemNameTrino = DBSystemNameKey.String("trino")
)

// Namespace: deployment
const (
	// DeploymentEnvironmentNameKey is the attribute Key conforming to the
	// "deployment.environment.name" semantic conventions. It represents the name of
	// the [deployment environment] (aka deployment tier).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "staging", "production"
	// Note: `deployment.environment.name` does not affect the uniqueness
	// constraints defined through
	// the `service.namespace`, `service.name` and `service.instance.id` resource
	// attributes.
	// This implies that resources carrying the following attribute combinations
	// MUST be
	// considered to be identifying the same service:
	//
	//   - `service.name=frontend`, `deployment.environment.name=production`
	//   - `service.name=frontend`, `deployment.environment.name=staging`.
	//
	//
	// [deployment environment]: https://wikipedia.org/wiki/Deployment_environment
	DeploymentEnvironmentNameKey = attribute.Key("deployment.environment.name")

	// DeploymentIDKey is the attribute Key conforming to the "deployment.id"
	// semantic conventions. It represents the id of the deployment.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "1208"
	DeploymentIDKey = attribute.Key("deployment.id")

	// DeploymentNameKey is the attribute Key conforming to the "deployment.name"
	// semantic conventions. It represents the name of the deployment.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "deploy my app", "deploy-frontend"
	DeploymentNameKey = attribute.Key("deployment.name")

	// DeploymentStatusKey is the attribute Key conforming to the
	// "deployment.status" semantic conventions. It represents the status of the
	// deployment.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	DeploymentStatusKey = attribute.Key("deployment.status")
)

// DeploymentEnvironmentName returns an attribute KeyValue conforming to the
// "deployment.environment.name" semantic conventions. It represents the name of
// the [deployment environment] (aka deployment tier).
//
// [deployment environment]: https://wikipedia.org/wiki/Deployment_environment
func DeploymentEnvironmentName(val string) attribute.KeyValue {
	return DeploymentEnvironmentNameKey.String(val)
}

// DeploymentID returns an attribute KeyValue conforming to the "deployment.id"
// semantic conventions. It represents the id of the deployment.
func DeploymentID(val string) attribute.KeyValue {
	return DeploymentIDKey.String(val)
}

// DeploymentName returns an attribute KeyValue conforming to the
// "deployment.name" semantic conventions. It represents the name of the
// deployment.
func DeploymentName(val string) attribute.KeyValue {
	return DeploymentNameKey.String(val)
}

// Enum values for deployment.status
var (
	// failed
	// Stability: development
	DeploymentStatusFailed = DeploymentStatusKey.String("failed")
	// succeeded
	// Stability: development
	DeploymentStatusSucceeded = DeploymentStatusKey.String("succeeded")
)

// Namespace: destination
const (
	// DestinationAddressKey is the attribute Key conforming to the
	// "destination.address" semantic conventions. It represents the destination
	// address - domain name if available without reverse DNS lookup; otherwise, IP
	// address or Unix domain socket name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "destination.example.com", "10.1.2.80", "/tmp/my.sock"
	// Note: When observed from the source side, and when communicating through an
	// intermediary, `destination.address` SHOULD represent the destination address
	// behind any intermediaries, for example proxies, if it's available.
	DestinationAddressKey = attribute.Key("destination.address")

	// DestinationPortKey is the attribute Key conforming to the "destination.port"
	// semantic conventions. It represents the destination port number.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 3389, 2888
	DestinationPortKey = attribute.Key("destination.port")
)

// DestinationAddress returns an attribute KeyValue conforming to the
// "destination.address" semantic conventions. It represents the destination
// address - domain name if available without reverse DNS lookup; otherwise, IP
// address or Unix domain socket name.
func DestinationAddress(val string) attribute.KeyValue {
	return DestinationAddressKey.String(val)
}

// DestinationPort returns an attribute KeyValue conforming to the
// "destination.port" semantic conventions. It represents the destination port
// number.
func DestinationPort(val int) attribute.KeyValue {
	return DestinationPortKey.Int(val)
}

// Namespace: device
const (
	// DeviceIDKey is the attribute Key conforming to the "device.id" semantic
	// conventions. It represents a unique identifier representing the device.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "123456789012345", "01:23:45:67:89:AB"
	// Note: Its value SHOULD be identical for all apps on a device and it SHOULD
	// NOT change if an app is uninstalled and re-installed.
	// However, it might be resettable by the user for all apps on a device.
	// Hardware IDs (e.g. vendor-specific serial number, IMEI or MAC address) MAY be
	// used as values.
	//
	// More information about Android identifier best practices can be found in the
	// [Android user data IDs guide].
	//
	// > [!WARNING]> This attribute may contain sensitive (PII) information. Caution
	// > should be taken when storing personal data or anything which can identify a
	// > user. GDPR and data protection laws may apply,
	// > ensure you do your own due diligence.> Due to these reasons, this
	// > identifier is not recommended for consumer applications and will likely
	// > result in rejection from both Google Play and App Store.
	// > However, it may be appropriate for specific enterprise scenarios, such as
	// > kiosk devices or enterprise-managed devices, with appropriate compliance
	// > clearance.
	// > Any instrumentation providing this identifier MUST implement it as an
	// > opt-in feature.> See [`app.installation.id`]>  for a more
	// > privacy-preserving alternative.
	//
	// [Android user data IDs guide]: https://developer.android.com/training/articles/user-data-ids
	// [`app.installation.id`]: /docs/registry/attributes/app.md#app-installation-id
	DeviceIDKey = attribute.Key("device.id")

	// DeviceManufacturerKey is the attribute Key conforming to the
	// "device.manufacturer" semantic conventions. It represents the name of the
	// device manufacturer.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Apple", "Samsung"
	// Note: The Android OS provides this field via [Build]. iOS apps SHOULD
	// hardcode the value `Apple`.
	//
	// [Build]: https://developer.android.com/reference/android/os/Build#MANUFACTURER
	DeviceManufacturerKey = attribute.Key("device.manufacturer")

	// DeviceModelIdentifierKey is the attribute Key conforming to the
	// "device.model.identifier" semantic conventions. It represents the model
	// identifier for the device.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "iPhone3,4", "SM-G920F"
	// Note: It's recommended this value represents a machine-readable version of
	// the model identifier rather than the market or consumer-friendly name of the
	// device.
	DeviceModelIdentifierKey = attribute.Key("device.model.identifier")

	// DeviceModelNameKey is the attribute Key conforming to the "device.model.name"
	// semantic conventions. It represents the marketing name for the device model.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "iPhone 6s Plus", "Samsung Galaxy S6"
	// Note: It's recommended this value represents a human-readable version of the
	// device model rather than a machine-readable alternative.
	DeviceModelNameKey = attribute.Key("device.model.name")
)

// DeviceID returns an attribute KeyValue conforming to the "device.id" semantic
// conventions. It represents a unique identifier representing the device.
func DeviceID(val string) attribute.KeyValue {
	return DeviceIDKey.String(val)
}

// DeviceManufacturer returns an attribute KeyValue conforming to the
// "device.manufacturer" semantic conventions. It represents the name of the
// device manufacturer.
func DeviceManufacturer(val string) attribute.KeyValue {
	return DeviceManufacturerKey.String(val)
}

// DeviceModelIdentifier returns an attribute KeyValue conforming to the
// "device.model.identifier" semantic conventions. It represents the model
// identifier for the device.
func DeviceModelIdentifier(val string) attribute.KeyValue {
	return DeviceModelIdentifierKey.String(val)
}

// DeviceModelName returns an attribute KeyValue conforming to the
// "device.model.name" semantic conventions. It represents the marketing name for
// the device model.
func DeviceModelName(val string) attribute.KeyValue {
	return DeviceModelNameKey.String(val)
}

// Namespace: disk
const (
	// DiskIODirectionKey is the attribute Key conforming to the "disk.io.direction"
	// semantic conventions. It represents the disk IO operation direction.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "read"
	DiskIODirectionKey = attribute.Key("disk.io.direction")
)

// Enum values for disk.io.direction
var (
	// read
	// Stability: development
	DiskIODirectionRead = DiskIODirectionKey.String("read")
	// write
	// Stability: development
	DiskIODirectionWrite = DiskIODirectionKey.String("write")
)

// Namespace: dns
const (
	// DNSAnswersKey is the attribute Key conforming to the "dns.answers" semantic
	// conventions. It represents the list of IPv4 or IPv6 addresses resolved during
	// DNS lookup.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "10.0.0.1", "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
	DNSAnswersKey = attribute.Key("dns.answers")

	// DNSQuestionNameKey is the attribute Key conforming to the "dns.question.name"
	// semantic conventions. It represents the name being queried.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "www.example.com", "opentelemetry.io"
	// Note: If the name field contains non-printable characters (below 32 or above
	// 126), those characters should be represented as escaped base 10 integers
	// (\DDD). Back slashes and quotes should be escaped. Tabs, carriage returns,
	// and line feeds should be converted to \t, \r, and \n respectively.
	DNSQuestionNameKey = attribute.Key("dns.question.name")
)

// DNSAnswers returns an attribute KeyValue conforming to the "dns.answers"
// semantic conventions. It represents the list of IPv4 or IPv6 addresses
// resolved during DNS lookup.
func DNSAnswers(val ...string) attribute.KeyValue {
	return DNSAnswersKey.StringSlice(val)
}

// DNSQuestionName returns an attribute KeyValue conforming to the
// "dns.question.name" semantic conventions. It represents the name being
// queried.
func DNSQuestionName(val string) attribute.KeyValue {
	return DNSQuestionNameKey.String(val)
}

// Namespace: elasticsearch
const (
	// ElasticsearchNodeNameKey is the attribute Key conforming to the
	// "elasticsearch.node.name" semantic conventions. It represents the represents
	// the human-readable identifier of the node/instance to which a request was
	// routed.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "instance-0000000001"
	ElasticsearchNodeNameKey = attribute.Key("elasticsearch.node.name")
)

// ElasticsearchNodeName returns an attribute KeyValue conforming to the
// "elasticsearch.node.name" semantic conventions. It represents the represents
// the human-readable identifier of the node/instance to which a request was
// routed.
func ElasticsearchNodeName(val string) attribute.KeyValue {
	return ElasticsearchNodeNameKey.String(val)
}

// Namespace: enduser
const (
	// EnduserIDKey is the attribute Key conforming to the "enduser.id" semantic
	// conventions. It represents the unique identifier of an end user in the
	// system. It maybe a username, email address, or other identifier.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "username"
	// Note: Unique identifier of an end user in the system.
	//
	// > [!Warning]
	// > This field contains sensitive (PII) information.
	EnduserIDKey = attribute.Key("enduser.id")

	// EnduserPseudoIDKey is the attribute Key conforming to the "enduser.pseudo.id"
	// semantic conventions. It represents the pseudonymous identifier of an end
	// user. This identifier should be a random value that is not directly linked or
	// associated with the end user's actual identity.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "QdH5CAWJgqVT4rOr0qtumf"
	// Note: Pseudonymous identifier of an end user.
	//
	// > [!Warning]
	// > This field contains sensitive (linkable PII) information.
	EnduserPseudoIDKey = attribute.Key("enduser.pseudo.id")
)

// EnduserID returns an attribute KeyValue conforming to the "enduser.id"
// semantic conventions. It represents the unique identifier of an end user in
// the system. It maybe a username, email address, or other identifier.
func EnduserID(val string) attribute.KeyValue {
	return EnduserIDKey.String(val)
}

// EnduserPseudoID returns an attribute KeyValue conforming to the
// "enduser.pseudo.id" semantic conventions. It represents the pseudonymous
// identifier of an end user. This identifier should be a random value that is
// not directly linked or associated with the end user's actual identity.
func EnduserPseudoID(val string) attribute.KeyValue {
	return EnduserPseudoIDKey.String(val)
}

// Namespace: error
const (
	// ErrorMessageKey is the attribute Key conforming to the "error.message"
	// semantic conventions. It represents a message providing more detail about an
	// error in human-readable form.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Unexpected input type: string", "The user has exceeded their
	// storage quota"
	// Note: `error.message` should provide additional context and detail about an
	// error.
	// It is NOT RECOMMENDED to duplicate the value of `error.type` in
	// `error.message`.
	// It is also NOT RECOMMENDED to duplicate the value of `exception.message` in
	// `error.message`.
	//
	// `error.message` is NOT RECOMMENDED for metrics or spans due to its unbounded
	// cardinality and overlap with span status.
	ErrorMessageKey = attribute.Key("error.message")

	// ErrorTypeKey is the attribute Key conforming to the "error.type" semantic
	// conventions. It represents the describes a class of error the operation ended
	// with.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "timeout", "java.net.UnknownHostException",
	// "server_certificate_invalid", "500"
	// Note: The `error.type` SHOULD be predictable, and SHOULD have low
	// cardinality.
	//
	// When `error.type` is set to a type (e.g., an exception type), its
	// canonical class name identifying the type within the artifact SHOULD be used.
	//
	// Instrumentations SHOULD document the list of errors they report.
	//
	// The cardinality of `error.type` within one instrumentation library SHOULD be
	// low.
	// Telemetry consumers that aggregate data from multiple instrumentation
	// libraries and applications
	// should be prepared for `error.type` to have high cardinality at query time
	// when no
	// additional filters are applied.
	//
	// If the operation has completed successfully, instrumentations SHOULD NOT set
	// `error.type`.
	//
	// If a specific domain defines its own set of error identifiers (such as HTTP
	// or gRPC status codes),
	// it's RECOMMENDED to:
	//
	//   - Use a domain-specific attribute
	//   - Set `error.type` to capture all errors, regardless of whether they are
	//     defined within the domain-specific set or not.
	ErrorTypeKey = attribute.Key("error.type")
)

// ErrorMessage returns an attribute KeyValue conforming to the "error.message"
// semantic conventions. It represents a message providing more detail about an
// error in human-readable form.
func ErrorMessage(val string) attribute.KeyValue {
	return ErrorMessageKey.String(val)
}

// Enum values for error.type
var (
	// A fallback error value to be used when the instrumentation doesn't define a
	// custom value.
	//
	// Stability: stable
	ErrorTypeOther = ErrorTypeKey.String("_OTHER")
)

// Namespace: exception
const (
	// ExceptionMessageKey is the attribute Key conforming to the
	// "exception.message" semantic conventions. It represents the exception
	// message.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "Division by zero", "Can't convert 'int' object to str implicitly"
	ExceptionMessageKey = attribute.Key("exception.message")

	// ExceptionStacktraceKey is the attribute Key conforming to the
	// "exception.stacktrace" semantic conventions. It represents a stacktrace as a
	// string in the natural representation for the language runtime. The
	// representation is to be determined and documented by each language SIG.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: Exception in thread "main" java.lang.RuntimeException: Test
	// exception\n at com.example.GenerateTrace.methodB(GenerateTrace.java:13)\n at
	// com.example.GenerateTrace.methodA(GenerateTrace.java:9)\n at
	// com.example.GenerateTrace.main(GenerateTrace.java:5)
	ExceptionStacktraceKey = attribute.Key("exception.stacktrace")

	// ExceptionTypeKey is the attribute Key conforming to the "exception.type"
	// semantic conventions. It represents the type of the exception (its
	// fully-qualified class name, if applicable). The dynamic type of the exception
	// should be preferred over the static type in languages that support it.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "java.net.ConnectException", "OSError"
	ExceptionTypeKey = attribute.Key("exception.type")
)

// ExceptionMessage returns an attribute KeyValue conforming to the
// "exception.message" semantic conventions. It represents the exception message.
func ExceptionMessage(val string) attribute.KeyValue {
	return ExceptionMessageKey.String(val)
}

// ExceptionStacktrace returns an attribute KeyValue conforming to the
// "exception.stacktrace" semantic conventions. It represents a stacktrace as a
// string in the natural representation for the language runtime. The
// representation is to be determined and documented by each language SIG.
func ExceptionStacktrace(val string) attribute.KeyValue {
	return ExceptionStacktraceKey.String(val)
}

// ExceptionType returns an attribute KeyValue conforming to the "exception.type"
// semantic conventions. It represents the type of the exception (its
// fully-qualified class name, if applicable). The dynamic type of the exception
// should be preferred over the static type in languages that support it.
func ExceptionType(val string) attribute.KeyValue {
	return ExceptionTypeKey.String(val)
}

// Namespace: faas
const (
	// FaaSColdstartKey is the attribute Key conforming to the "faas.coldstart"
	// semantic conventions. It represents a boolean that is true if the serverless
	// function is executed for the first time (aka cold-start).
	//
	// Type: boolean
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	FaaSColdstartKey = attribute.Key("faas.coldstart")

	// FaaSCronKey is the attribute Key conforming to the "faas.cron" semantic
	// conventions. It represents a string containing the schedule period as
	// [Cron Expression].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 0/5 * * * ? *
	//
	// [Cron Expression]: https://docs.oracle.com/cd/E12058_01/doc/doc.1014/e12030/cron_expressions.htm
	FaaSCronKey = attribute.Key("faas.cron")

	// FaaSDocumentCollectionKey is the attribute Key conforming to the
	// "faas.document.collection" semantic conventions. It represents the name of
	// the source on which the triggering operation was performed. For example, in
	// Cloud Storage or S3 corresponds to the bucket name, and in Cosmos DB to the
	// database name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "myBucketName", "myDbName"
	FaaSDocumentCollectionKey = attribute.Key("faas.document.collection")

	// FaaSDocumentNameKey is the attribute Key conforming to the
	// "faas.document.name" semantic conventions. It represents the document
	// name/table subjected to the operation. For example, in Cloud Storage or S3 is
	// the name of the file, and in Cosmos DB the table name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "myFile.txt", "myTableName"
	FaaSDocumentNameKey = attribute.Key("faas.document.name")

	// FaaSDocumentOperationKey is the attribute Key conforming to the
	// "faas.document.operation" semantic conventions. It represents the describes
	// the type of the operation that was performed on the data.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	FaaSDocumentOperationKey = attribute.Key("faas.document.operation")

	// FaaSDocumentTimeKey is the attribute Key conforming to the
	// "faas.document.time" semantic conventions. It represents a string containing
	// the time when the data was accessed in the [ISO 8601] format expressed in
	// [UTC].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 2020-01-23T13:47:06Z
	//
	// [ISO 8601]: https://www.iso.org/iso-8601-date-and-time-format.html
	// [UTC]: https://www.w3.org/TR/NOTE-datetime
	FaaSDocumentTimeKey = attribute.Key("faas.document.time")

	// FaaSInstanceKey is the attribute Key conforming to the "faas.instance"
	// semantic conventions. It represents the execution environment ID as a string,
	// that will be potentially reused for other invocations to the same
	// function/function version.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2021/06/28/[$LATEST]2f399eb14537447da05ab2a2e39309de"
	// Note: - **AWS Lambda:** Use the (full) log stream name.
	FaaSInstanceKey = attribute.Key("faas.instance")

	// FaaSInvocationIDKey is the attribute Key conforming to the
	// "faas.invocation_id" semantic conventions. It represents the invocation ID of
	// the current function invocation.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: af9d5aa4-a685-4c5f-a22b-444f80b3cc28
	FaaSInvocationIDKey = attribute.Key("faas.invocation_id")

	// FaaSInvokedNameKey is the attribute Key conforming to the "faas.invoked_name"
	// semantic conventions. It represents the name of the invoked function.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: my-function
	// Note: SHOULD be equal to the `faas.name` resource attribute of the invoked
	// function.
	FaaSInvokedNameKey = attribute.Key("faas.invoked_name")

	// FaaSInvokedProviderKey is the attribute Key conforming to the
	// "faas.invoked_provider" semantic conventions. It represents the cloud
	// provider of the invoked function.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: SHOULD be equal to the `cloud.provider` resource attribute of the
	// invoked function.
	FaaSInvokedProviderKey = attribute.Key("faas.invoked_provider")

	// FaaSInvokedRegionKey is the attribute Key conforming to the
	// "faas.invoked_region" semantic conventions. It represents the cloud region of
	// the invoked function.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: eu-central-1
	// Note: SHOULD be equal to the `cloud.region` resource attribute of the invoked
	// function.
	FaaSInvokedRegionKey = attribute.Key("faas.invoked_region")

	// FaaSMaxMemoryKey is the attribute Key conforming to the "faas.max_memory"
	// semantic conventions. It represents the amount of memory available to the
	// serverless function converted to Bytes.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Note: It's recommended to set this attribute since e.g. too little memory can
	// easily stop a Java AWS Lambda function from working correctly. On AWS Lambda,
	// the environment variable `AWS_LAMBDA_FUNCTION_MEMORY_SIZE` provides this
	// information (which must be multiplied by 1,048,576).
	FaaSMaxMemoryKey = attribute.Key("faas.max_memory")

	// FaaSNameKey is the attribute Key conforming to the "faas.name" semantic
	// conventions. It represents the name of the single function that this runtime
	// instance executes.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "my-function", "myazurefunctionapp/some-function-name"
	// Note: This is the name of the function as configured/deployed on the FaaS
	// platform and is usually different from the name of the callback
	// function (which may be stored in the
	// [`code.namespace`/`code.function.name`]
	// span attributes).
	//
	// For some cloud providers, the above definition is ambiguous. The following
	// definition of function name MUST be used for this attribute
	// (and consequently the span name) for the listed cloud providers/products:
	//
	//   - **Azure:** The full name `<FUNCAPP>/<FUNC>`, i.e., function app name
	//     followed by a forward slash followed by the function name (this form
	//     can also be seen in the resource JSON for the function).
	//     This means that a span attribute MUST be used, as an Azure function
	//     app can host multiple functions that would usually share
	//     a TracerProvider (see also the `cloud.resource_id` attribute).
	//
	//
	// [`code.namespace`/`code.function.name`]: /docs/general/attributes.md#source-code-attributes
	FaaSNameKey = attribute.Key("faas.name")

	// FaaSTimeKey is the attribute Key conforming to the "faas.time" semantic
	// conventions. It represents a string containing the function invocation time
	// in the [ISO 8601] format expressed in [UTC].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 2020-01-23T13:47:06Z
	//
	// [ISO 8601]: https://www.iso.org/iso-8601-date-and-time-format.html
	// [UTC]: https://www.w3.org/TR/NOTE-datetime
	FaaSTimeKey = attribute.Key("faas.time")

	// FaaSTriggerKey is the attribute Key conforming to the "faas.trigger" semantic
	// conventions. It represents the type of the trigger which caused this function
	// invocation.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	FaaSTriggerKey = attribute.Key("faas.trigger")

	// FaaSVersionKey is the attribute Key conforming to the "faas.version" semantic
	// conventions. It represents the immutable version of the function being
	// executed.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "26", "pinkfroid-00002"
	// Note: Depending on the cloud provider and platform, use:
	//
	//   - **AWS Lambda:** The [function version]
	//     (an integer represented as a decimal string).
	//   - **Google Cloud Run (Services):** The [revision]
	//     (i.e., the function name plus the revision suffix).
	//   - **Google Cloud Functions:** The value of the
	//     [`K_REVISION` environment variable].
	//   - **Azure Functions:** Not applicable. Do not set this attribute.
	//
	//
	// [function version]: https://docs.aws.amazon.com/lambda/latest/dg/configuration-versions.html
	// [revision]: https://cloud.google.com/run/docs/managing/revisions
	// [`K_REVISION` environment variable]: https://cloud.google.com/functions/docs/env-var#runtime_environment_variables_set_automatically
	FaaSVersionKey = attribute.Key("faas.version")
)

// FaaSColdstart returns an attribute KeyValue conforming to the "faas.coldstart"
// semantic conventions. It represents a boolean that is true if the serverless
// function is executed for the first time (aka cold-start).
func FaaSColdstart(val bool) attribute.KeyValue {
	return FaaSColdstartKey.Bool(val)
}

// FaaSCron returns an attribute KeyValue conforming to the "faas.cron" semantic
// conventions. It represents a string containing the schedule period as
// [Cron Expression].
//
// [Cron Expression]: https://docs.oracle.com/cd/E12058_01/doc/doc.1014/e12030/cron_expressions.htm
func FaaSCron(val string) attribute.KeyValue {
	return FaaSCronKey.String(val)
}

// FaaSDocumentCollection returns an attribute KeyValue conforming to the
// "faas.document.collection" semantic conventions. It represents the name of the
// source on which the triggering operation was performed. For example, in Cloud
// Storage or S3 corresponds to the bucket name, and in Cosmos DB to the database
// name.
func FaaSDocumentCollection(val string) attribute.KeyValue {
	return FaaSDocumentCollectionKey.String(val)
}

// FaaSDocumentName returns an attribute KeyValue conforming to the
// "faas.document.name" semantic conventions. It represents the document
// name/table subjected to the operation. For example, in Cloud Storage or S3 is
// the name of the file, and in Cosmos DB the table name.
func FaaSDocumentName(val string) attribute.KeyValue {
	return FaaSDocumentNameKey.String(val)
}

// FaaSDocumentTime returns an attribute KeyValue conforming to the
// "faas.document.time" semantic conventions. It represents a string containing
// the time when the data was accessed in the [ISO 8601] format expressed in
// [UTC].
//
// [ISO 8601]: https://www.iso.org/iso-8601-date-and-time-format.html
// [UTC]: https://www.w3.org/TR/NOTE-datetime
func FaaSDocumentTime(val string) attribute.KeyValue {
	return FaaSDocumentTimeKey.String(val)
}

// FaaSInstance returns an attribute KeyValue conforming to the "faas.instance"
// semantic conventions. It represents the execution environment ID as a string,
// that will be potentially reused for other invocations to the same
// function/function version.
func FaaSInstance(val string) attribute.KeyValue {
	return FaaSInstanceKey.String(val)
}

// FaaSInvocationID returns an attribute KeyValue conforming to the
// "faas.invocation_id" semantic conventions. It represents the invocation ID of
// the current function invocation.
func FaaSInvocationID(val string) attribute.KeyValue {
	return FaaSInvocationIDKey.String(val)
}

// FaaSInvokedName returns an attribute KeyValue conforming to the
// "faas.invoked_name" semantic conventions. It represents the name of the
// invoked function.
func FaaSInvokedName(val string) attribute.KeyValue {
	return FaaSInvokedNameKey.String(val)
}

// FaaSInvokedRegion returns an attribute KeyValue conforming to the
// "faas.invoked_region" semantic conventions. It represents the cloud region of
// the invoked function.
func FaaSInvokedRegion(val string) attribute.KeyValue {
	return FaaSInvokedRegionKey.String(val)
}

// FaaSMaxMemory returns an attribute KeyValue conforming to the
// "faas.max_memory" semantic conventions. It represents the amount of memory
// available to the serverless function converted to Bytes.
func FaaSMaxMemory(val int) attribute.KeyValue {
	return FaaSMaxMemoryKey.Int(val)
}

// FaaSName returns an attribute KeyValue conforming to the "faas.name" semantic
// conventions. It represents the name of the single function that this runtime
// instance executes.
func FaaSName(val string) attribute.KeyValue {
	return FaaSNameKey.String(val)
}

// FaaSTime returns an attribute KeyValue conforming to the "faas.time" semantic
// conventions. It represents a string containing the function invocation time in
// the [ISO 8601] format expressed in [UTC].
//
// [ISO 8601]: https://www.iso.org/iso-8601-date-and-time-format.html
// [UTC]: https://www.w3.org/TR/NOTE-datetime
func FaaSTime(val string) attribute.KeyValue {
	return FaaSTimeKey.String(val)
}

// FaaSVersion returns an attribute KeyValue conforming to the "faas.version"
// semantic conventions. It represents the immutable version of the function
// being executed.
func FaaSVersion(val string) attribute.KeyValue {
	return FaaSVersionKey.String(val)
}

// Enum values for faas.document.operation
var (
	// When a new object is created.
	// Stability: development
	FaaSDocumentOperationInsert = FaaSDocumentOperationKey.String("insert")
	// When an object is modified.
	// Stability: development
	FaaSDocumentOperationEdit = FaaSDocumentOperationKey.String("edit")
	// When an object is deleted.
	// Stability: development
	FaaSDocumentOperationDelete = FaaSDocumentOperationKey.String("delete")
)

// Enum values for faas.invoked_provider
var (
	// Alibaba Cloud
	// Stability: development
	FaaSInvokedProviderAlibabaCloud = FaaSInvokedProviderKey.String("alibaba_cloud")
	// Amazon Web Services
	// Stability: development
	FaaSInvokedProviderAWS = FaaSInvokedProviderKey.String("aws")
	// Microsoft Azure
	// Stability: development
	FaaSInvokedProviderAzure = FaaSInvokedProviderKey.String("azure")
	// Google Cloud Platform
	// Stability: development
	FaaSInvokedProviderGCP = FaaSInvokedProviderKey.String("gcp")
	// Tencent Cloud
	// Stability: development
	FaaSInvokedProviderTencentCloud = FaaSInvokedProviderKey.String("tencent_cloud")
)

// Enum values for faas.trigger
var (
	// A response to some data source operation such as a database or filesystem
	// read/write
	// Stability: development
	FaaSTriggerDatasource = FaaSTriggerKey.String("datasource")
	// To provide an answer to an inbound HTTP request
	// Stability: development
	FaaSTriggerHTTP = FaaSTriggerKey.String("http")
	// A function is set to be executed when messages are sent to a messaging system
	// Stability: development
	FaaSTriggerPubSub = FaaSTriggerKey.String("pubsub")
	// A function is scheduled to be executed regularly
	// Stability: development
	FaaSTriggerTimer = FaaSTriggerKey.String("timer")
	// If none of the others apply
	// Stability: development
	FaaSTriggerOther = FaaSTriggerKey.String("other")
)

// Namespace: feature_flag
const (
	// FeatureFlagContextIDKey is the attribute Key conforming to the
	// "feature_flag.context.id" semantic conventions. It represents the unique
	// identifier for the flag evaluation context. For example, the targeting key.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Release_Candidate
	//
	// Examples: "5157782b-2203-4c80-a857-dbbd5e7761db"
	FeatureFlagContextIDKey = attribute.Key("feature_flag.context.id")

	// FeatureFlagKeyKey is the attribute Key conforming to the "feature_flag.key"
	// semantic conventions. It represents the lookup key of the feature flag.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Release_Candidate
	//
	// Examples: "logo-color"
	FeatureFlagKeyKey = attribute.Key("feature_flag.key")

	// FeatureFlagProviderNameKey is the attribute Key conforming to the
	// "feature_flag.provider.name" semantic conventions. It represents the
	// identifies the feature flag provider.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Release_Candidate
	//
	// Examples: "Flag Manager"
	FeatureFlagProviderNameKey = attribute.Key("feature_flag.provider.name")

	// FeatureFlagResultReasonKey is the attribute Key conforming to the
	// "feature_flag.result.reason" semantic conventions. It represents the reason
	// code which shows how a feature flag value was determined.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Release_Candidate
	//
	// Examples: "static", "targeting_match", "error", "default"
	FeatureFlagResultReasonKey = attribute.Key("feature_flag.result.reason")

	// FeatureFlagResultValueKey is the attribute Key conforming to the
	// "feature_flag.result.value" semantic conventions. It represents the evaluated
	// value of the feature flag.
	//
	// Type: any
	// RequirementLevel: Recommended
	// Stability: Release_Candidate
	//
	// Examples: "#ff0000", true, 3
	// Note: With some feature flag providers, feature flag results can be quite
	// large or contain private or sensitive details.
	// Because of this, `feature_flag.result.variant` is often the preferred
	// attribute if it is available.
	//
	// It may be desirable to redact or otherwise limit the size and scope of
	// `feature_flag.result.value` if possible.
	// Because the evaluated flag value is unstructured and may be any type, it is
	// left to the instrumentation author to determine how best to achieve this.
	FeatureFlagResultValueKey = attribute.Key("feature_flag.result.value")

	// FeatureFlagResultVariantKey is the attribute Key conforming to the
	// "feature_flag.result.variant" semantic conventions. It represents a semantic
	// identifier for an evaluated flag value.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Release_Candidate
	//
	// Examples: "red", "true", "on"
	// Note: A semantic identifier, commonly referred to as a variant, provides a
	// means
	// for referring to a value without including the value itself. This can
	// provide additional context for understanding the meaning behind a value.
	// For example, the variant `red` maybe be used for the value `#c05543`.
	FeatureFlagResultVariantKey = attribute.Key("feature_flag.result.variant")

	// FeatureFlagSetIDKey is the attribute Key conforming to the
	// "feature_flag.set.id" semantic conventions. It represents the identifier of
	// the [flag set] to which the feature flag belongs.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Release_Candidate
	//
	// Examples: "proj-1", "ab98sgs", "service1/dev"
	//
	// [flag set]: https://openfeature.dev/specification/glossary/#flag-set
	FeatureFlagSetIDKey = attribute.Key("feature_flag.set.id")

	// FeatureFlagVersionKey is the attribute Key conforming to the
	// "feature_flag.version" semantic conventions. It represents the version of the
	// ruleset used during the evaluation. This may be any stable value which
	// uniquely identifies the ruleset.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Release_Candidate
	//
	// Examples: "1", "01ABCDEF"
	FeatureFlagVersionKey = attribute.Key("feature_flag.version")
)

// FeatureFlagContextID returns an attribute KeyValue conforming to the
// "feature_flag.context.id" semantic conventions. It represents the unique
// identifier for the flag evaluation context. For example, the targeting key.
func FeatureFlagContextID(val string) attribute.KeyValue {
	return FeatureFlagContextIDKey.String(val)
}

// FeatureFlagKey returns an attribute KeyValue conforming to the
// "feature_flag.key" semantic conventions. It represents the lookup key of the
// feature flag.
func FeatureFlagKey(val string) attribute.KeyValue {
	return FeatureFlagKeyKey.String(val)
}

// FeatureFlagProviderName returns an attribute KeyValue conforming to the
// "feature_flag.provider.name" semantic conventions. It represents the
// identifies the feature flag provider.
func FeatureFlagProviderName(val string) attribute.KeyValue {
	return FeatureFlagProviderNameKey.String(val)
}

// FeatureFlagResultVariant returns an attribute KeyValue conforming to the
// "feature_flag.result.variant" semantic conventions. It represents a semantic
// identifier for an evaluated flag value.
func FeatureFlagResultVariant(val string) attribute.KeyValue {
	return FeatureFlagResultVariantKey.String(val)
}

// FeatureFlagSetID returns an attribute KeyValue conforming to the
// "feature_flag.set.id" semantic conventions. It represents the identifier of
// the [flag set] to which the feature flag belongs.
//
// [flag set]: https://openfeature.dev/specification/glossary/#flag-set
func FeatureFlagSetID(val string) attribute.KeyValue {
	return FeatureFlagSetIDKey.String(val)
}

// FeatureFlagVersion returns an attribute KeyValue conforming to the
// "feature_flag.version" semantic conventions. It represents the version of the
// ruleset used during the evaluation. This may be any stable value which
// uniquely identifies the ruleset.
func FeatureFlagVersion(val string) attribute.KeyValue {
	return FeatureFlagVersionKey.String(val)
}

// Enum values for feature_flag.result.reason
var (
	// The resolved value is static (no dynamic evaluation).
	// Stability: release_candidate
	FeatureFlagResultReasonStatic = FeatureFlagResultReasonKey.String("static")
	// The resolved value fell back to a pre-configured value (no dynamic evaluation
	// occurred or dynamic evaluation yielded no result).
	// Stability: release_candidate
	FeatureFlagResultReasonDefault = FeatureFlagResultReasonKey.String("default")
	// The resolved value was the result of a dynamic evaluation, such as a rule or
	// specific user-targeting.
	// Stability: release_candidate
	FeatureFlagResultReasonTargetingMatch = FeatureFlagResultReasonKey.String("targeting_match")
	// The resolved value was the result of pseudorandom assignment.
	// Stability: release_candidate
	FeatureFlagResultReasonSplit = FeatureFlagResultReasonKey.String("split")
	// The resolved value was retrieved from cache.
	// Stability: release_candidate
	FeatureFlagResultReasonCached = FeatureFlagResultReasonKey.String("cached")
	// The resolved value was the result of the flag being disabled in the
	// management system.
	// Stability: release_candidate
	FeatureFlagResultReasonDisabled = FeatureFlagResultReasonKey.String("disabled")
	// The reason for the resolved value could not be determined.
	// Stability: release_candidate
	FeatureFlagResultReasonUnknown = FeatureFlagResultReasonKey.String("unknown")
	// The resolved value is non-authoritative or possibly out of date
	// Stability: release_candidate
	FeatureFlagResultReasonStale = FeatureFlagResultReasonKey.String("stale")
	// The resolved value was the result of an error.
	// Stability: release_candidate
	FeatureFlagResultReasonError = FeatureFlagResultReasonKey.String("error")
)

// Namespace: file
const (
	// FileAccessedKey is the attribute Key conforming to the "file.accessed"
	// semantic conventions. It represents the time when the file was last accessed,
	// in ISO 8601 format.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2021-01-01T12:00:00Z"
	// Note: This attribute might not be supported by some file systems — NFS,
	// FAT32, in embedded OS, etc.
	FileAccessedKey = attribute.Key("file.accessed")

	// FileAttributesKey is the attribute Key conforming to the "file.attributes"
	// semantic conventions. It represents the array of file attributes.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "readonly", "hidden"
	// Note: Attributes names depend on the OS or file system. Here’s a
	// non-exhaustive list of values expected for this attribute: `archive`,
	// `compressed`, `directory`, `encrypted`, `execute`, `hidden`, `immutable`,
	// `journaled`, `read`, `readonly`, `symbolic link`, `system`, `temporary`,
	// `write`.
	FileAttributesKey = attribute.Key("file.attributes")

	// FileChangedKey is the attribute Key conforming to the "file.changed" semantic
	// conventions. It represents the time when the file attributes or metadata was
	// last changed, in ISO 8601 format.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2021-01-01T12:00:00Z"
	// Note: `file.changed` captures the time when any of the file's properties or
	// attributes (including the content) are changed, while `file.modified`
	// captures the timestamp when the file content is modified.
	FileChangedKey = attribute.Key("file.changed")

	// FileCreatedKey is the attribute Key conforming to the "file.created" semantic
	// conventions. It represents the time when the file was created, in ISO 8601
	// format.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2021-01-01T12:00:00Z"
	// Note: This attribute might not be supported by some file systems — NFS,
	// FAT32, in embedded OS, etc.
	FileCreatedKey = attribute.Key("file.created")

	// FileDirectoryKey is the attribute Key conforming to the "file.directory"
	// semantic conventions. It represents the directory where the file is located.
	// It should include the drive letter, when appropriate.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "/home/user", "C:\Program Files\MyApp"
	FileDirectoryKey = attribute.Key("file.directory")

	// FileExtensionKey is the attribute Key conforming to the "file.extension"
	// semantic conventions. It represents the file extension, excluding the leading
	// dot.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "png", "gz"
	// Note: When the file name has multiple extensions (example.tar.gz), only the
	// last one should be captured ("gz", not "tar.gz").
	FileExtensionKey = attribute.Key("file.extension")

	// FileForkNameKey is the attribute Key conforming to the "file.fork_name"
	// semantic conventions. It represents the name of the fork. A fork is
	// additional data associated with a filesystem object.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Zone.Identifier"
	// Note: On Linux, a resource fork is used to store additional data with a
	// filesystem object. A file always has at least one fork for the data portion,
	// and additional forks may exist.
	// On NTFS, this is analogous to an Alternate Data Stream (ADS), and the default
	// data stream for a file is just called $DATA. Zone.Identifier is commonly used
	// by Windows to track contents downloaded from the Internet. An ADS is
	// typically of the form: C:\path\to\filename.extension:some_fork_name, and
	// some_fork_name is the value that should populate `fork_name`.
	// `filename.extension` should populate `file.name`, and `extension` should
	// populate `file.extension`. The full path, `file.path`, will include the fork
	// name.
	FileForkNameKey = attribute.Key("file.fork_name")

	// FileGroupIDKey is the attribute Key conforming to the "file.group.id"
	// semantic conventions. It represents the primary Group ID (GID) of the file.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "1000"
	FileGroupIDKey = attribute.Key("file.group.id")

	// FileGroupNameKey is the attribute Key conforming to the "file.group.name"
	// semantic conventions. It represents the primary group name of the file.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "users"
	FileGroupNameKey = attribute.Key("file.group.name")

	// FileInodeKey is the attribute Key conforming to the "file.inode" semantic
	// conventions. It represents the inode representing the file in the filesystem.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "256383"
	FileInodeKey = attribute.Key("file.inode")

	// FileModeKey is the attribute Key conforming to the "file.mode" semantic
	// conventions. It represents the mode of the file in octal representation.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "0640"
	FileModeKey = attribute.Key("file.mode")

	// FileModifiedKey is the attribute Key conforming to the "file.modified"
	// semantic conventions. It represents the time when the file content was last
	// modified, in ISO 8601 format.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2021-01-01T12:00:00Z"
	FileModifiedKey = attribute.Key("file.modified")

	// FileNameKey is the attribute Key conforming to the "file.name" semantic
	// conventions. It represents the name of the file including the extension,
	// without the directory.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "example.png"
	FileNameKey = attribute.Key("file.name")

	// FileOwnerIDKey is the attribute Key conforming to the "file.owner.id"
	// semantic conventions. It represents the user ID (UID) or security identifier
	// (SID) of the file owner.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "1000"
	FileOwnerIDKey = attribute.Key("file.owner.id")

	// FileOwnerNameKey is the attribute Key conforming to the "file.owner.name"
	// semantic conventions. It represents the username of the file owner.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "root"
	FileOwnerNameKey = attribute.Key("file.owner.name")

	// FilePathKey is the attribute Key conforming to the "file.path" semantic
	// conventions. It represents the full path to the file, including the file
	// name. It should include the drive letter, when appropriate.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "/home/alice/example.png", "C:\Program Files\MyApp\myapp.exe"
	FilePathKey = attribute.Key("file.path")

	// FileSizeKey is the attribute Key conforming to the "file.size" semantic
	// conventions. It represents the file size in bytes.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	FileSizeKey = attribute.Key("file.size")

	// FileSymbolicLinkTargetPathKey is the attribute Key conforming to the
	// "file.symbolic_link.target_path" semantic conventions. It represents the path
	// to the target of a symbolic link.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "/usr/bin/python3"
	// Note: This attribute is only applicable to symbolic links.
	FileSymbolicLinkTargetPathKey = attribute.Key("file.symbolic_link.target_path")
)

// FileAccessed returns an attribute KeyValue conforming to the "file.accessed"
// semantic conventions. It represents the time when the file was last accessed,
// in ISO 8601 format.
func FileAccessed(val string) attribute.KeyValue {
	return FileAccessedKey.String(val)
}

// FileAttributes returns an attribute KeyValue conforming to the
// "file.attributes" semantic conventions. It represents the array of file
// attributes.
func FileAttributes(val ...string) attribute.KeyValue {
	return FileAttributesKey.StringSlice(val)
}

// FileChanged returns an attribute KeyValue conforming to the "file.changed"
// semantic conventions. It represents the time when the file attributes or
// metadata was last changed, in ISO 8601 format.
func FileChanged(val string) attribute.KeyValue {
	return FileChangedKey.String(val)
}

// FileCreated returns an attribute KeyValue conforming to the "file.created"
// semantic conventions. It represents the time when the file was created, in ISO
// 8601 format.
func FileCreated(val string) attribute.KeyValue {
	return FileCreatedKey.String(val)
}

// FileDirectory returns an attribute KeyValue conforming to the "file.directory"
// semantic conventions. It represents the directory where the file is located.
// It should include the drive letter, when appropriate.
func FileDirectory(val string) attribute.KeyValue {
	return FileDirectoryKey.String(val)
}

// FileExtension returns an attribute KeyValue conforming to the "file.extension"
// semantic conventions. It represents the file extension, excluding the leading
// dot.
func FileExtension(val string) attribute.KeyValue {
	return FileExtensionKey.String(val)
}

// FileForkName returns an attribute KeyValue conforming to the "file.fork_name"
// semantic conventions. It represents the name of the fork. A fork is additional
// data associated with a filesystem object.
func FileForkName(val string) attribute.KeyValue {
	return FileForkNameKey.String(val)
}

// FileGroupID returns an attribute KeyValue conforming to the "file.group.id"
// semantic conventions. It represents the primary Group ID (GID) of the file.
func FileGroupID(val string) attribute.KeyValue {
	return FileGroupIDKey.String(val)
}

// FileGroupName returns an attribute KeyValue conforming to the
// "file.group.name" semantic conventions. It represents the primary group name
// of the file.
func FileGroupName(val string) attribute.KeyValue {
	return FileGroupNameKey.String(val)
}

// FileInode returns an attribute KeyValue conforming to the "file.inode"
// semantic conventions. It represents the inode representing the file in the
// filesystem.
func FileInode(val string) attribute.KeyValue {
	return FileInodeKey.String(val)
}

// FileMode returns an attribute KeyValue conforming to the "file.mode" semantic
// conventions. It represents the mode of the file in octal representation.
func FileMode(val string) attribute.KeyValue {
	return FileModeKey.String(val)
}

// FileModified returns an attribute KeyValue conforming to the "file.modified"
// semantic conventions. It represents the time when the file content was last
// modified, in ISO 8601 format.
func FileModified(val string) attribute.KeyValue {
	return FileModifiedKey.String(val)
}

// FileName returns an attribute KeyValue conforming to the "file.name" semantic
// conventions. It represents the name of the file including the extension,
// without the directory.
func FileName(val string) attribute.KeyValue {
	return FileNameKey.String(val)
}

// FileOwnerID returns an attribute KeyValue conforming to the "file.owner.id"
// semantic conventions. It represents the user ID (UID) or security identifier
// (SID) of the file owner.
func FileOwnerID(val string) attribute.KeyValue {
	return FileOwnerIDKey.String(val)
}

// FileOwnerName returns an attribute KeyValue conforming to the
// "file.owner.name" semantic conventions. It represents the username of the file
// owner.
func FileOwnerName(val string) attribute.KeyValue {
	return FileOwnerNameKey.String(val)
}

// FilePath returns an attribute KeyValue conforming to the "file.path" semantic
// conventions. It represents the full path to the file, including the file name.
// It should include the drive letter, when appropriate.
func FilePath(val string) attribute.KeyValue {
	return FilePathKey.String(val)
}

// FileSize returns an attribute KeyValue conforming to the "file.size" semantic
// conventions. It represents the file size in bytes.
func FileSize(val int) attribute.KeyValue {
	return FileSizeKey.Int(val)
}

// FileSymbolicLinkTargetPath returns an attribute KeyValue conforming to the
// "file.symbolic_link.target_path" semantic conventions. It represents the path
// to the target of a symbolic link.
func FileSymbolicLinkTargetPath(val string) attribute.KeyValue {
	return FileSymbolicLinkTargetPathKey.String(val)
}

// Namespace: gcp
const (
	// GCPAppHubApplicationContainerKey is the attribute Key conforming to the
	// "gcp.apphub.application.container" semantic conventions. It represents the
	// container within GCP where the AppHub application is defined.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "projects/my-container-project"
	GCPAppHubApplicationContainerKey = attribute.Key("gcp.apphub.application.container")

	// GCPAppHubApplicationIDKey is the attribute Key conforming to the
	// "gcp.apphub.application.id" semantic conventions. It represents the name of
	// the application as configured in AppHub.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "my-application"
	GCPAppHubApplicationIDKey = attribute.Key("gcp.apphub.application.id")

	// GCPAppHubApplicationLocationKey is the attribute Key conforming to the
	// "gcp.apphub.application.location" semantic conventions. It represents the GCP
	// zone or region where the application is defined.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "us-central1"
	GCPAppHubApplicationLocationKey = attribute.Key("gcp.apphub.application.location")

	// GCPAppHubServiceCriticalityTypeKey is the attribute Key conforming to the
	// "gcp.apphub.service.criticality_type" semantic conventions. It represents the
	// criticality of a service indicates its importance to the business.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: [See AppHub type enum]
	//
	// [See AppHub type enum]: https://cloud.google.com/app-hub/docs/reference/rest/v1/Attributes#type
	GCPAppHubServiceCriticalityTypeKey = attribute.Key("gcp.apphub.service.criticality_type")

	// GCPAppHubServiceEnvironmentTypeKey is the attribute Key conforming to the
	// "gcp.apphub.service.environment_type" semantic conventions. It represents the
	// environment of a service is the stage of a software lifecycle.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: [See AppHub environment type]
	//
	// [See AppHub environment type]: https://cloud.google.com/app-hub/docs/reference/rest/v1/Attributes#type_1
	GCPAppHubServiceEnvironmentTypeKey = attribute.Key("gcp.apphub.service.environment_type")

	// GCPAppHubServiceIDKey is the attribute Key conforming to the
	// "gcp.apphub.service.id" semantic conventions. It represents the name of the
	// service as configured in AppHub.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "my-service"
	GCPAppHubServiceIDKey = attribute.Key("gcp.apphub.service.id")

	// GCPAppHubWorkloadCriticalityTypeKey is the attribute Key conforming to the
	// "gcp.apphub.workload.criticality_type" semantic conventions. It represents
	// the criticality of a workload indicates its importance to the business.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: [See AppHub type enum]
	//
	// [See AppHub type enum]: https://cloud.google.com/app-hub/docs/reference/rest/v1/Attributes#type
	GCPAppHubWorkloadCriticalityTypeKey = attribute.Key("gcp.apphub.workload.criticality_type")

	// GCPAppHubWorkloadEnvironmentTypeKey is the attribute Key conforming to the
	// "gcp.apphub.workload.environment_type" semantic conventions. It represents
	// the environment of a workload is the stage of a software lifecycle.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: [See AppHub environment type]
	//
	// [See AppHub environment type]: https://cloud.google.com/app-hub/docs/reference/rest/v1/Attributes#type_1
	GCPAppHubWorkloadEnvironmentTypeKey = attribute.Key("gcp.apphub.workload.environment_type")

	// GCPAppHubWorkloadIDKey is the attribute Key conforming to the
	// "gcp.apphub.workload.id" semantic conventions. It represents the name of the
	// workload as configured in AppHub.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "my-workload"
	GCPAppHubWorkloadIDKey = attribute.Key("gcp.apphub.workload.id")

	// GCPClientServiceKey is the attribute Key conforming to the
	// "gcp.client.service" semantic conventions. It represents the identifies the
	// Google Cloud service for which the official client library is intended.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "appengine", "run", "firestore", "alloydb", "spanner"
	// Note: Intended to be a stable identifier for Google Cloud client libraries
	// that is uniform across implementation languages. The value should be derived
	// from the canonical service domain for the service; for example,
	// 'foo.googleapis.com' should result in a value of 'foo'.
	GCPClientServiceKey = attribute.Key("gcp.client.service")

	// GCPCloudRunJobExecutionKey is the attribute Key conforming to the
	// "gcp.cloud_run.job.execution" semantic conventions. It represents the name of
	// the Cloud Run [execution] being run for the Job, as set by the
	// [`CLOUD_RUN_EXECUTION`] environment variable.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "job-name-xxxx", "sample-job-mdw84"
	//
	// [execution]: https://cloud.google.com/run/docs/managing/job-executions
	// [`CLOUD_RUN_EXECUTION`]: https://cloud.google.com/run/docs/container-contract#jobs-env-vars
	GCPCloudRunJobExecutionKey = attribute.Key("gcp.cloud_run.job.execution")

	// GCPCloudRunJobTaskIndexKey is the attribute Key conforming to the
	// "gcp.cloud_run.job.task_index" semantic conventions. It represents the index
	// for a task within an execution as provided by the [`CLOUD_RUN_TASK_INDEX`]
	// environment variable.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 0, 1
	//
	// [`CLOUD_RUN_TASK_INDEX`]: https://cloud.google.com/run/docs/container-contract#jobs-env-vars
	GCPCloudRunJobTaskIndexKey = attribute.Key("gcp.cloud_run.job.task_index")

	// GCPGCEInstanceHostnameKey is the attribute Key conforming to the
	// "gcp.gce.instance.hostname" semantic conventions. It represents the hostname
	// of a GCE instance. This is the full value of the default or [custom hostname]
	// .
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "my-host1234.example.com",
	// "sample-vm.us-west1-b.c.my-project.internal"
	//
	// [custom hostname]: https://cloud.google.com/compute/docs/instances/custom-hostname-vm
	GCPGCEInstanceHostnameKey = attribute.Key("gcp.gce.instance.hostname")

	// GCPGCEInstanceNameKey is the attribute Key conforming to the
	// "gcp.gce.instance.name" semantic conventions. It represents the instance name
	// of a GCE instance. This is the value provided by `host.name`, the visible
	// name of the instance in the Cloud Console UI, and the prefix for the default
	// hostname of the instance as defined by the [default internal DNS name].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "instance-1", "my-vm-name"
	//
	// [default internal DNS name]: https://cloud.google.com/compute/docs/internal-dns#instance-fully-qualified-domain-names
	GCPGCEInstanceNameKey = attribute.Key("gcp.gce.instance.name")
)

// GCPAppHubApplicationContainer returns an attribute KeyValue conforming to the
// "gcp.apphub.application.container" semantic conventions. It represents the
// container within GCP where the AppHub application is defined.
func GCPAppHubApplicationContainer(val string) attribute.KeyValue {
	return GCPAppHubApplicationContainerKey.String(val)
}

// GCPAppHubApplicationID returns an attribute KeyValue conforming to the
// "gcp.apphub.application.id" semantic conventions. It represents the name of
// the application as configured in AppHub.
func GCPAppHubApplicationID(val string) attribute.KeyValue {
	return GCPAppHubApplicationIDKey.String(val)
}

// GCPAppHubApplicationLocation returns an attribute KeyValue conforming to the
// "gcp.apphub.application.location" semantic conventions. It represents the GCP
// zone or region where the application is defined.
func GCPAppHubApplicationLocation(val string) attribute.KeyValue {
	return GCPAppHubApplicationLocationKey.String(val)
}

// GCPAppHubServiceID returns an attribute KeyValue conforming to the
// "gcp.apphub.service.id" semantic conventions. It represents the name of the
// service as configured in AppHub.
func GCPAppHubServiceID(val string) attribute.KeyValue {
	return GCPAppHubServiceIDKey.String(val)
}

// GCPAppHubWorkloadID returns an attribute KeyValue conforming to the
// "gcp.apphub.workload.id" semantic conventions. It represents the name of the
// workload as configured in AppHub.
func GCPAppHubWorkloadID(val string) attribute.KeyValue {
	return GCPAppHubWorkloadIDKey.String(val)
}

// GCPClientService returns an attribute KeyValue conforming to the
// "gcp.client.service" semantic conventions. It represents the identifies the
// Google Cloud service for which the official client library is intended.
func GCPClientService(val string) attribute.KeyValue {
	return GCPClientServiceKey.String(val)
}

// GCPCloudRunJobExecution returns an attribute KeyValue conforming to the
// "gcp.cloud_run.job.execution" semantic conventions. It represents the name of
// the Cloud Run [execution] being run for the Job, as set by the
// [`CLOUD_RUN_EXECUTION`] environment variable.
//
// [execution]: https://cloud.google.com/run/docs/managing/job-executions
// [`CLOUD_RUN_EXECUTION`]: https://cloud.google.com/run/docs/container-contract#jobs-env-vars
func GCPCloudRunJobExecution(val string) attribute.KeyValue {
	return GCPCloudRunJobExecutionKey.String(val)
}

// GCPCloudRunJobTaskIndex returns an attribute KeyValue conforming to the
// "gcp.cloud_run.job.task_index" semantic conventions. It represents the index
// for a task within an execution as provided by the [`CLOUD_RUN_TASK_INDEX`]
// environment variable.
//
// [`CLOUD_RUN_TASK_INDEX`]: https://cloud.google.com/run/docs/container-contract#jobs-env-vars
func GCPCloudRunJobTaskIndex(val int) attribute.KeyValue {
	return GCPCloudRunJobTaskIndexKey.Int(val)
}

// GCPGCEInstanceHostname returns an attribute KeyValue conforming to the
// "gcp.gce.instance.hostname" semantic conventions. It represents the hostname
// of a GCE instance. This is the full value of the default or [custom hostname]
// .
//
// [custom hostname]: https://cloud.google.com/compute/docs/instances/custom-hostname-vm
func GCPGCEInstanceHostname(val string) attribute.KeyValue {
	return GCPGCEInstanceHostnameKey.String(val)
}

// GCPGCEInstanceName returns an attribute KeyValue conforming to the
// "gcp.gce.instance.name" semantic conventions. It represents the instance name
// of a GCE instance. This is the value provided by `host.name`, the visible name
// of the instance in the Cloud Console UI, and the prefix for the default
// hostname of the instance as defined by the [default internal DNS name].
//
// [default internal DNS name]: https://cloud.google.com/compute/docs/internal-dns#instance-fully-qualified-domain-names
func GCPGCEInstanceName(val string) attribute.KeyValue {
	return GCPGCEInstanceNameKey.String(val)
}

// Enum values for gcp.apphub.service.criticality_type
var (
	// Mission critical service.
	// Stability: development
	GCPAppHubServiceCriticalityTypeMissionCritical = GCPAppHubServiceCriticalityTypeKey.String("MISSION_CRITICAL")
	// High impact.
	// Stability: development
	GCPAppHubServiceCriticalityTypeHigh = GCPAppHubServiceCriticalityTypeKey.String("HIGH")
	// Medium impact.
	// Stability: development
	GCPAppHubServiceCriticalityTypeMedium = GCPAppHubServiceCriticalityTypeKey.String("MEDIUM")
	// Low impact.
	// Stability: development
	GCPAppHubServiceCriticalityTypeLow = GCPAppHubServiceCriticalityTypeKey.String("LOW")
)

// Enum values for gcp.apphub.service.environment_type
var (
	// Production environment.
	// Stability: development
	GCPAppHubServiceEnvironmentTypeProduction = GCPAppHubServiceEnvironmentTypeKey.String("PRODUCTION")
	// Staging environment.
	// Stability: development
	GCPAppHubServiceEnvironmentTypeStaging = GCPAppHubServiceEnvironmentTypeKey.String("STAGING")
	// Test environment.
	// Stability: development
	GCPAppHubServiceEnvironmentTypeTest = GCPAppHubServiceEnvironmentTypeKey.String("TEST")
	// Development environment.
	// Stability: development
	GCPAppHubServiceEnvironmentTypeDevelopment = GCPAppHubServiceEnvironmentTypeKey.String("DEVELOPMENT")
)

// Enum values for gcp.apphub.workload.criticality_type
var (
	// Mission critical service.
	// Stability: development
	GCPAppHubWorkloadCriticalityTypeMissionCritical = GCPAppHubWorkloadCriticalityTypeKey.String("MISSION_CRITICAL")
	// High impact.
	// Stability: development
	GCPAppHubWorkloadCriticalityTypeHigh = GCPAppHubWorkloadCriticalityTypeKey.String("HIGH")
	// Medium impact.
	// Stability: development
	GCPAppHubWorkloadCriticalityTypeMedium = GCPAppHubWorkloadCriticalityTypeKey.String("MEDIUM")
	// Low impact.
	// Stability: development
	GCPAppHubWorkloadCriticalityTypeLow = GCPAppHubWorkloadCriticalityTypeKey.String("LOW")
)

// Enum values for gcp.apphub.workload.environment_type
var (
	// Production environment.
	// Stability: development
	GCPAppHubWorkloadEnvironmentTypeProduction = GCPAppHubWorkloadEnvironmentTypeKey.String("PRODUCTION")
	// Staging environment.
	// Stability: development
	GCPAppHubWorkloadEnvironmentTypeStaging = GCPAppHubWorkloadEnvironmentTypeKey.String("STAGING")
	// Test environment.
	// Stability: development
	GCPAppHubWorkloadEnvironmentTypeTest = GCPAppHubWorkloadEnvironmentTypeKey.String("TEST")
	// Development environment.
	// Stability: development
	GCPAppHubWorkloadEnvironmentTypeDevelopment = GCPAppHubWorkloadEnvironmentTypeKey.String("DEVELOPMENT")
)

// Namespace: gen_ai
const (
	// GenAIAgentDescriptionKey is the attribute Key conforming to the
	// "gen_ai.agent.description" semantic conventions. It represents the free-form
	// description of the GenAI agent provided by the application.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Helps with math problems", "Generates fiction stories"
	GenAIAgentDescriptionKey = attribute.Key("gen_ai.agent.description")

	// GenAIAgentIDKey is the attribute Key conforming to the "gen_ai.agent.id"
	// semantic conventions. It represents the unique identifier of the GenAI agent.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "asst_5j66UpCpwteGg4YSxUnt7lPY"
	GenAIAgentIDKey = attribute.Key("gen_ai.agent.id")

	// GenAIAgentNameKey is the attribute Key conforming to the "gen_ai.agent.name"
	// semantic conventions. It represents the human-readable name of the GenAI
	// agent provided by the application.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Math Tutor", "Fiction Writer"
	GenAIAgentNameKey = attribute.Key("gen_ai.agent.name")

	// GenAIConversationIDKey is the attribute Key conforming to the
	// "gen_ai.conversation.id" semantic conventions. It represents the unique
	// identifier for a conversation (session, thread), used to store and correlate
	// messages within this conversation.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "conv_5j66UpCpwteGg4YSxUnt7lPY"
	GenAIConversationIDKey = attribute.Key("gen_ai.conversation.id")

	// GenAIDataSourceIDKey is the attribute Key conforming to the
	// "gen_ai.data_source.id" semantic conventions. It represents the data source
	// identifier.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "H7STPQYOND"
	// Note: Data sources are used by AI agents and RAG applications to store
	// grounding data. A data source may be an external database, object store,
	// document collection, website, or any other storage system used by the GenAI
	// agent or application. The `gen_ai.data_source.id` SHOULD match the identifier
	// used by the GenAI system rather than a name specific to the external storage,
	// such as a database or object store. Semantic conventions referencing
	// `gen_ai.data_source.id` MAY also leverage additional attributes, such as
	// `db.*`, to further identify and describe the data source.
	GenAIDataSourceIDKey = attribute.Key("gen_ai.data_source.id")

	// GenAIInputMessagesKey is the attribute Key conforming to the
	// "gen_ai.input.messages" semantic conventions. It represents the chat history
	// provided to the model as an input.
	//
	// Type: any
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "[\n {\n "role": "user",\n "parts": [\n {\n "type": "text",\n
	// "content": "Weather in Paris?"\n }\n ]\n },\n {\n "role": "assistant",\n
	// "parts": [\n {\n "type": "tool_call",\n "id":
	// "call_VSPygqKTWdrhaFErNvMV18Yl",\n "name": "get_weather",\n "arguments": {\n
	// "location": "Paris"\n }\n }\n ]\n },\n {\n "role": "tool",\n "parts": [\n {\n
	// "type": "tool_call_response",\n "id": " call_VSPygqKTWdrhaFErNvMV18Yl",\n
	// "result": "rainy, 57°F"\n }\n ]\n }\n]\n"
	// Note: Instrumentations MUST follow [Input messages JSON schema].
	// When the attribute is recorded on events, it MUST be recorded in structured
	// form. When recorded on spans, it MAY be recorded as a JSON string if
	// structured
	// format is not supported and SHOULD be recorded in structured form otherwise.
	//
	// Messages MUST be provided in the order they were sent to the model.
	// Instrumentations MAY provide a way for users to filter or truncate
	// input messages.
	//
	// > [!Warning]
	// > This attribute is likely to contain sensitive information including
	// > user/PII data.
	//
	// See [Recording content on attributes]
	// section for more details.
	//
	// [Input messages JSON schema]: /docs/gen-ai/gen-ai-input-messages.json
	// [Recording content on attributes]: /docs/gen-ai/gen-ai-spans.md#recording-content-on-attributes
	GenAIInputMessagesKey = attribute.Key("gen_ai.input.messages")

	// GenAIOperationNameKey is the attribute Key conforming to the
	// "gen_ai.operation.name" semantic conventions. It represents the name of the
	// operation being performed.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: If one of the predefined values applies, but specific system uses a
	// different name it's RECOMMENDED to document it in the semantic conventions
	// for specific GenAI system and use system-specific name in the
	// instrumentation. If a different name is not documented, instrumentation
	// libraries SHOULD use applicable predefined value.
	GenAIOperationNameKey = attribute.Key("gen_ai.operation.name")

	// GenAIOutputMessagesKey is the attribute Key conforming to the
	// "gen_ai.output.messages" semantic conventions. It represents the messages
	// returned by the model where each message represents a specific model response
	// (choice, candidate).
	//
	// Type: any
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "[\n {\n "role": "assistant",\n "parts": [\n {\n "type": "text",\n
	// "content": "The weather in Paris is currently rainy with a temperature of
	// 57°F."\n }\n ],\n "finish_reason": "stop"\n }\n]\n"
	// Note: Instrumentations MUST follow [Output messages JSON schema]
	//
	// Each message represents a single output choice/candidate generated by
	// the model. Each message corresponds to exactly one generation
	// (choice/candidate) and vice versa - one choice cannot be split across
	// multiple messages or one message cannot contain parts from multiple choices.
	//
	// When the attribute is recorded on events, it MUST be recorded in structured
	// form. When recorded on spans, it MAY be recorded as a JSON string if
	// structured
	// format is not supported and SHOULD be recorded in structured form otherwise.
	//
	// Instrumentations MAY provide a way for users to filter or truncate
	// output messages.
	//
	// > [!Warning]
	// > This attribute is likely to contain sensitive information including
	// > user/PII data.
	//
	// See [Recording content on attributes]
	// section for more details.
	//
	// [Output messages JSON schema]: /docs/gen-ai/gen-ai-output-messages.json
	// [Recording content on attributes]: /docs/gen-ai/gen-ai-spans.md#recording-content-on-attributes
	GenAIOutputMessagesKey = attribute.Key("gen_ai.output.messages")

	// GenAIOutputTypeKey is the attribute Key conforming to the
	// "gen_ai.output.type" semantic conventions. It represents the represents the
	// content type requested by the client.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: This attribute SHOULD be used when the client requests output of a
	// specific type. The model may return zero or more outputs of this type.
	// This attribute specifies the output modality and not the actual output
	// format. For example, if an image is requested, the actual output could be a
	// URL pointing to an image file.
	// Additional output format details may be recorded in the future in the
	// `gen_ai.output.{type}.*` attributes.
	GenAIOutputTypeKey = attribute.Key("gen_ai.output.type")

	// GenAIProviderNameKey is the attribute Key conforming to the
	// "gen_ai.provider.name" semantic conventions. It represents the Generative AI
	// provider as identified by the client or server instrumentation.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: The attribute SHOULD be set based on the instrumentation's best
	// knowledge and may differ from the actual model provider.
	//
	// Multiple providers, including Azure OpenAI, Gemini, and AI hosting platforms
	// are accessible using the OpenAI REST API and corresponding client libraries,
	// but may proxy or host models from different providers.
	//
	// The `gen_ai.request.model`, `gen_ai.response.model`, and `server.address`
	// attributes may help identify the actual system in use.
	//
	// The `gen_ai.provider.name` attribute acts as a discriminator that
	// identifies the GenAI telemetry format flavor specific to that provider
	// within GenAI semantic conventions.
	// It SHOULD be set consistently with provider-specific attributes and signals.
	// For example, GenAI spans, metrics, and events related to AWS Bedrock
	// should have the `gen_ai.provider.name` set to `aws.bedrock` and include
	// applicable `aws.bedrock.*` attributes and are not expected to include
	// `openai.*` attributes.
	GenAIProviderNameKey = attribute.Key("gen_ai.provider.name")

	// GenAIRequestChoiceCountKey is the attribute Key conforming to the
	// "gen_ai.request.choice.count" semantic conventions. It represents the target
	// number of candidate completions to return.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 3
	GenAIRequestChoiceCountKey = attribute.Key("gen_ai.request.choice.count")

	// GenAIRequestEncodingFormatsKey is the attribute Key conforming to the
	// "gen_ai.request.encoding_formats" semantic conventions. It represents the
	// encoding formats requested in an embeddings operation, if specified.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "base64"], ["float", "binary"
	// Note: In some GenAI systems the encoding formats are called embedding types.
	// Also, some GenAI systems only accept a single format per request.
	GenAIRequestEncodingFormatsKey = attribute.Key("gen_ai.request.encoding_formats")

	// GenAIRequestFrequencyPenaltyKey is the attribute Key conforming to the
	// "gen_ai.request.frequency_penalty" semantic conventions. It represents the
	// frequency penalty setting for the GenAI request.
	//
	// Type: double
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 0.1
	GenAIRequestFrequencyPenaltyKey = attribute.Key("gen_ai.request.frequency_penalty")

	// GenAIRequestMaxTokensKey is the attribute Key conforming to the
	// "gen_ai.request.max_tokens" semantic conventions. It represents the maximum
	// number of tokens the model generates for a request.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 100
	GenAIRequestMaxTokensKey = attribute.Key("gen_ai.request.max_tokens")

	// GenAIRequestModelKey is the attribute Key conforming to the
	// "gen_ai.request.model" semantic conventions. It represents the name of the
	// GenAI model a request is being made to.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: gpt-4
	GenAIRequestModelKey = attribute.Key("gen_ai.request.model")

	// GenAIRequestPresencePenaltyKey is the attribute Key conforming to the
	// "gen_ai.request.presence_penalty" semantic conventions. It represents the
	// presence penalty setting for the GenAI request.
	//
	// Type: double
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 0.1
	GenAIRequestPresencePenaltyKey = attribute.Key("gen_ai.request.presence_penalty")

	// GenAIRequestSeedKey is the attribute Key conforming to the
	// "gen_ai.request.seed" semantic conventions. It represents the requests with
	// same seed value more likely to return same result.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 100
	GenAIRequestSeedKey = attribute.Key("gen_ai.request.seed")

	// GenAIRequestStopSequencesKey is the attribute Key conforming to the
	// "gen_ai.request.stop_sequences" semantic conventions. It represents the list
	// of sequences that the model will use to stop generating further tokens.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "forest", "lived"
	GenAIRequestStopSequencesKey = attribute.Key("gen_ai.request.stop_sequences")

	// GenAIRequestTemperatureKey is the attribute Key conforming to the
	// "gen_ai.request.temperature" semantic conventions. It represents the
	// temperature setting for the GenAI request.
	//
	// Type: double
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 0.0
	GenAIRequestTemperatureKey = attribute.Key("gen_ai.request.temperature")

	// GenAIRequestTopKKey is the attribute Key conforming to the
	// "gen_ai.request.top_k" semantic conventions. It represents the top_k sampling
	// setting for the GenAI request.
	//
	// Type: double
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1.0
	GenAIRequestTopKKey = attribute.Key("gen_ai.request.top_k")

	// GenAIRequestTopPKey is the attribute Key conforming to the
	// "gen_ai.request.top_p" semantic conventions. It represents the top_p sampling
	// setting for the GenAI request.
	//
	// Type: double
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1.0
	GenAIRequestTopPKey = attribute.Key("gen_ai.request.top_p")

	// GenAIResponseFinishReasonsKey is the attribute Key conforming to the
	// "gen_ai.response.finish_reasons" semantic conventions. It represents the
	// array of reasons the model stopped generating tokens, corresponding to each
	// generation received.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "stop"], ["stop", "length"
	GenAIResponseFinishReasonsKey = attribute.Key("gen_ai.response.finish_reasons")

	// GenAIResponseIDKey is the attribute Key conforming to the
	// "gen_ai.response.id" semantic conventions. It represents the unique
	// identifier for the completion.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "chatcmpl-123"
	GenAIResponseIDKey = attribute.Key("gen_ai.response.id")

	// GenAIResponseModelKey is the attribute Key conforming to the
	// "gen_ai.response.model" semantic conventions. It represents the name of the
	// model that generated the response.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "gpt-4-0613"
	GenAIResponseModelKey = attribute.Key("gen_ai.response.model")

	// GenAISystemInstructionsKey is the attribute Key conforming to the
	// "gen_ai.system_instructions" semantic conventions. It represents the system
	// message or instructions provided to the GenAI model separately from the chat
	// history.
	//
	// Type: any
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "[\n {\n "type": "text",\n "content": "You are an Agent that greet
	// users, always use greetings tool to respond"\n }\n]\n", "[\n {\n "type":
	// "text",\n "content": "You are a language translator."\n },\n {\n "type":
	// "text",\n "content": "Your mission is to translate text in English to
	// French."\n }\n]\n"
	// Note: This attribute SHOULD be used when the corresponding provider or API
	// allows to provide system instructions or messages separately from the
	// chat history.
	//
	// Instructions that are part of the chat history SHOULD be recorded in
	// `gen_ai.input.messages` attribute instead.
	//
	// Instrumentations MUST follow [System instructions JSON schema].
	//
	// When recorded on spans, it MAY be recorded as a JSON string if structured
	// format is not supported and SHOULD be recorded in structured form otherwise.
	//
	// Instrumentations MAY provide a way for users to filter or truncate
	// system instructions.
	//
	// > [!Warning]
	// > This attribute may contain sensitive information.
	//
	// See [Recording content on attributes]
	// section for more details.
	//
	// [System instructions JSON schema]: /docs/gen-ai/gen-ai-system-instructions.json
	// [Recording content on attributes]: /docs/gen-ai/gen-ai-spans.md#recording-content-on-attributes
	GenAISystemInstructionsKey = attribute.Key("gen_ai.system_instructions")

	// GenAITokenTypeKey is the attribute Key conforming to the "gen_ai.token.type"
	// semantic conventions. It represents the type of token being counted.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "input", "output"
	GenAITokenTypeKey = attribute.Key("gen_ai.token.type")

	// GenAIToolCallIDKey is the attribute Key conforming to the
	// "gen_ai.tool.call.id" semantic conventions. It represents the tool call
	// identifier.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "call_mszuSIzqtI65i1wAUOE8w5H4"
	GenAIToolCallIDKey = attribute.Key("gen_ai.tool.call.id")

	// GenAIToolDescriptionKey is the attribute Key conforming to the
	// "gen_ai.tool.description" semantic conventions. It represents the tool
	// description.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Multiply two numbers"
	GenAIToolDescriptionKey = attribute.Key("gen_ai.tool.description")

	// GenAIToolNameKey is the attribute Key conforming to the "gen_ai.tool.name"
	// semantic conventions. It represents the name of the tool utilized by the
	// agent.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Flights"
	GenAIToolNameKey = attribute.Key("gen_ai.tool.name")

	// GenAIToolTypeKey is the attribute Key conforming to the "gen_ai.tool.type"
	// semantic conventions. It represents the type of the tool utilized by the
	// agent.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "function", "extension", "datastore"
	// Note: Extension: A tool executed on the agent-side to directly call external
	// APIs, bridging the gap between the agent and real-world systems.
	// Agent-side operations involve actions that are performed by the agent on the
	// server or within the agent's controlled environment.
	// Function: A tool executed on the client-side, where the agent generates
	// parameters for a predefined function, and the client executes the logic.
	// Client-side operations are actions taken on the user's end or within the
	// client application.
	// Datastore: A tool used by the agent to access and query structured or
	// unstructured external data for retrieval-augmented tasks or knowledge
	// updates.
	GenAIToolTypeKey = attribute.Key("gen_ai.tool.type")

	// GenAIUsageInputTokensKey is the attribute Key conforming to the
	// "gen_ai.usage.input_tokens" semantic conventions. It represents the number of
	// tokens used in the GenAI input (prompt).
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 100
	GenAIUsageInputTokensKey = attribute.Key("gen_ai.usage.input_tokens")

	// GenAIUsageOutputTokensKey is the attribute Key conforming to the
	// "gen_ai.usage.output_tokens" semantic conventions. It represents the number
	// of tokens used in the GenAI response (completion).
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 180
	GenAIUsageOutputTokensKey = attribute.Key("gen_ai.usage.output_tokens")
)

// GenAIAgentDescription returns an attribute KeyValue conforming to the
// "gen_ai.agent.description" semantic conventions. It represents the free-form
// description of the GenAI agent provided by the application.
func GenAIAgentDescription(val string) attribute.KeyValue {
	return GenAIAgentDescriptionKey.String(val)
}

// GenAIAgentID returns an attribute KeyValue conforming to the "gen_ai.agent.id"
// semantic conventions. It represents the unique identifier of the GenAI agent.
func GenAIAgentID(val string) attribute.KeyValue {
	return GenAIAgentIDKey.String(val)
}

// GenAIAgentName returns an attribute KeyValue conforming to the
// "gen_ai.agent.name" semantic conventions. It represents the human-readable
// name of the GenAI agent provided by the application.
func GenAIAgentName(val string) attribute.KeyValue {
	return GenAIAgentNameKey.String(val)
}

// GenAIConversationID returns an attribute KeyValue conforming to the
// "gen_ai.conversation.id" semantic conventions. It represents the unique
// identifier for a conversation (session, thread), used to store and correlate
// messages within this conversation.
func GenAIConversationID(val string) attribute.KeyValue {
	return GenAIConversationIDKey.String(val)
}

// GenAIDataSourceID returns an attribute KeyValue conforming to the
// "gen_ai.data_source.id" semantic conventions. It represents the data source
// identifier.
func GenAIDataSourceID(val string) attribute.KeyValue {
	return GenAIDataSourceIDKey.String(val)
}

// GenAIRequestChoiceCount returns an attribute KeyValue conforming to the
// "gen_ai.request.choice.count" semantic conventions. It represents the target
// number of candidate completions to return.
func GenAIRequestChoiceCount(val int) attribute.KeyValue {
	return GenAIRequestChoiceCountKey.Int(val)
}

// GenAIRequestEncodingFormats returns an attribute KeyValue conforming to the
// "gen_ai.request.encoding_formats" semantic conventions. It represents the
// encoding formats requested in an embeddings operation, if specified.
func GenAIRequestEncodingFormats(val ...string) attribute.KeyValue {
	return GenAIRequestEncodingFormatsKey.StringSlice(val)
}

// GenAIRequestFrequencyPenalty returns an attribute KeyValue conforming to the
// "gen_ai.request.frequency_penalty" semantic conventions. It represents the
// frequency penalty setting for the GenAI request.
func GenAIRequestFrequencyPenalty(val float64) attribute.KeyValue {
	return GenAIRequestFrequencyPenaltyKey.Float64(val)
}

// GenAIRequestMaxTokens returns an attribute KeyValue conforming to the
// "gen_ai.request.max_tokens" semantic conventions. It represents the maximum
// number of tokens the model generates for a request.
func GenAIRequestMaxTokens(val int) attribute.KeyValue {
	return GenAIRequestMaxTokensKey.Int(val)
}

// GenAIRequestModel returns an attribute KeyValue conforming to the
// "gen_ai.request.model" semantic conventions. It represents the name of the
// GenAI model a request is being made to.
func GenAIRequestModel(val string) attribute.KeyValue {
	return GenAIRequestModelKey.String(val)
}

// GenAIRequestPresencePenalty returns an attribute KeyValue conforming to the
// "gen_ai.request.presence_penalty" semantic conventions. It represents the
// presence penalty setting for the GenAI request.
func GenAIRequestPresencePenalty(val float64) attribute.KeyValue {
	return GenAIRequestPresencePenaltyKey.Float64(val)
}

// GenAIRequestSeed returns an attribute KeyValue conforming to the
// "gen_ai.request.seed" semantic conventions. It represents the requests with
// same seed value more likely to return same result.
func GenAIRequestSeed(val int) attribute.KeyValue {
	return GenAIRequestSeedKey.Int(val)
}

// GenAIRequestStopSequences returns an attribute KeyValue conforming to the
// "gen_ai.request.stop_sequences" semantic conventions. It represents the list
// of sequences that the model will use to stop generating further tokens.
func GenAIRequestStopSequences(val ...string) attribute.KeyValue {
	return GenAIRequestStopSequencesKey.StringSlice(val)
}

// GenAIRequestTemperature returns an attribute KeyValue conforming to the
// "gen_ai.request.temperature" semantic conventions. It represents the
// temperature setting for the GenAI request.
func GenAIRequestTemperature(val float64) attribute.KeyValue {
	return GenAIRequestTemperatureKey.Float64(val)
}

// GenAIRequestTopK returns an attribute KeyValue conforming to the
// "gen_ai.request.top_k" semantic conventions. It represents the top_k sampling
// setting for the GenAI request.
func GenAIRequestTopK(val float64) attribute.KeyValue {
	return GenAIRequestTopKKey.Float64(val)
}

// GenAIRequestTopP returns an attribute KeyValue conforming to the
// "gen_ai.request.top_p" semantic conventions. It represents the top_p sampling
// setting for the GenAI request.
func GenAIRequestTopP(val float64) attribute.KeyValue {
	return GenAIRequestTopPKey.Float64(val)
}

// GenAIResponseFinishReasons returns an attribute KeyValue conforming to the
// "gen_ai.response.finish_reasons" semantic conventions. It represents the array
// of reasons the model stopped generating tokens, corresponding to each
// generation received.
func GenAIResponseFinishReasons(val ...string) attribute.KeyValue {
	return GenAIResponseFinishReasonsKey.StringSlice(val)
}

// GenAIResponseID returns an attribute KeyValue conforming to the
// "gen_ai.response.id" semantic conventions. It represents the unique identifier
// for the completion.
func GenAIResponseID(val string) attribute.KeyValue {
	return GenAIResponseIDKey.String(val)
}

// GenAIResponseModel returns an attribute KeyValue conforming to the
// "gen_ai.response.model" semantic conventions. It represents the name of the
// model that generated the response.
func GenAIResponseModel(val string) attribute.KeyValue {
	return GenAIResponseModelKey.String(val)
}

// GenAIToolCallID returns an attribute KeyValue conforming to the
// "gen_ai.tool.call.id" semantic conventions. It represents the tool call
// identifier.
func GenAIToolCallID(val string) attribute.KeyValue {
	return GenAIToolCallIDKey.String(val)
}

// GenAIToolDescription returns an attribute KeyValue conforming to the
// "gen_ai.tool.description" semantic conventions. It represents the tool
// description.
func GenAIToolDescription(val string) attribute.KeyValue {
	return GenAIToolDescriptionKey.String(val)
}

// GenAIToolName returns an attribute KeyValue conforming to the
// "gen_ai.tool.name" semantic conventions. It represents the name of the tool
// utilized by the agent.
func GenAIToolName(val string) attribute.KeyValue {
	return GenAIToolNameKey.String(val)
}

// GenAIToolType returns an attribute KeyValue conforming to the
// "gen_ai.tool.type" semantic conventions. It represents the type of the tool
// utilized by the agent.
func GenAIToolType(val string) attribute.KeyValue {
	return GenAIToolTypeKey.String(val)
}

// GenAIUsageInputTokens returns an attribute KeyValue conforming to the
// "gen_ai.usage.input_tokens" semantic conventions. It represents the number of
// tokens used in the GenAI input (prompt).
func GenAIUsageInputTokens(val int) attribute.KeyValue {
	return GenAIUsageInputTokensKey.Int(val)
}

// GenAIUsageOutputTokens returns an attribute KeyValue conforming to the
// "gen_ai.usage.output_tokens" semantic conventions. It represents the number of
// tokens used in the GenAI response (completion).
func GenAIUsageOutputTokens(val int) attribute.KeyValue {
	return GenAIUsageOutputTokensKey.Int(val)
}

// Enum values for gen_ai.operation.name
var (
	// Chat completion operation such as [OpenAI Chat API]
	// Stability: development
	//
	// [OpenAI Chat API]: https://platform.openai.com/docs/api-reference/chat
	GenAIOperationNameChat = GenAIOperationNameKey.String("chat")
	// Multimodal content generation operation such as [Gemini Generate Content]
	// Stability: development
	//
	// [Gemini Generate Content]: https://ai.google.dev/api/generate-content
	GenAIOperationNameGenerateContent = GenAIOperationNameKey.String("generate_content")
	// Text completions operation such as [OpenAI Completions API (Legacy)]
	// Stability: development
	//
	// [OpenAI Completions API (Legacy)]: https://platform.openai.com/docs/api-reference/completions
	GenAIOperationNameTextCompletion = GenAIOperationNameKey.String("text_completion")
	// Embeddings operation such as [OpenAI Create embeddings API]
	// Stability: development
	//
	// [OpenAI Create embeddings API]: https://platform.openai.com/docs/api-reference/embeddings/create
	GenAIOperationNameEmbeddings = GenAIOperationNameKey.String("embeddings")
	// Create GenAI agent
	// Stability: development
	GenAIOperationNameCreateAgent = GenAIOperationNameKey.String("create_agent")
	// Invoke GenAI agent
	// Stability: development
	GenAIOperationNameInvokeAgent = GenAIOperationNameKey.String("invoke_agent")
	// Execute a tool
	// Stability: development
	GenAIOperationNameExecuteTool = GenAIOperationNameKey.String("execute_tool")
)

// Enum values for gen_ai.output.type
var (
	// Plain text
	// Stability: development
	GenAIOutputTypeText = GenAIOutputTypeKey.String("text")
	// JSON object with known or unknown schema
	// Stability: development
	GenAIOutputTypeJSON = GenAIOutputTypeKey.String("json")
	// Image
	// Stability: development
	GenAIOutputTypeImage = GenAIOutputTypeKey.String("image")
	// Speech
	// Stability: development
	GenAIOutputTypeSpeech = GenAIOutputTypeKey.String("speech")
)

// Enum values for gen_ai.provider.name
var (
	// [OpenAI]
	// Stability: development
	//
	// [OpenAI]: https://openai.com/
	GenAIProviderNameOpenAI = GenAIProviderNameKey.String("openai")
	// Any Google generative AI endpoint
	// Stability: development
	GenAIProviderNameGCPGenAI = GenAIProviderNameKey.String("gcp.gen_ai")
	// [Vertex AI]
	// Stability: development
	//
	// [Vertex AI]: https://cloud.google.com/vertex-ai
	GenAIProviderNameGCPVertexAI = GenAIProviderNameKey.String("gcp.vertex_ai")
	// [Gemini]
	// Stability: development
	//
	// [Gemini]: https://cloud.google.com/products/gemini
	GenAIProviderNameGCPGemini = GenAIProviderNameKey.String("gcp.gemini")
	// [Anthropic]
	// Stability: development
	//
	// [Anthropic]: https://www.anthropic.com/
	GenAIProviderNameAnthropic = GenAIProviderNameKey.String("anthropic")
	// [Cohere]
	// Stability: development
	//
	// [Cohere]: https://cohere.com/
	GenAIProviderNameCohere = GenAIProviderNameKey.String("cohere")
	// Azure AI Inference
	// Stability: development
	GenAIProviderNameAzureAIInference = GenAIProviderNameKey.String("azure.ai.inference")
	// [Azure OpenAI]
	// Stability: development
	//
	// [Azure OpenAI]: https://azure.microsoft.com/products/ai-services/openai-service/
	GenAIProviderNameAzureAIOpenAI = GenAIProviderNameKey.String("azure.ai.openai")
	// [IBM Watsonx AI]
	// Stability: development
	//
	// [IBM Watsonx AI]: https://www.ibm.com/products/watsonx-ai
	GenAIProviderNameIBMWatsonxAI = GenAIProviderNameKey.String("ibm.watsonx.ai")
	// [AWS Bedrock]
	// Stability: development
	//
	// [AWS Bedrock]: https://aws.amazon.com/bedrock
	GenAIProviderNameAWSBedrock = GenAIProviderNameKey.String("aws.bedrock")
	// [Perplexity]
	// Stability: development
	//
	// [Perplexity]: https://www.perplexity.ai/
	GenAIProviderNamePerplexity = GenAIProviderNameKey.String("perplexity")
	// [xAI]
	// Stability: development
	//
	// [xAI]: https://x.ai/
	GenAIProviderNameXAI = GenAIProviderNameKey.String("x_ai")
	// [DeepSeek]
	// Stability: development
	//
	// [DeepSeek]: https://www.deepseek.com/
	GenAIProviderNameDeepseek = GenAIProviderNameKey.String("deepseek")
	// [Groq]
	// Stability: development
	//
	// [Groq]: https://groq.com/
	GenAIProviderNameGroq = GenAIProviderNameKey.String("groq")
	// [Mistral AI]
	// Stability: development
	//
	// [Mistral AI]: https://mistral.ai/
	GenAIProviderNameMistralAI = GenAIProviderNameKey.String("mistral_ai")
)

// Enum values for gen_ai.token.type
var (
	// Input tokens (prompt, input, etc.)
	// Stability: development
	GenAITokenTypeInput = GenAITokenTypeKey.String("input")
	// Output tokens (completion, response, etc.)
	// Stability: development
	GenAITokenTypeOutput = GenAITokenTypeKey.String("output")
)

// Namespace: geo
const (
	// GeoContinentCodeKey is the attribute Key conforming to the
	// "geo.continent.code" semantic conventions. It represents the two-letter code
	// representing continent’s name.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	GeoContinentCodeKey = attribute.Key("geo.continent.code")

	// GeoCountryISOCodeKey is the attribute Key conforming to the
	// "geo.country.iso_code" semantic conventions. It represents the two-letter ISO
	// Country Code ([ISO 3166-1 alpha2]).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "CA"
	//
	// [ISO 3166-1 alpha2]: https://wikipedia.org/wiki/ISO_3166-1#Codes
	GeoCountryISOCodeKey = attribute.Key("geo.country.iso_code")

	// GeoLocalityNameKey is the attribute Key conforming to the "geo.locality.name"
	// semantic conventions. It represents the locality name. Represents the name of
	// a city, town, village, or similar populated place.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Montreal", "Berlin"
	GeoLocalityNameKey = attribute.Key("geo.locality.name")

	// GeoLocationLatKey is the attribute Key conforming to the "geo.location.lat"
	// semantic conventions. It represents the latitude of the geo location in
	// [WGS84].
	//
	// Type: double
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 45.505918
	//
	// [WGS84]: https://wikipedia.org/wiki/World_Geodetic_System#WGS84
	GeoLocationLatKey = attribute.Key("geo.location.lat")

	// GeoLocationLonKey is the attribute Key conforming to the "geo.location.lon"
	// semantic conventions. It represents the longitude of the geo location in
	// [WGS84].
	//
	// Type: double
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: -73.61483
	//
	// [WGS84]: https://wikipedia.org/wiki/World_Geodetic_System#WGS84
	GeoLocationLonKey = attribute.Key("geo.location.lon")

	// GeoPostalCodeKey is the attribute Key conforming to the "geo.postal_code"
	// semantic conventions. It represents the postal code associated with the
	// location. Values appropriate for this field may also be known as a postcode
	// or ZIP code and will vary widely from country to country.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "94040"
	GeoPostalCodeKey = attribute.Key("geo.postal_code")

	// GeoRegionISOCodeKey is the attribute Key conforming to the
	// "geo.region.iso_code" semantic conventions. It represents the region ISO code
	// ([ISO 3166-2]).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "CA-QC"
	//
	// [ISO 3166-2]: https://wikipedia.org/wiki/ISO_3166-2
	GeoRegionISOCodeKey = attribute.Key("geo.region.iso_code")
)

// GeoCountryISOCode returns an attribute KeyValue conforming to the
// "geo.country.iso_code" semantic conventions. It represents the two-letter ISO
// Country Code ([ISO 3166-1 alpha2]).
//
// [ISO 3166-1 alpha2]: https://wikipedia.org/wiki/ISO_3166-1#Codes
func GeoCountryISOCode(val string) attribute.KeyValue {
	return GeoCountryISOCodeKey.String(val)
}

// GeoLocalityName returns an attribute KeyValue conforming to the
// "geo.locality.name" semantic conventions. It represents the locality name.
// Represents the name of a city, town, village, or similar populated place.
func GeoLocalityName(val string) attribute.KeyValue {
	return GeoLocalityNameKey.String(val)
}

// GeoLocationLat returns an attribute KeyValue conforming to the
// "geo.location.lat" semantic conventions. It represents the latitude of the geo
// location in [WGS84].
//
// [WGS84]: https://wikipedia.org/wiki/World_Geodetic_System#WGS84
func GeoLocationLat(val float64) attribute.KeyValue {
	return GeoLocationLatKey.Float64(val)
}

// GeoLocationLon returns an attribute KeyValue conforming to the
// "geo.location.lon" semantic conventions. It represents the longitude of the
// geo location in [WGS84].
//
// [WGS84]: https://wikipedia.org/wiki/World_Geodetic_System#WGS84
func GeoLocationLon(val float64) attribute.KeyValue {
	return GeoLocationLonKey.Float64(val)
}

// GeoPostalCode returns an attribute KeyValue conforming to the
// "geo.postal_code" semantic conventions. It represents the postal code
// associated with the location. Values appropriate for this field may also be
// known as a postcode or ZIP code and will vary widely from country to country.
func GeoPostalCode(val string) attribute.KeyValue {
	return GeoPostalCodeKey.String(val)
}

// GeoRegionISOCode returns an attribute KeyValue conforming to the
// "geo.region.iso_code" semantic conventions. It represents the region ISO code
// ([ISO 3166-2]).
//
// [ISO 3166-2]: https://wikipedia.org/wiki/ISO_3166-2
func GeoRegionISOCode(val string) attribute.KeyValue {
	return GeoRegionISOCodeKey.String(val)
}

// Enum values for geo.continent.code
var (
	// Africa
	// Stability: development
	GeoContinentCodeAf = GeoContinentCodeKey.String("AF")
	// Antarctica
	// Stability: development
	GeoContinentCodeAn = GeoContinentCodeKey.String("AN")
	// Asia
	// Stability: development
	GeoContinentCodeAs = GeoContinentCodeKey.String("AS")
	// Europe
	// Stability: development
	GeoContinentCodeEu = GeoContinentCodeKey.String("EU")
	// North America
	// Stability: development
	GeoContinentCodeNa = GeoContinentCodeKey.String("NA")
	// Oceania
	// Stability: development
	GeoContinentCodeOc = GeoContinentCodeKey.String("OC")
	// South America
	// Stability: development
	GeoContinentCodeSa = GeoContinentCodeKey.String("SA")
)

// Namespace: go
const (
	// GoMemoryTypeKey is the attribute Key conforming to the "go.memory.type"
	// semantic conventions. It represents the type of memory.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "other", "stack"
	GoMemoryTypeKey = attribute.Key("go.memory.type")
)

// Enum values for go.memory.type
var (
	// Memory allocated from the heap that is reserved for stack space, whether or
	// not it is currently in-use.
	// Stability: development
	GoMemoryTypeStack = GoMemoryTypeKey.String("stack")
	// Memory used by the Go runtime, excluding other categories of memory usage
	// described in this enumeration.
	// Stability: development
	GoMemoryTypeOther = GoMemoryTypeKey.String("other")
)

// Namespace: graphql
const (
	// GraphQLDocumentKey is the attribute Key conforming to the "graphql.document"
	// semantic conventions. It represents the GraphQL document being executed.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: query findBookById { bookById(id: ?) { name } }
	// Note: The value may be sanitized to exclude sensitive information.
	GraphQLDocumentKey = attribute.Key("graphql.document")

	// GraphQLOperationNameKey is the attribute Key conforming to the
	// "graphql.operation.name" semantic conventions. It represents the name of the
	// operation being executed.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: findBookById
	GraphQLOperationNameKey = attribute.Key("graphql.operation.name")

	// GraphQLOperationTypeKey is the attribute Key conforming to the
	// "graphql.operation.type" semantic conventions. It represents the type of the
	// operation being executed.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "query", "mutation", "subscription"
	GraphQLOperationTypeKey = attribute.Key("graphql.operation.type")
)

// GraphQLDocument returns an attribute KeyValue conforming to the
// "graphql.document" semantic conventions. It represents the GraphQL document
// being executed.
func GraphQLDocument(val string) attribute.KeyValue {
	return GraphQLDocumentKey.String(val)
}

// GraphQLOperationName returns an attribute KeyValue conforming to the
// "graphql.operation.name" semantic conventions. It represents the name of the
// operation being executed.
func GraphQLOperationName(val string) attribute.KeyValue {
	return GraphQLOperationNameKey.String(val)
}

// Enum values for graphql.operation.type
var (
	// GraphQL query
	// Stability: development
	GraphQLOperationTypeQuery = GraphQLOperationTypeKey.String("query")
	// GraphQL mutation
	// Stability: development
	GraphQLOperationTypeMutation = GraphQLOperationTypeKey.String("mutation")
	// GraphQL subscription
	// Stability: development
	GraphQLOperationTypeSubscription = GraphQLOperationTypeKey.String("subscription")
)

// Namespace: heroku
const (
	// HerokuAppIDKey is the attribute Key conforming to the "heroku.app.id"
	// semantic conventions. It represents the unique identifier for the
	// application.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2daa2797-e42b-4624-9322-ec3f968df4da"
	HerokuAppIDKey = attribute.Key("heroku.app.id")

	// HerokuReleaseCommitKey is the attribute Key conforming to the
	// "heroku.release.commit" semantic conventions. It represents the commit hash
	// for the current release.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "e6134959463efd8966b20e75b913cafe3f5ec"
	HerokuReleaseCommitKey = attribute.Key("heroku.release.commit")

	// HerokuReleaseCreationTimestampKey is the attribute Key conforming to the
	// "heroku.release.creation_timestamp" semantic conventions. It represents the
	// time and date the release was created.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2022-10-23T18:00:42Z"
	HerokuReleaseCreationTimestampKey = attribute.Key("heroku.release.creation_timestamp")
)

// HerokuAppID returns an attribute KeyValue conforming to the "heroku.app.id"
// semantic conventions. It represents the unique identifier for the application.
func HerokuAppID(val string) attribute.KeyValue {
	return HerokuAppIDKey.String(val)
}

// HerokuReleaseCommit returns an attribute KeyValue conforming to the
// "heroku.release.commit" semantic conventions. It represents the commit hash
// for the current release.
func HerokuReleaseCommit(val string) attribute.KeyValue {
	return HerokuReleaseCommitKey.String(val)
}

// HerokuReleaseCreationTimestamp returns an attribute KeyValue conforming to the
// "heroku.release.creation_timestamp" semantic conventions. It represents the
// time and date the release was created.
func HerokuReleaseCreationTimestamp(val string) attribute.KeyValue {
	return HerokuReleaseCreationTimestampKey.String(val)
}

// Namespace: host
const (
	// HostArchKey is the attribute Key conforming to the "host.arch" semantic
	// conventions. It represents the CPU architecture the host system is running
	// on.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	HostArchKey = attribute.Key("host.arch")

	// HostCPUCacheL2SizeKey is the attribute Key conforming to the
	// "host.cpu.cache.l2.size" semantic conventions. It represents the amount of
	// level 2 memory cache available to the processor (in Bytes).
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 12288000
	HostCPUCacheL2SizeKey = attribute.Key("host.cpu.cache.l2.size")

	// HostCPUFamilyKey is the attribute Key conforming to the "host.cpu.family"
	// semantic conventions. It represents the family or generation of the CPU.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "6", "PA-RISC 1.1e"
	HostCPUFamilyKey = attribute.Key("host.cpu.family")

	// HostCPUModelIDKey is the attribute Key conforming to the "host.cpu.model.id"
	// semantic conventions. It represents the model identifier. It provides more
	// granular information about the CPU, distinguishing it from other CPUs within
	// the same family.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "6", "9000/778/B180L"
	HostCPUModelIDKey = attribute.Key("host.cpu.model.id")

	// HostCPUModelNameKey is the attribute Key conforming to the
	// "host.cpu.model.name" semantic conventions. It represents the model
	// designation of the processor.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz"
	HostCPUModelNameKey = attribute.Key("host.cpu.model.name")

	// HostCPUSteppingKey is the attribute Key conforming to the "host.cpu.stepping"
	// semantic conventions. It represents the stepping or core revisions.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "1", "r1p1"
	HostCPUSteppingKey = attribute.Key("host.cpu.stepping")

	// HostCPUVendorIDKey is the attribute Key conforming to the
	// "host.cpu.vendor.id" semantic conventions. It represents the processor
	// manufacturer identifier. A maximum 12-character string.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "GenuineIntel"
	// Note: [CPUID] command returns the vendor ID string in EBX, EDX and ECX
	// registers. Writing these to memory in this order results in a 12-character
	// string.
	//
	// [CPUID]: https://wiki.osdev.org/CPUID
	HostCPUVendorIDKey = attribute.Key("host.cpu.vendor.id")

	// HostIDKey is the attribute Key conforming to the "host.id" semantic
	// conventions. It represents the unique host ID. For Cloud, this must be the
	// instance_id assigned by the cloud provider. For non-containerized systems,
	// this should be the `machine-id`. See the table below for the sources to use
	// to determine the `machine-id` based on operating system.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "fdbf79e8af94cb7f9e8df36789187052"
	HostIDKey = attribute.Key("host.id")

	// HostImageIDKey is the attribute Key conforming to the "host.image.id"
	// semantic conventions. It represents the VM image ID or host OS image ID. For
	// Cloud, this value is from the provider.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "ami-07b06b442921831e5"
	HostImageIDKey = attribute.Key("host.image.id")

	// HostImageNameKey is the attribute Key conforming to the "host.image.name"
	// semantic conventions. It represents the name of the VM image or OS install
	// the host was instantiated from.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "infra-ami-eks-worker-node-7d4ec78312", "CentOS-8-x86_64-1905"
	HostImageNameKey = attribute.Key("host.image.name")

	// HostImageVersionKey is the attribute Key conforming to the
	// "host.image.version" semantic conventions. It represents the version string
	// of the VM image or host OS as defined in [Version Attributes].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "0.1"
	//
	// [Version Attributes]: /docs/resource/README.md#version-attributes
	HostImageVersionKey = attribute.Key("host.image.version")

	// HostIPKey is the attribute Key conforming to the "host.ip" semantic
	// conventions. It represents the available IP addresses of the host, excluding
	// loopback interfaces.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "192.168.1.140", "fe80::abc2:4a28:737a:609e"
	// Note: IPv4 Addresses MUST be specified in dotted-quad notation. IPv6
	// addresses MUST be specified in the [RFC 5952] format.
	//
	// [RFC 5952]: https://www.rfc-editor.org/rfc/rfc5952.html
	HostIPKey = attribute.Key("host.ip")

	// HostMacKey is the attribute Key conforming to the "host.mac" semantic
	// conventions. It represents the available MAC addresses of the host, excluding
	// loopback interfaces.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "AC-DE-48-23-45-67", "AC-DE-48-23-45-67-01-9F"
	// Note: MAC Addresses MUST be represented in [IEEE RA hexadecimal form]: as
	// hyphen-separated octets in uppercase hexadecimal form from most to least
	// significant.
	//
	// [IEEE RA hexadecimal form]: https://standards.ieee.org/wp-content/uploads/import/documents/tutorials/eui.pdf
	HostMacKey = attribute.Key("host.mac")

	// HostNameKey is the attribute Key conforming to the "host.name" semantic
	// conventions. It represents the name of the host. On Unix systems, it may
	// contain what the hostname command returns, or the fully qualified hostname,
	// or another name specified by the user.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "opentelemetry-test"
	HostNameKey = attribute.Key("host.name")

	// HostTypeKey is the attribute Key conforming to the "host.type" semantic
	// conventions. It represents the type of host. For Cloud, this must be the
	// machine type.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "n1-standard-1"
	HostTypeKey = attribute.Key("host.type")
)

// HostCPUCacheL2Size returns an attribute KeyValue conforming to the
// "host.cpu.cache.l2.size" semantic conventions. It represents the amount of
// level 2 memory cache available to the processor (in Bytes).
func HostCPUCacheL2Size(val int) attribute.KeyValue {
	return HostCPUCacheL2SizeKey.Int(val)
}

// HostCPUFamily returns an attribute KeyValue conforming to the
// "host.cpu.family" semantic conventions. It represents the family or generation
// of the CPU.
func HostCPUFamily(val string) attribute.KeyValue {
	return HostCPUFamilyKey.String(val)
}

// HostCPUModelID returns an attribute KeyValue conforming to the
// "host.cpu.model.id" semantic conventions. It represents the model identifier.
// It provides more granular information about the CPU, distinguishing it from
// other CPUs within the same family.
func HostCPUModelID(val string) attribute.KeyValue {
	return HostCPUModelIDKey.String(val)
}

// HostCPUModelName returns an attribute KeyValue conforming to the
// "host.cpu.model.name" semantic conventions. It represents the model
// designation of the processor.
func HostCPUModelName(val string) attribute.KeyValue {
	return HostCPUModelNameKey.String(val)
}

// HostCPUStepping returns an attribute KeyValue conforming to the
// "host.cpu.stepping" semantic conventions. It represents the stepping or core
// revisions.
func HostCPUStepping(val string) attribute.KeyValue {
	return HostCPUSteppingKey.String(val)
}

// HostCPUVendorID returns an attribute KeyValue conforming to the
// "host.cpu.vendor.id" semantic conventions. It represents the processor
// manufacturer identifier. A maximum 12-character string.
func HostCPUVendorID(val string) attribute.KeyValue {
	return HostCPUVendorIDKey.String(val)
}

// HostID returns an attribute KeyValue conforming to the "host.id" semantic
// conventions. It represents the unique host ID. For Cloud, this must be the
// instance_id assigned by the cloud provider. For non-containerized systems,
// this should be the `machine-id`. See the table below for the sources to use to
// determine the `machine-id` based on operating system.
func HostID(val string) attribute.KeyValue {
	return HostIDKey.String(val)
}

// HostImageID returns an attribute KeyValue conforming to the "host.image.id"
// semantic conventions. It represents the VM image ID or host OS image ID. For
// Cloud, this value is from the provider.
func HostImageID(val string) attribute.KeyValue {
	return HostImageIDKey.String(val)
}

// HostImageName returns an attribute KeyValue conforming to the
// "host.image.name" semantic conventions. It represents the name of the VM image
// or OS install the host was instantiated from.
func HostImageName(val string) attribute.KeyValue {
	return HostImageNameKey.String(val)
}

// HostImageVersion returns an attribute KeyValue conforming to the
// "host.image.version" semantic conventions. It represents the version string of
// the VM image or host OS as defined in [Version Attributes].
//
// [Version Attributes]: /docs/resource/README.md#version-attributes
func HostImageVersion(val string) attribute.KeyValue {
	return HostImageVersionKey.String(val)
}

// HostIP returns an attribute KeyValue conforming to the "host.ip" semantic
// conventions. It represents the available IP addresses of the host, excluding
// loopback interfaces.
func HostIP(val ...string) attribute.KeyValue {
	return HostIPKey.StringSlice(val)
}

// HostMac returns an attribute KeyValue conforming to the "host.mac" semantic
// conventions. It represents the available MAC addresses of the host, excluding
// loopback interfaces.
func HostMac(val ...string) attribute.KeyValue {
	return HostMacKey.StringSlice(val)
}

// HostName returns an attribute KeyValue conforming to the "host.name" semantic
// conventions. It represents the name of the host. On Unix systems, it may
// contain what the hostname command returns, or the fully qualified hostname, or
// another name specified by the user.
func HostName(val string) attribute.KeyValue {
	return HostNameKey.String(val)
}

// HostType returns an attribute KeyValue conforming to the "host.type" semantic
// conventions. It represents the type of host. For Cloud, this must be the
// machine type.
func HostType(val string) attribute.KeyValue {
	return HostTypeKey.String(val)
}

// Enum values for host.arch
var (
	// AMD64
	// Stability: development
	HostArchAMD64 = HostArchKey.String("amd64")
	// ARM32
	// Stability: development
	HostArchARM32 = HostArchKey.String("arm32")
	// ARM64
	// Stability: development
	HostArchARM64 = HostArchKey.String("arm64")
	// Itanium
	// Stability: development
	HostArchIA64 = HostArchKey.String("ia64")
	// 32-bit PowerPC
	// Stability: development
	HostArchPPC32 = HostArchKey.String("ppc32")
	// 64-bit PowerPC
	// Stability: development
	HostArchPPC64 = HostArchKey.String("ppc64")
	// IBM z/Architecture
	// Stability: development
	HostArchS390x = HostArchKey.String("s390x")
	// 32-bit x86
	// Stability: development
	HostArchX86 = HostArchKey.String("x86")
)

// Namespace: http
const (
	// HTTPConnectionStateKey is the attribute Key conforming to the
	// "http.connection.state" semantic conventions. It represents the state of the
	// HTTP connection in the HTTP connection pool.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "active", "idle"
	HTTPConnectionStateKey = attribute.Key("http.connection.state")

	// HTTPRequestBodySizeKey is the attribute Key conforming to the
	// "http.request.body.size" semantic conventions. It represents the size of the
	// request payload body in bytes. This is the number of bytes transferred
	// excluding headers and is often, but not always, present as the
	// [Content-Length] header. For requests using transport encoding, this should
	// be the compressed size.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// [Content-Length]: https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length
	HTTPRequestBodySizeKey = attribute.Key("http.request.body.size")

	// HTTPRequestMethodKey is the attribute Key conforming to the
	// "http.request.method" semantic conventions. It represents the HTTP request
	// method.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "GET", "POST", "HEAD"
	// Note: HTTP request method value SHOULD be "known" to the instrumentation.
	// By default, this convention defines "known" methods as the ones listed in
	// [RFC9110]
	// and the PATCH method defined in [RFC5789].
	//
	// If the HTTP request method is not known to instrumentation, it MUST set the
	// `http.request.method` attribute to `_OTHER`.
	//
	// If the HTTP instrumentation could end up converting valid HTTP request
	// methods to `_OTHER`, then it MUST provide a way to override
	// the list of known HTTP methods. If this override is done via environment
	// variable, then the environment variable MUST be named
	// OTEL_INSTRUMENTATION_HTTP_KNOWN_METHODS and support a comma-separated list of
	// case-sensitive known HTTP methods
	// (this list MUST be a full override of the default known method, it is not a
	// list of known methods in addition to the defaults).
	//
	// HTTP method names are case-sensitive and `http.request.method` attribute
	// value MUST match a known HTTP method name exactly.
	// Instrumentations for specific web frameworks that consider HTTP methods to be
	// case insensitive, SHOULD populate a canonical equivalent.
	// Tracing instrumentations that do so, MUST also set
	// `http.request.method_original` to the original value.
	//
	// [RFC9110]: https://www.rfc-editor.org/rfc/rfc9110.html#name-methods
	// [RFC5789]: https://www.rfc-editor.org/rfc/rfc5789.html
	HTTPRequestMethodKey = attribute.Key("http.request.method")

	// HTTPRequestMethodOriginalKey is the attribute Key conforming to the
	// "http.request.method_original" semantic conventions. It represents the
	// original HTTP method sent by the client in the request line.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "GeT", "ACL", "foo"
	HTTPRequestMethodOriginalKey = attribute.Key("http.request.method_original")

	// HTTPRequestResendCountKey is the attribute Key conforming to the
	// "http.request.resend_count" semantic conventions. It represents the ordinal
	// number of request resending attempt (for any reason, including redirects).
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Note: The resend count SHOULD be updated each time an HTTP request gets
	// resent by the client, regardless of what was the cause of the resending (e.g.
	// redirection, authorization failure, 503 Server Unavailable, network issues,
	// or any other).
	HTTPRequestResendCountKey = attribute.Key("http.request.resend_count")

	// HTTPRequestSizeKey is the attribute Key conforming to the "http.request.size"
	// semantic conventions. It represents the total size of the request in bytes.
	// This should be the total number of bytes sent over the wire, including the
	// request line (HTTP/1.1), framing (HTTP/2 and HTTP/3), headers, and request
	// body if any.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	HTTPRequestSizeKey = attribute.Key("http.request.size")

	// HTTPResponseBodySizeKey is the attribute Key conforming to the
	// "http.response.body.size" semantic conventions. It represents the size of the
	// response payload body in bytes. This is the number of bytes transferred
	// excluding headers and is often, but not always, present as the
	// [Content-Length] header. For requests using transport encoding, this should
	// be the compressed size.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// [Content-Length]: https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length
	HTTPResponseBodySizeKey = attribute.Key("http.response.body.size")

	// HTTPResponseSizeKey is the attribute Key conforming to the
	// "http.response.size" semantic conventions. It represents the total size of
	// the response in bytes. This should be the total number of bytes sent over the
	// wire, including the status line (HTTP/1.1), framing (HTTP/2 and HTTP/3),
	// headers, and response body and trailers if any.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	HTTPResponseSizeKey = attribute.Key("http.response.size")

	// HTTPResponseStatusCodeKey is the attribute Key conforming to the
	// "http.response.status_code" semantic conventions. It represents the
	// [HTTP response status code].
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: 200
	//
	// [HTTP response status code]: https://tools.ietf.org/html/rfc7231#section-6
	HTTPResponseStatusCodeKey = attribute.Key("http.response.status_code")

	// HTTPRouteKey is the attribute Key conforming to the "http.route" semantic
	// conventions. It represents the matched route, that is, the path template in
	// the format used by the respective server framework.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "/users/:userID?", "{controller}/{action}/{id?}"
	// Note: MUST NOT be populated when this is not supported by the HTTP server
	// framework as the route attribute should have low-cardinality and the URI path
	// can NOT substitute it.
	// SHOULD include the [application root] if there is one.
	//
	// [application root]: /docs/http/http-spans.md#http-server-definitions
	HTTPRouteKey = attribute.Key("http.route")
)

// HTTPRequestBodySize returns an attribute KeyValue conforming to the
// "http.request.body.size" semantic conventions. It represents the size of the
// request payload body in bytes. This is the number of bytes transferred
// excluding headers and is often, but not always, present as the
// [Content-Length] header. For requests using transport encoding, this should be
// the compressed size.
//
// [Content-Length]: https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length
func HTTPRequestBodySize(val int) attribute.KeyValue {
	return HTTPRequestBodySizeKey.Int(val)
}

// HTTPRequestHeader returns an attribute KeyValue conforming to the
// "http.request.header" semantic conventions. It represents the HTTP request
// headers, `<key>` being the normalized HTTP Header name (lowercase), the value
// being the header values.
func HTTPRequestHeader(key string, val ...string) attribute.KeyValue {
	return attribute.StringSlice("http.request.header."+key, val)
}

// HTTPRequestMethodOriginal returns an attribute KeyValue conforming to the
// "http.request.method_original" semantic conventions. It represents the
// original HTTP method sent by the client in the request line.
func HTTPRequestMethodOriginal(val string) attribute.KeyValue {
	return HTTPRequestMethodOriginalKey.String(val)
}

// HTTPRequestResendCount returns an attribute KeyValue conforming to the
// "http.request.resend_count" semantic conventions. It represents the ordinal
// number of request resending attempt (for any reason, including redirects).
func HTTPRequestResendCount(val int) attribute.KeyValue {
	return HTTPRequestResendCountKey.Int(val)
}

// HTTPRequestSize returns an attribute KeyValue conforming to the
// "http.request.size" semantic conventions. It represents the total size of the
// request in bytes. This should be the total number of bytes sent over the wire,
// including the request line (HTTP/1.1), framing (HTTP/2 and HTTP/3), headers,
// and request body if any.
func HTTPRequestSize(val int) attribute.KeyValue {
	return HTTPRequestSizeKey.Int(val)
}

// HTTPResponseBodySize returns an attribute KeyValue conforming to the
// "http.response.body.size" semantic conventions. It represents the size of the
// response payload body in bytes. This is the number of bytes transferred
// excluding headers and is often, but not always, present as the
// [Content-Length] header. For requests using transport encoding, this should be
// the compressed size.
//
// [Content-Length]: https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length
func HTTPResponseBodySize(val int) attribute.KeyValue {
	return HTTPResponseBodySizeKey.Int(val)
}

// HTTPResponseHeader returns an attribute KeyValue conforming to the
// "http.response.header" semantic conventions. It represents the HTTP response
// headers, `<key>` being the normalized HTTP Header name (lowercase), the value
// being the header values.
func HTTPResponseHeader(key string, val ...string) attribute.KeyValue {
	return attribute.StringSlice("http.response.header."+key, val)
}

// HTTPResponseSize returns an attribute KeyValue conforming to the
// "http.response.size" semantic conventions. It represents the total size of the
// response in bytes. This should be the total number of bytes sent over the
// wire, including the status line (HTTP/1.1), framing (HTTP/2 and HTTP/3),
// headers, and response body and trailers if any.
func HTTPResponseSize(val int) attribute.KeyValue {
	return HTTPResponseSizeKey.Int(val)
}

// HTTPResponseStatusCode returns an attribute KeyValue conforming to the
// "http.response.status_code" semantic conventions. It represents the
// [HTTP response status code].
//
// [HTTP response status code]: https://tools.ietf.org/html/rfc7231#section-6
func HTTPResponseStatusCode(val int) attribute.KeyValue {
	return HTTPResponseStatusCodeKey.Int(val)
}

// HTTPRoute returns an attribute KeyValue conforming to the "http.route"
// semantic conventions. It represents the matched route, that is, the path
// template in the format used by the respective server framework.
func HTTPRoute(val string) attribute.KeyValue {
	return HTTPRouteKey.String(val)
}

// Enum values for http.connection.state
var (
	// active state.
	// Stability: development
	HTTPConnectionStateActive = HTTPConnectionStateKey.String("active")
	// idle state.
	// Stability: development
	HTTPConnectionStateIdle = HTTPConnectionStateKey.String("idle")
)

// Enum values for http.request.method
var (
	// CONNECT method.
	// Stability: stable
	HTTPRequestMethodConnect = HTTPRequestMethodKey.String("CONNECT")
	// DELETE method.
	// Stability: stable
	HTTPRequestMethodDelete = HTTPRequestMethodKey.String("DELETE")
	// GET method.
	// Stability: stable
	HTTPRequestMethodGet = HTTPRequestMethodKey.String("GET")
	// HEAD method.
	// Stability: stable
	HTTPRequestMethodHead = HTTPRequestMethodKey.String("HEAD")
	// OPTIONS method.
	// Stability: stable
	HTTPRequestMethodOptions = HTTPRequestMethodKey.String("OPTIONS")
	// PATCH method.
	// Stability: stable
	HTTPRequestMethodPatch = HTTPRequestMethodKey.String("PATCH")
	// POST method.
	// Stability: stable
	HTTPRequestMethodPost = HTTPRequestMethodKey.String("POST")
	// PUT method.
	// Stability: stable
	HTTPRequestMethodPut = HTTPRequestMethodKey.String("PUT")
	// TRACE method.
	// Stability: stable
	HTTPRequestMethodTrace = HTTPRequestMethodKey.String("TRACE")
	// Any HTTP method that the instrumentation has no prior knowledge of.
	// Stability: stable
	HTTPRequestMethodOther = HTTPRequestMethodKey.String("_OTHER")
)

// Namespace: hw
const (
	// HwBatteryCapacityKey is the attribute Key conforming to the
	// "hw.battery.capacity" semantic conventions. It represents the design capacity
	// in Watts-hours or Amper-hours.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "9.3Ah", "50Wh"
	HwBatteryCapacityKey = attribute.Key("hw.battery.capacity")

	// HwBatteryChemistryKey is the attribute Key conforming to the
	// "hw.battery.chemistry" semantic conventions. It represents the battery
	// [chemistry], e.g. Lithium-Ion, Nickel-Cadmium, etc.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Li-ion", "NiMH"
	//
	// [chemistry]: https://schemas.dmtf.org/wbem/cim-html/2.31.0/CIM_Battery.html
	HwBatteryChemistryKey = attribute.Key("hw.battery.chemistry")

	// HwBatteryStateKey is the attribute Key conforming to the "hw.battery.state"
	// semantic conventions. It represents the current state of the battery.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	HwBatteryStateKey = attribute.Key("hw.battery.state")

	// HwBiosVersionKey is the attribute Key conforming to the "hw.bios_version"
	// semantic conventions. It represents the BIOS version of the hardware
	// component.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "1.2.3"
	HwBiosVersionKey = attribute.Key("hw.bios_version")

	// HwDriverVersionKey is the attribute Key conforming to the "hw.driver_version"
	// semantic conventions. It represents the driver version for the hardware
	// component.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "10.2.1-3"
	HwDriverVersionKey = attribute.Key("hw.driver_version")

	// HwEnclosureTypeKey is the attribute Key conforming to the "hw.enclosure.type"
	// semantic conventions. It represents the type of the enclosure (useful for
	// modular systems).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Computer", "Storage", "Switch"
	HwEnclosureTypeKey = attribute.Key("hw.enclosure.type")

	// HwFirmwareVersionKey is the attribute Key conforming to the
	// "hw.firmware_version" semantic conventions. It represents the firmware
	// version of the hardware component.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2.0.1"
	HwFirmwareVersionKey = attribute.Key("hw.firmware_version")

	// HwGpuTaskKey is the attribute Key conforming to the "hw.gpu.task" semantic
	// conventions. It represents the type of task the GPU is performing.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	HwGpuTaskKey = attribute.Key("hw.gpu.task")

	// HwIDKey is the attribute Key conforming to the "hw.id" semantic conventions.
	// It represents an identifier for the hardware component, unique within the
	// monitored host.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "win32battery_battery_testsysa33_1"
	HwIDKey = attribute.Key("hw.id")

	// HwLimitTypeKey is the attribute Key conforming to the "hw.limit_type"
	// semantic conventions. It represents the type of limit for hardware
	// components.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	HwLimitTypeKey = attribute.Key("hw.limit_type")

	// HwLogicalDiskRaidLevelKey is the attribute Key conforming to the
	// "hw.logical_disk.raid_level" semantic conventions. It represents the RAID
	// Level of the logical disk.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "RAID0+1", "RAID5", "RAID10"
	HwLogicalDiskRaidLevelKey = attribute.Key("hw.logical_disk.raid_level")

	// HwLogicalDiskStateKey is the attribute Key conforming to the
	// "hw.logical_disk.state" semantic conventions. It represents the state of the
	// logical disk space usage.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	HwLogicalDiskStateKey = attribute.Key("hw.logical_disk.state")

	// HwMemoryTypeKey is the attribute Key conforming to the "hw.memory.type"
	// semantic conventions. It represents the type of the memory module.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "DDR4", "DDR5", "LPDDR5"
	HwMemoryTypeKey = attribute.Key("hw.memory.type")

	// HwModelKey is the attribute Key conforming to the "hw.model" semantic
	// conventions. It represents the descriptive model name of the hardware
	// component.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "PERC H740P", "Intel(R) Core(TM) i7-10700K", "Dell XPS 15 Battery"
	HwModelKey = attribute.Key("hw.model")

	// HwNameKey is the attribute Key conforming to the "hw.name" semantic
	// conventions. It represents an easily-recognizable name for the hardware
	// component.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "eth0"
	HwNameKey = attribute.Key("hw.name")

	// HwNetworkLogicalAddressesKey is the attribute Key conforming to the
	// "hw.network.logical_addresses" semantic conventions. It represents the
	// logical addresses of the adapter (e.g. IP address, or WWPN).
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "172.16.8.21", "57.11.193.42"
	HwNetworkLogicalAddressesKey = attribute.Key("hw.network.logical_addresses")

	// HwNetworkPhysicalAddressKey is the attribute Key conforming to the
	// "hw.network.physical_address" semantic conventions. It represents the
	// physical address of the adapter (e.g. MAC address, or WWNN).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "00-90-F5-E9-7B-36"
	HwNetworkPhysicalAddressKey = attribute.Key("hw.network.physical_address")

	// HwParentKey is the attribute Key conforming to the "hw.parent" semantic
	// conventions. It represents the unique identifier of the parent component
	// (typically the `hw.id` attribute of the enclosure, or disk controller).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "dellStorage_perc_0"
	HwParentKey = attribute.Key("hw.parent")

	// HwPhysicalDiskSmartAttributeKey is the attribute Key conforming to the
	// "hw.physical_disk.smart_attribute" semantic conventions. It represents the
	// [S.M.A.R.T.] (Self-Monitoring, Analysis, and Reporting Technology) attribute
	// of the physical disk.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Spin Retry Count", "Seek Error Rate", "Raw Read Error Rate"
	//
	// [S.M.A.R.T.]: https://wikipedia.org/wiki/S.M.A.R.T.
	HwPhysicalDiskSmartAttributeKey = attribute.Key("hw.physical_disk.smart_attribute")

	// HwPhysicalDiskStateKey is the attribute Key conforming to the
	// "hw.physical_disk.state" semantic conventions. It represents the state of the
	// physical disk endurance utilization.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	HwPhysicalDiskStateKey = attribute.Key("hw.physical_disk.state")

	// HwPhysicalDiskTypeKey is the attribute Key conforming to the
	// "hw.physical_disk.type" semantic conventions. It represents the type of the
	// physical disk.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "HDD", "SSD", "10K"
	HwPhysicalDiskTypeKey = attribute.Key("hw.physical_disk.type")

	// HwSensorLocationKey is the attribute Key conforming to the
	// "hw.sensor_location" semantic conventions. It represents the location of the
	// sensor.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "cpu0", "ps1", "INLET", "CPU0_DIE", "AMBIENT", "MOTHERBOARD", "PS0
	// V3_3", "MAIN_12V", "CPU_VCORE"
	HwSensorLocationKey = attribute.Key("hw.sensor_location")

	// HwSerialNumberKey is the attribute Key conforming to the "hw.serial_number"
	// semantic conventions. It represents the serial number of the hardware
	// component.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "CNFCP0123456789"
	HwSerialNumberKey = attribute.Key("hw.serial_number")

	// HwStateKey is the attribute Key conforming to the "hw.state" semantic
	// conventions. It represents the current state of the component.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	HwStateKey = attribute.Key("hw.state")

	// HwTapeDriveOperationTypeKey is the attribute Key conforming to the
	// "hw.tape_drive.operation_type" semantic conventions. It represents the type
	// of tape drive operation.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	HwTapeDriveOperationTypeKey = attribute.Key("hw.tape_drive.operation_type")

	// HwTypeKey is the attribute Key conforming to the "hw.type" semantic
	// conventions. It represents the type of the component.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: Describes the category of the hardware component for which `hw.state`
	// is being reported. For example, `hw.type=temperature` along with
	// `hw.state=degraded` would indicate that the temperature of the hardware
	// component has been reported as `degraded`.
	HwTypeKey = attribute.Key("hw.type")

	// HwVendorKey is the attribute Key conforming to the "hw.vendor" semantic
	// conventions. It represents the vendor name of the hardware component.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Dell", "HP", "Intel", "AMD", "LSI", "Lenovo"
	HwVendorKey = attribute.Key("hw.vendor")
)

// HwBatteryCapacity returns an attribute KeyValue conforming to the
// "hw.battery.capacity" semantic conventions. It represents the design capacity
// in Watts-hours or Amper-hours.
func HwBatteryCapacity(val string) attribute.KeyValue {
	return HwBatteryCapacityKey.String(val)
}

// HwBatteryChemistry returns an attribute KeyValue conforming to the
// "hw.battery.chemistry" semantic conventions. It represents the battery
// [chemistry], e.g. Lithium-Ion, Nickel-Cadmium, etc.
//
// [chemistry]: https://schemas.dmtf.org/wbem/cim-html/2.31.0/CIM_Battery.html
func HwBatteryChemistry(val string) attribute.KeyValue {
	return HwBatteryChemistryKey.String(val)
}

// HwBiosVersion returns an attribute KeyValue conforming to the
// "hw.bios_version" semantic conventions. It represents the BIOS version of the
// hardware component.
func HwBiosVersion(val string) attribute.KeyValue {
	return HwBiosVersionKey.String(val)
}

// HwDriverVersion returns an attribute KeyValue conforming to the
// "hw.driver_version" semantic conventions. It represents the driver version for
// the hardware component.
func HwDriverVersion(val string) attribute.KeyValue {
	return HwDriverVersionKey.String(val)
}

// HwEnclosureType returns an attribute KeyValue conforming to the
// "hw.enclosure.type" semantic conventions. It represents the type of the
// enclosure (useful for modular systems).
func HwEnclosureType(val string) attribute.KeyValue {
	return HwEnclosureTypeKey.String(val)
}

// HwFirmwareVersion returns an attribute KeyValue conforming to the
// "hw.firmware_version" semantic conventions. It represents the firmware version
// of the hardware component.
func HwFirmwareVersion(val string) attribute.KeyValue {
	return HwFirmwareVersionKey.String(val)
}

// HwID returns an attribute KeyValue conforming to the "hw.id" semantic
// conventions. It represents an identifier for the hardware component, unique
// within the monitored host.
func HwID(val string) attribute.KeyValue {
	return HwIDKey.String(val)
}

// HwLogicalDiskRaidLevel returns an attribute KeyValue conforming to the
// "hw.logical_disk.raid_level" semantic conventions. It represents the RAID
// Level of the logical disk.
func HwLogicalDiskRaidLevel(val string) attribute.KeyValue {
	return HwLogicalDiskRaidLevelKey.String(val)
}

// HwMemoryType returns an attribute KeyValue conforming to the "hw.memory.type"
// semantic conventions. It represents the type of the memory module.
func HwMemoryType(val string) attribute.KeyValue {
	return HwMemoryTypeKey.String(val)
}

// HwModel returns an attribute KeyValue conforming to the "hw.model" semantic
// conventions. It represents the descriptive model name of the hardware
// component.
func HwModel(val string) attribute.KeyValue {
	return HwModelKey.String(val)
}

// HwName returns an attribute KeyValue conforming to the "hw.name" semantic
// conventions. It represents an easily-recognizable name for the hardware
// component.
func HwName(val string) attribute.KeyValue {
	return HwNameKey.String(val)
}

// HwNetworkLogicalAddresses returns an attribute KeyValue conforming to the
// "hw.network.logical_addresses" semantic conventions. It represents the logical
// addresses of the adapter (e.g. IP address, or WWPN).
func HwNetworkLogicalAddresses(val ...string) attribute.KeyValue {
	return HwNetworkLogicalAddressesKey.StringSlice(val)
}

// HwNetworkPhysicalAddress returns an attribute KeyValue conforming to the
// "hw.network.physical_address" semantic conventions. It represents the physical
// address of the adapter (e.g. MAC address, or WWNN).
func HwNetworkPhysicalAddress(val string) attribute.KeyValue {
	return HwNetworkPhysicalAddressKey.String(val)
}

// HwParent returns an attribute KeyValue conforming to the "hw.parent" semantic
// conventions. It represents the unique identifier of the parent component
// (typically the `hw.id` attribute of the enclosure, or disk controller).
func HwParent(val string) attribute.KeyValue {
	return HwParentKey.String(val)
}

// HwPhysicalDiskSmartAttribute returns an attribute KeyValue conforming to the
// "hw.physical_disk.smart_attribute" semantic conventions. It represents the
// [S.M.A.R.T.] (Self-Monitoring, Analysis, and Reporting Technology) attribute
// of the physical disk.
//
// [S.M.A.R.T.]: https://wikipedia.org/wiki/S.M.A.R.T.
func HwPhysicalDiskSmartAttribute(val string) attribute.KeyValue {
	return HwPhysicalDiskSmartAttributeKey.String(val)
}

// HwPhysicalDiskType returns an attribute KeyValue conforming to the
// "hw.physical_disk.type" semantic conventions. It represents the type of the
// physical disk.
func HwPhysicalDiskType(val string) attribute.KeyValue {
	return HwPhysicalDiskTypeKey.String(val)
}

// HwSensorLocation returns an attribute KeyValue conforming to the
// "hw.sensor_location" semantic conventions. It represents the location of the
// sensor.
func HwSensorLocation(val string) attribute.KeyValue {
	return HwSensorLocationKey.String(val)
}

// HwSerialNumber returns an attribute KeyValue conforming to the
// "hw.serial_number" semantic conventions. It represents the serial number of
// the hardware component.
func HwSerialNumber(val string) attribute.KeyValue {
	return HwSerialNumberKey.String(val)
}

// HwVendor returns an attribute KeyValue conforming to the "hw.vendor" semantic
// conventions. It represents the vendor name of the hardware component.
func HwVendor(val string) attribute.KeyValue {
	return HwVendorKey.String(val)
}

// Enum values for hw.battery.state
var (
	// Charging
	// Stability: development
	HwBatteryStateCharging = HwBatteryStateKey.String("charging")
	// Discharging
	// Stability: development
	HwBatteryStateDischarging = HwBatteryStateKey.String("discharging")
)

// Enum values for hw.gpu.task
var (
	// Decoder
	// Stability: development
	HwGpuTaskDecoder = HwGpuTaskKey.String("decoder")
	// Encoder
	// Stability: development
	HwGpuTaskEncoder = HwGpuTaskKey.String("encoder")
	// General
	// Stability: development
	HwGpuTaskGeneral = HwGpuTaskKey.String("general")
)

// Enum values for hw.limit_type
var (
	// Critical
	// Stability: development
	HwLimitTypeCritical = HwLimitTypeKey.String("critical")
	// Degraded
	// Stability: development
	HwLimitTypeDegraded = HwLimitTypeKey.String("degraded")
	// High Critical
	// Stability: development
	HwLimitTypeHighCritical = HwLimitTypeKey.String("high.critical")
	// High Degraded
	// Stability: development
	HwLimitTypeHighDegraded = HwLimitTypeKey.String("high.degraded")
	// Low Critical
	// Stability: development
	HwLimitTypeLowCritical = HwLimitTypeKey.String("low.critical")
	// Low Degraded
	// Stability: development
	HwLimitTypeLowDegraded = HwLimitTypeKey.String("low.degraded")
	// Maximum
	// Stability: development
	HwLimitTypeMax = HwLimitTypeKey.String("max")
	// Throttled
	// Stability: development
	HwLimitTypeThrottled = HwLimitTypeKey.String("throttled")
	// Turbo
	// Stability: development
	HwLimitTypeTurbo = HwLimitTypeKey.String("turbo")
)

// Enum values for hw.logical_disk.state
var (
	// Used
	// Stability: development
	HwLogicalDiskStateUsed = HwLogicalDiskStateKey.String("used")
	// Free
	// Stability: development
	HwLogicalDiskStateFree = HwLogicalDiskStateKey.String("free")
)

// Enum values for hw.physical_disk.state
var (
	// Remaining
	// Stability: development
	HwPhysicalDiskStateRemaining = HwPhysicalDiskStateKey.String("remaining")
)

// Enum values for hw.state
var (
	// Degraded
	// Stability: development
	HwStateDegraded = HwStateKey.String("degraded")
	// Failed
	// Stability: development
	HwStateFailed = HwStateKey.String("failed")
	// Needs Cleaning
	// Stability: development
	HwStateNeedsCleaning = HwStateKey.String("needs_cleaning")
	// OK
	// Stability: development
	HwStateOk = HwStateKey.String("ok")
	// Predicted Failure
	// Stability: development
	HwStatePredictedFailure = HwStateKey.String("predicted_failure")
)

// Enum values for hw.tape_drive.operation_type
var (
	// Mount
	// Stability: development
	HwTapeDriveOperationTypeMount = HwTapeDriveOperationTypeKey.String("mount")
	// Unmount
	// Stability: development
	HwTapeDriveOperationTypeUnmount = HwTapeDriveOperationTypeKey.String("unmount")
	// Clean
	// Stability: development
	HwTapeDriveOperationTypeClean = HwTapeDriveOperationTypeKey.String("clean")
)

// Enum values for hw.type
var (
	// Battery
	// Stability: development
	HwTypeBattery = HwTypeKey.String("battery")
	// CPU
	// Stability: development
	HwTypeCPU = HwTypeKey.String("cpu")
	// Disk controller
	// Stability: development
	HwTypeDiskController = HwTypeKey.String("disk_controller")
	// Enclosure
	// Stability: development
	HwTypeEnclosure = HwTypeKey.String("enclosure")
	// Fan
	// Stability: development
	HwTypeFan = HwTypeKey.String("fan")
	// GPU
	// Stability: development
	HwTypeGpu = HwTypeKey.String("gpu")
	// Logical disk
	// Stability: development
	HwTypeLogicalDisk = HwTypeKey.String("logical_disk")
	// Memory
	// Stability: development
	HwTypeMemory = HwTypeKey.String("memory")
	// Network
	// Stability: development
	HwTypeNetwork = HwTypeKey.String("network")
	// Physical disk
	// Stability: development
	HwTypePhysicalDisk = HwTypeKey.String("physical_disk")
	// Power supply
	// Stability: development
	HwTypePowerSupply = HwTypeKey.String("power_supply")
	// Tape drive
	// Stability: development
	HwTypeTapeDrive = HwTypeKey.String("tape_drive")
	// Temperature
	// Stability: development
	HwTypeTemperature = HwTypeKey.String("temperature")
	// Voltage
	// Stability: development
	HwTypeVoltage = HwTypeKey.String("voltage")
)

// Namespace: ios
const (
	// IOSAppStateKey is the attribute Key conforming to the "ios.app.state"
	// semantic conventions. It represents the this attribute represents the state
	// of the application.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: The iOS lifecycle states are defined in the
	// [UIApplicationDelegate documentation], and from which the `OS terminology`
	// column values are derived.
	//
	// [UIApplicationDelegate documentation]: https://developer.apple.com/documentation/uikit/uiapplicationdelegate
	IOSAppStateKey = attribute.Key("ios.app.state")
)

// Enum values for ios.app.state
var (
	// The app has become `active`. Associated with UIKit notification
	// `applicationDidBecomeActive`.
	//
	// Stability: development
	IOSAppStateActive = IOSAppStateKey.String("active")
	// The app is now `inactive`. Associated with UIKit notification
	// `applicationWillResignActive`.
	//
	// Stability: development
	IOSAppStateInactive = IOSAppStateKey.String("inactive")
	// The app is now in the background. This value is associated with UIKit
	// notification `applicationDidEnterBackground`.
	//
	// Stability: development
	IOSAppStateBackground = IOSAppStateKey.String("background")
	// The app is now in the foreground. This value is associated with UIKit
	// notification `applicationWillEnterForeground`.
	//
	// Stability: development
	IOSAppStateForeground = IOSAppStateKey.String("foreground")
	// The app is about to terminate. Associated with UIKit notification
	// `applicationWillTerminate`.
	//
	// Stability: development
	IOSAppStateTerminate = IOSAppStateKey.String("terminate")
)

// Namespace: k8s
const (
	// K8SClusterNameKey is the attribute Key conforming to the "k8s.cluster.name"
	// semantic conventions. It represents the name of the cluster.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "opentelemetry-cluster"
	K8SClusterNameKey = attribute.Key("k8s.cluster.name")

	// K8SClusterUIDKey is the attribute Key conforming to the "k8s.cluster.uid"
	// semantic conventions. It represents a pseudo-ID for the cluster, set to the
	// UID of the `kube-system` namespace.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "218fc5a9-a5f1-4b54-aa05-46717d0ab26d"
	// Note: K8s doesn't have support for obtaining a cluster ID. If this is ever
	// added, we will recommend collecting the `k8s.cluster.uid` through the
	// official APIs. In the meantime, we are able to use the `uid` of the
	// `kube-system` namespace as a proxy for cluster ID. Read on for the
	// rationale.
	//
	// Every object created in a K8s cluster is assigned a distinct UID. The
	// `kube-system` namespace is used by Kubernetes itself and will exist
	// for the lifetime of the cluster. Using the `uid` of the `kube-system`
	// namespace is a reasonable proxy for the K8s ClusterID as it will only
	// change if the cluster is rebuilt. Furthermore, Kubernetes UIDs are
	// UUIDs as standardized by
	// [ISO/IEC 9834-8 and ITU-T X.667].
	// Which states:
	//
	// > If generated according to one of the mechanisms defined in Rec.
	// > ITU-T X.667 | ISO/IEC 9834-8, a UUID is either guaranteed to be
	// > different from all other UUIDs generated before 3603 A.D., or is
	// > extremely likely to be different (depending on the mechanism chosen).
	//
	// Therefore, UIDs between clusters should be extremely unlikely to
	// conflict.
	//
	// [ISO/IEC 9834-8 and ITU-T X.667]: https://www.itu.int/ITU-T/studygroups/com17/oid.html
	K8SClusterUIDKey = attribute.Key("k8s.cluster.uid")

	// K8SContainerNameKey is the attribute Key conforming to the
	// "k8s.container.name" semantic conventions. It represents the name of the
	// Container from Pod specification, must be unique within a Pod. Container
	// runtime usually uses different globally unique name (`container.name`).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "redis"
	K8SContainerNameKey = attribute.Key("k8s.container.name")

	// K8SContainerRestartCountKey is the attribute Key conforming to the
	// "k8s.container.restart_count" semantic conventions. It represents the number
	// of times the container was restarted. This attribute can be used to identify
	// a particular container (running or stopped) within a container spec.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	K8SContainerRestartCountKey = attribute.Key("k8s.container.restart_count")

	// K8SContainerStatusLastTerminatedReasonKey is the attribute Key conforming to
	// the "k8s.container.status.last_terminated_reason" semantic conventions. It
	// represents the last terminated reason of the Container.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Evicted", "Error"
	K8SContainerStatusLastTerminatedReasonKey = attribute.Key("k8s.container.status.last_terminated_reason")

	// K8SContainerStatusReasonKey is the attribute Key conforming to the
	// "k8s.container.status.reason" semantic conventions. It represents the reason
	// for the container state. Corresponds to the `reason` field of the:
	// [K8s ContainerStateWaiting] or [K8s ContainerStateTerminated].
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "ContainerCreating", "CrashLoopBackOff",
	// "CreateContainerConfigError", "ErrImagePull", "ImagePullBackOff",
	// "OOMKilled", "Completed", "Error", "ContainerCannotRun"
	//
	// [K8s ContainerStateWaiting]: https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.30/#containerstatewaiting-v1-core
	// [K8s ContainerStateTerminated]: https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.30/#containerstateterminated-v1-core
	K8SContainerStatusReasonKey = attribute.Key("k8s.container.status.reason")

	// K8SContainerStatusStateKey is the attribute Key conforming to the
	// "k8s.container.status.state" semantic conventions. It represents the state of
	// the container. [K8s ContainerState].
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "terminated", "running", "waiting"
	//
	// [K8s ContainerState]: https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.30/#containerstate-v1-core
	K8SContainerStatusStateKey = attribute.Key("k8s.container.status.state")

	// K8SCronJobNameKey is the attribute Key conforming to the "k8s.cronjob.name"
	// semantic conventions. It represents the name of the CronJob.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "opentelemetry"
	K8SCronJobNameKey = attribute.Key("k8s.cronjob.name")

	// K8SCronJobUIDKey is the attribute Key conforming to the "k8s.cronjob.uid"
	// semantic conventions. It represents the UID of the CronJob.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "275ecb36-5aa8-4c2a-9c47-d8bb681b9aff"
	K8SCronJobUIDKey = attribute.Key("k8s.cronjob.uid")

	// K8SDaemonSetNameKey is the attribute Key conforming to the
	// "k8s.daemonset.name" semantic conventions. It represents the name of the
	// DaemonSet.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "opentelemetry"
	K8SDaemonSetNameKey = attribute.Key("k8s.daemonset.name")

	// K8SDaemonSetUIDKey is the attribute Key conforming to the "k8s.daemonset.uid"
	// semantic conventions. It represents the UID of the DaemonSet.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "275ecb36-5aa8-4c2a-9c47-d8bb681b9aff"
	K8SDaemonSetUIDKey = attribute.Key("k8s.daemonset.uid")

	// K8SDeploymentNameKey is the attribute Key conforming to the
	// "k8s.deployment.name" semantic conventions. It represents the name of the
	// Deployment.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "opentelemetry"
	K8SDeploymentNameKey = attribute.Key("k8s.deployment.name")

	// K8SDeploymentUIDKey is the attribute Key conforming to the
	// "k8s.deployment.uid" semantic conventions. It represents the UID of the
	// Deployment.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "275ecb36-5aa8-4c2a-9c47-d8bb681b9aff"
	K8SDeploymentUIDKey = attribute.Key("k8s.deployment.uid")

	// K8SHPAMetricTypeKey is the attribute Key conforming to the
	// "k8s.hpa.metric.type" semantic conventions. It represents the type of metric
	// source for the horizontal pod autoscaler.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Resource", "ContainerResource"
	// Note: This attribute reflects the `type` field of spec.metrics[] in the HPA.
	K8SHPAMetricTypeKey = attribute.Key("k8s.hpa.metric.type")

	// K8SHPANameKey is the attribute Key conforming to the "k8s.hpa.name" semantic
	// conventions. It represents the name of the horizontal pod autoscaler.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "opentelemetry"
	K8SHPANameKey = attribute.Key("k8s.hpa.name")

	// K8SHPAScaletargetrefAPIVersionKey is the attribute Key conforming to the
	// "k8s.hpa.scaletargetref.api_version" semantic conventions. It represents the
	// API version of the target resource to scale for the HorizontalPodAutoscaler.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "apps/v1", "autoscaling/v2"
	// Note: This maps to the `apiVersion` field in the `scaleTargetRef` of the HPA
	// spec.
	K8SHPAScaletargetrefAPIVersionKey = attribute.Key("k8s.hpa.scaletargetref.api_version")

	// K8SHPAScaletargetrefKindKey is the attribute Key conforming to the
	// "k8s.hpa.scaletargetref.kind" semantic conventions. It represents the kind of
	// the target resource to scale for the HorizontalPodAutoscaler.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Deployment", "StatefulSet"
	// Note: This maps to the `kind` field in the `scaleTargetRef` of the HPA spec.
	K8SHPAScaletargetrefKindKey = attribute.Key("k8s.hpa.scaletargetref.kind")

	// K8SHPAScaletargetrefNameKey is the attribute Key conforming to the
	// "k8s.hpa.scaletargetref.name" semantic conventions. It represents the name of
	// the target resource to scale for the HorizontalPodAutoscaler.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "my-deployment", "my-statefulset"
	// Note: This maps to the `name` field in the `scaleTargetRef` of the HPA spec.
	K8SHPAScaletargetrefNameKey = attribute.Key("k8s.hpa.scaletargetref.name")

	// K8SHPAUIDKey is the attribute Key conforming to the "k8s.hpa.uid" semantic
	// conventions. It represents the UID of the horizontal pod autoscaler.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "275ecb36-5aa8-4c2a-9c47-d8bb681b9aff"
	K8SHPAUIDKey = attribute.Key("k8s.hpa.uid")

	// K8SHugepageSizeKey is the attribute Key conforming to the "k8s.hugepage.size"
	// semantic conventions. It represents the size (identifier) of the K8s huge
	// page.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2Mi"
	K8SHugepageSizeKey = attribute.Key("k8s.hugepage.size")

	// K8SJobNameKey is the attribute Key conforming to the "k8s.job.name" semantic
	// conventions. It represents the name of the Job.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "opentelemetry"
	K8SJobNameKey = attribute.Key("k8s.job.name")

	// K8SJobUIDKey is the attribute Key conforming to the "k8s.job.uid" semantic
	// conventions. It represents the UID of the Job.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "275ecb36-5aa8-4c2a-9c47-d8bb681b9aff"
	K8SJobUIDKey = attribute.Key("k8s.job.uid")

	// K8SNamespaceNameKey is the attribute Key conforming to the
	// "k8s.namespace.name" semantic conventions. It represents the name of the
	// namespace that the pod is running in.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "default"
	K8SNamespaceNameKey = attribute.Key("k8s.namespace.name")

	// K8SNamespacePhaseKey is the attribute Key conforming to the
	// "k8s.namespace.phase" semantic conventions. It represents the phase of the
	// K8s namespace.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "active", "terminating"
	// Note: This attribute aligns with the `phase` field of the
	// [K8s NamespaceStatus]
	//
	// [K8s NamespaceStatus]: https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.30/#namespacestatus-v1-core
	K8SNamespacePhaseKey = attribute.Key("k8s.namespace.phase")

	// K8SNodeConditionStatusKey is the attribute Key conforming to the
	// "k8s.node.condition.status" semantic conventions. It represents the status of
	// the condition, one of True, False, Unknown.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "true", "false", "unknown"
	// Note: This attribute aligns with the `status` field of the
	// [NodeCondition]
	//
	// [NodeCondition]: https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.30/#nodecondition-v1-core
	K8SNodeConditionStatusKey = attribute.Key("k8s.node.condition.status")

	// K8SNodeConditionTypeKey is the attribute Key conforming to the
	// "k8s.node.condition.type" semantic conventions. It represents the condition
	// type of a K8s Node.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Ready", "DiskPressure"
	// Note: K8s Node conditions as described
	// by [K8s documentation].
	//
	// This attribute aligns with the `type` field of the
	// [NodeCondition]
	//
	// The set of possible values is not limited to those listed here. Managed
	// Kubernetes environments,
	// or custom controllers MAY introduce additional node condition types.
	// When this occurs, the exact value as reported by the Kubernetes API SHOULD be
	// used.
	//
	// [K8s documentation]: https://v1-32.docs.kubernetes.io/docs/reference/node/node-status/#condition
	// [NodeCondition]: https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.30/#nodecondition-v1-core
	K8SNodeConditionTypeKey = attribute.Key("k8s.node.condition.type")

	// K8SNodeNameKey is the attribute Key conforming to the "k8s.node.name"
	// semantic conventions. It represents the name of the Node.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "node-1"
	K8SNodeNameKey = attribute.Key("k8s.node.name")

	// K8SNodeUIDKey is the attribute Key conforming to the "k8s.node.uid" semantic
	// conventions. It represents the UID of the Node.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "1eb3a0c6-0477-4080-a9cb-0cb7db65c6a2"
	K8SNodeUIDKey = attribute.Key("k8s.node.uid")

	// K8SPodNameKey is the attribute Key conforming to the "k8s.pod.name" semantic
	// conventions. It represents the name of the Pod.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "opentelemetry-pod-autoconf"
	K8SPodNameKey = attribute.Key("k8s.pod.name")

	// K8SPodUIDKey is the attribute Key conforming to the "k8s.pod.uid" semantic
	// conventions. It represents the UID of the Pod.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "275ecb36-5aa8-4c2a-9c47-d8bb681b9aff"
	K8SPodUIDKey = attribute.Key("k8s.pod.uid")

	// K8SReplicaSetNameKey is the attribute Key conforming to the
	// "k8s.replicaset.name" semantic conventions. It represents the name of the
	// ReplicaSet.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "opentelemetry"
	K8SReplicaSetNameKey = attribute.Key("k8s.replicaset.name")

	// K8SReplicaSetUIDKey is the attribute Key conforming to the
	// "k8s.replicaset.uid" semantic conventions. It represents the UID of the
	// ReplicaSet.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "275ecb36-5aa8-4c2a-9c47-d8bb681b9aff"
	K8SReplicaSetUIDKey = attribute.Key("k8s.replicaset.uid")

	// K8SReplicationControllerNameKey is the attribute Key conforming to the
	// "k8s.replicationcontroller.name" semantic conventions. It represents the name
	// of the replication controller.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "opentelemetry"
	K8SReplicationControllerNameKey = attribute.Key("k8s.replicationcontroller.name")

	// K8SReplicationControllerUIDKey is the attribute Key conforming to the
	// "k8s.replicationcontroller.uid" semantic conventions. It represents the UID
	// of the replication controller.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "275ecb36-5aa8-4c2a-9c47-d8bb681b9aff"
	K8SReplicationControllerUIDKey = attribute.Key("k8s.replicationcontroller.uid")

	// K8SResourceQuotaNameKey is the attribute Key conforming to the
	// "k8s.resourcequota.name" semantic conventions. It represents the name of the
	// resource quota.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "opentelemetry"
	K8SResourceQuotaNameKey = attribute.Key("k8s.resourcequota.name")

	// K8SResourceQuotaResourceNameKey is the attribute Key conforming to the
	// "k8s.resourcequota.resource_name" semantic conventions. It represents the
	// name of the K8s resource a resource quota defines.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "count/replicationcontrollers"
	// Note: The value for this attribute can be either the full
	// `count/<resource>[.<group>]` string (e.g., count/deployments.apps,
	// count/pods), or, for certain core Kubernetes resources, just the resource
	// name (e.g., pods, services, configmaps). Both forms are supported by
	// Kubernetes for object count quotas. See
	// [Kubernetes Resource Quotas documentation] for more details.
	//
	// [Kubernetes Resource Quotas documentation]: https://kubernetes.io/docs/concepts/policy/resource-quotas/#object-count-quota
	K8SResourceQuotaResourceNameKey = attribute.Key("k8s.resourcequota.resource_name")

	// K8SResourceQuotaUIDKey is the attribute Key conforming to the
	// "k8s.resourcequota.uid" semantic conventions. It represents the UID of the
	// resource quota.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "275ecb36-5aa8-4c2a-9c47-d8bb681b9aff"
	K8SResourceQuotaUIDKey = attribute.Key("k8s.resourcequota.uid")

	// K8SStatefulSetNameKey is the attribute Key conforming to the
	// "k8s.statefulset.name" semantic conventions. It represents the name of the
	// StatefulSet.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "opentelemetry"
	K8SStatefulSetNameKey = attribute.Key("k8s.statefulset.name")

	// K8SStatefulSetUIDKey is the attribute Key conforming to the
	// "k8s.statefulset.uid" semantic conventions. It represents the UID of the
	// StatefulSet.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "275ecb36-5aa8-4c2a-9c47-d8bb681b9aff"
	K8SStatefulSetUIDKey = attribute.Key("k8s.statefulset.uid")

	// K8SStorageclassNameKey is the attribute Key conforming to the
	// "k8s.storageclass.name" semantic conventions. It represents the name of K8s
	// [StorageClass] object.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "gold.storageclass.storage.k8s.io"
	//
	// [StorageClass]: https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.30/#storageclass-v1-storage-k8s-io
	K8SStorageclassNameKey = attribute.Key("k8s.storageclass.name")

	// K8SVolumeNameKey is the attribute Key conforming to the "k8s.volume.name"
	// semantic conventions. It represents the name of the K8s volume.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "volume0"
	K8SVolumeNameKey = attribute.Key("k8s.volume.name")

	// K8SVolumeTypeKey is the attribute Key conforming to the "k8s.volume.type"
	// semantic conventions. It represents the type of the K8s volume.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "emptyDir", "persistentVolumeClaim"
	K8SVolumeTypeKey = attribute.Key("k8s.volume.type")
)

// K8SClusterName returns an attribute KeyValue conforming to the
// "k8s.cluster.name" semantic conventions. It represents the name of the
// cluster.
func K8SClusterName(val string) attribute.KeyValue {
	return K8SClusterNameKey.String(val)
}

// K8SClusterUID returns an attribute KeyValue conforming to the
// "k8s.cluster.uid" semantic conventions. It represents a pseudo-ID for the
// cluster, set to the UID of the `kube-system` namespace.
func K8SClusterUID(val string) attribute.KeyValue {
	return K8SClusterUIDKey.String(val)
}

// K8SContainerName returns an attribute KeyValue conforming to the
// "k8s.container.name" semantic conventions. It represents the name of the
// Container from Pod specification, must be unique within a Pod. Container
// runtime usually uses different globally unique name (`container.name`).
func K8SContainerName(val string) attribute.KeyValue {
	return K8SContainerNameKey.String(val)
}

// K8SContainerRestartCount returns an attribute KeyValue conforming to the
// "k8s.container.restart_count" semantic conventions. It represents the number
// of times the container was restarted. This attribute can be used to identify a
// particular container (running or stopped) within a container spec.
func K8SContainerRestartCount(val int) attribute.KeyValue {
	return K8SContainerRestartCountKey.Int(val)
}

// K8SContainerStatusLastTerminatedReason returns an attribute KeyValue
// conforming to the "k8s.container.status.last_terminated_reason" semantic
// conventions. It represents the last terminated reason of the Container.
func K8SContainerStatusLastTerminatedReason(val string) attribute.KeyValue {
	return K8SContainerStatusLastTerminatedReasonKey.String(val)
}

// K8SCronJobAnnotation returns an attribute KeyValue conforming to the
// "k8s.cronjob.annotation" semantic conventions. It represents the cronjob
// annotation placed on the CronJob, the `<key>` being the annotation name, the
// value being the annotation value.
func K8SCronJobAnnotation(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.cronjob.annotation."+key, val)
}

// K8SCronJobLabel returns an attribute KeyValue conforming to the
// "k8s.cronjob.label" semantic conventions. It represents the label placed on
// the CronJob, the `<key>` being the label name, the value being the label
// value.
func K8SCronJobLabel(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.cronjob.label."+key, val)
}

// K8SCronJobName returns an attribute KeyValue conforming to the
// "k8s.cronjob.name" semantic conventions. It represents the name of the
// CronJob.
func K8SCronJobName(val string) attribute.KeyValue {
	return K8SCronJobNameKey.String(val)
}

// K8SCronJobUID returns an attribute KeyValue conforming to the
// "k8s.cronjob.uid" semantic conventions. It represents the UID of the CronJob.
func K8SCronJobUID(val string) attribute.KeyValue {
	return K8SCronJobUIDKey.String(val)
}

// K8SDaemonSetAnnotation returns an attribute KeyValue conforming to the
// "k8s.daemonset.annotation" semantic conventions. It represents the annotation
// placed on the DaemonSet, the `<key>` being the annotation name, the value
// being the annotation value, even if the value is empty.
func K8SDaemonSetAnnotation(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.daemonset.annotation."+key, val)
}

// K8SDaemonSetLabel returns an attribute KeyValue conforming to the
// "k8s.daemonset.label" semantic conventions. It represents the label placed on
// the DaemonSet, the `<key>` being the label name, the value being the label
// value, even if the value is empty.
func K8SDaemonSetLabel(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.daemonset.label."+key, val)
}

// K8SDaemonSetName returns an attribute KeyValue conforming to the
// "k8s.daemonset.name" semantic conventions. It represents the name of the
// DaemonSet.
func K8SDaemonSetName(val string) attribute.KeyValue {
	return K8SDaemonSetNameKey.String(val)
}

// K8SDaemonSetUID returns an attribute KeyValue conforming to the
// "k8s.daemonset.uid" semantic conventions. It represents the UID of the
// DaemonSet.
func K8SDaemonSetUID(val string) attribute.KeyValue {
	return K8SDaemonSetUIDKey.String(val)
}

// K8SDeploymentAnnotation returns an attribute KeyValue conforming to the
// "k8s.deployment.annotation" semantic conventions. It represents the annotation
// placed on the Deployment, the `<key>` being the annotation name, the value
// being the annotation value, even if the value is empty.
func K8SDeploymentAnnotation(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.deployment.annotation."+key, val)
}

// K8SDeploymentLabel returns an attribute KeyValue conforming to the
// "k8s.deployment.label" semantic conventions. It represents the label placed on
// the Deployment, the `<key>` being the label name, the value being the label
// value, even if the value is empty.
func K8SDeploymentLabel(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.deployment.label."+key, val)
}

// K8SDeploymentName returns an attribute KeyValue conforming to the
// "k8s.deployment.name" semantic conventions. It represents the name of the
// Deployment.
func K8SDeploymentName(val string) attribute.KeyValue {
	return K8SDeploymentNameKey.String(val)
}

// K8SDeploymentUID returns an attribute KeyValue conforming to the
// "k8s.deployment.uid" semantic conventions. It represents the UID of the
// Deployment.
func K8SDeploymentUID(val string) attribute.KeyValue {
	return K8SDeploymentUIDKey.String(val)
}

// K8SHPAMetricType returns an attribute KeyValue conforming to the
// "k8s.hpa.metric.type" semantic conventions. It represents the type of metric
// source for the horizontal pod autoscaler.
func K8SHPAMetricType(val string) attribute.KeyValue {
	return K8SHPAMetricTypeKey.String(val)
}

// K8SHPAName returns an attribute KeyValue conforming to the "k8s.hpa.name"
// semantic conventions. It represents the name of the horizontal pod autoscaler.
func K8SHPAName(val string) attribute.KeyValue {
	return K8SHPANameKey.String(val)
}

// K8SHPAScaletargetrefAPIVersion returns an attribute KeyValue conforming to the
// "k8s.hpa.scaletargetref.api_version" semantic conventions. It represents the
// API version of the target resource to scale for the HorizontalPodAutoscaler.
func K8SHPAScaletargetrefAPIVersion(val string) attribute.KeyValue {
	return K8SHPAScaletargetrefAPIVersionKey.String(val)
}

// K8SHPAScaletargetrefKind returns an attribute KeyValue conforming to the
// "k8s.hpa.scaletargetref.kind" semantic conventions. It represents the kind of
// the target resource to scale for the HorizontalPodAutoscaler.
func K8SHPAScaletargetrefKind(val string) attribute.KeyValue {
	return K8SHPAScaletargetrefKindKey.String(val)
}

// K8SHPAScaletargetrefName returns an attribute KeyValue conforming to the
// "k8s.hpa.scaletargetref.name" semantic conventions. It represents the name of
// the target resource to scale for the HorizontalPodAutoscaler.
func K8SHPAScaletargetrefName(val string) attribute.KeyValue {
	return K8SHPAScaletargetrefNameKey.String(val)
}

// K8SHPAUID returns an attribute KeyValue conforming to the "k8s.hpa.uid"
// semantic conventions. It represents the UID of the horizontal pod autoscaler.
func K8SHPAUID(val string) attribute.KeyValue {
	return K8SHPAUIDKey.String(val)
}

// K8SHugepageSize returns an attribute KeyValue conforming to the
// "k8s.hugepage.size" semantic conventions. It represents the size (identifier)
// of the K8s huge page.
func K8SHugepageSize(val string) attribute.KeyValue {
	return K8SHugepageSizeKey.String(val)
}

// K8SJobAnnotation returns an attribute KeyValue conforming to the
// "k8s.job.annotation" semantic conventions. It represents the annotation placed
// on the Job, the `<key>` being the annotation name, the value being the
// annotation value, even if the value is empty.
func K8SJobAnnotation(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.job.annotation."+key, val)
}

// K8SJobLabel returns an attribute KeyValue conforming to the "k8s.job.label"
// semantic conventions. It represents the label placed on the Job, the `<key>`
// being the label name, the value being the label value, even if the value is
// empty.
func K8SJobLabel(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.job.label."+key, val)
}

// K8SJobName returns an attribute KeyValue conforming to the "k8s.job.name"
// semantic conventions. It represents the name of the Job.
func K8SJobName(val string) attribute.KeyValue {
	return K8SJobNameKey.String(val)
}

// K8SJobUID returns an attribute KeyValue conforming to the "k8s.job.uid"
// semantic conventions. It represents the UID of the Job.
func K8SJobUID(val string) attribute.KeyValue {
	return K8SJobUIDKey.String(val)
}

// K8SNamespaceAnnotation returns an attribute KeyValue conforming to the
// "k8s.namespace.annotation" semantic conventions. It represents the annotation
// placed on the Namespace, the `<key>` being the annotation name, the value
// being the annotation value, even if the value is empty.
func K8SNamespaceAnnotation(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.namespace.annotation."+key, val)
}

// K8SNamespaceLabel returns an attribute KeyValue conforming to the
// "k8s.namespace.label" semantic conventions. It represents the label placed on
// the Namespace, the `<key>` being the label name, the value being the label
// value, even if the value is empty.
func K8SNamespaceLabel(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.namespace.label."+key, val)
}

// K8SNamespaceName returns an attribute KeyValue conforming to the
// "k8s.namespace.name" semantic conventions. It represents the name of the
// namespace that the pod is running in.
func K8SNamespaceName(val string) attribute.KeyValue {
	return K8SNamespaceNameKey.String(val)
}

// K8SNodeAnnotation returns an attribute KeyValue conforming to the
// "k8s.node.annotation" semantic conventions. It represents the annotation
// placed on the Node, the `<key>` being the annotation name, the value being the
// annotation value, even if the value is empty.
func K8SNodeAnnotation(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.node.annotation."+key, val)
}

// K8SNodeLabel returns an attribute KeyValue conforming to the "k8s.node.label"
// semantic conventions. It represents the label placed on the Node, the `<key>`
// being the label name, the value being the label value, even if the value is
// empty.
func K8SNodeLabel(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.node.label."+key, val)
}

// K8SNodeName returns an attribute KeyValue conforming to the "k8s.node.name"
// semantic conventions. It represents the name of the Node.
func K8SNodeName(val string) attribute.KeyValue {
	return K8SNodeNameKey.String(val)
}

// K8SNodeUID returns an attribute KeyValue conforming to the "k8s.node.uid"
// semantic conventions. It represents the UID of the Node.
func K8SNodeUID(val string) attribute.KeyValue {
	return K8SNodeUIDKey.String(val)
}

// K8SPodAnnotation returns an attribute KeyValue conforming to the
// "k8s.pod.annotation" semantic conventions. It represents the annotation placed
// on the Pod, the `<key>` being the annotation name, the value being the
// annotation value.
func K8SPodAnnotation(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.pod.annotation."+key, val)
}

// K8SPodLabel returns an attribute KeyValue conforming to the "k8s.pod.label"
// semantic conventions. It represents the label placed on the Pod, the `<key>`
// being the label name, the value being the label value.
func K8SPodLabel(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.pod.label."+key, val)
}

// K8SPodName returns an attribute KeyValue conforming to the "k8s.pod.name"
// semantic conventions. It represents the name of the Pod.
func K8SPodName(val string) attribute.KeyValue {
	return K8SPodNameKey.String(val)
}

// K8SPodUID returns an attribute KeyValue conforming to the "k8s.pod.uid"
// semantic conventions. It represents the UID of the Pod.
func K8SPodUID(val string) attribute.KeyValue {
	return K8SPodUIDKey.String(val)
}

// K8SReplicaSetAnnotation returns an attribute KeyValue conforming to the
// "k8s.replicaset.annotation" semantic conventions. It represents the annotation
// placed on the ReplicaSet, the `<key>` being the annotation name, the value
// being the annotation value, even if the value is empty.
func K8SReplicaSetAnnotation(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.replicaset.annotation."+key, val)
}

// K8SReplicaSetLabel returns an attribute KeyValue conforming to the
// "k8s.replicaset.label" semantic conventions. It represents the label placed on
// the ReplicaSet, the `<key>` being the label name, the value being the label
// value, even if the value is empty.
func K8SReplicaSetLabel(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.replicaset.label."+key, val)
}

// K8SReplicaSetName returns an attribute KeyValue conforming to the
// "k8s.replicaset.name" semantic conventions. It represents the name of the
// ReplicaSet.
func K8SReplicaSetName(val string) attribute.KeyValue {
	return K8SReplicaSetNameKey.String(val)
}

// K8SReplicaSetUID returns an attribute KeyValue conforming to the
// "k8s.replicaset.uid" semantic conventions. It represents the UID of the
// ReplicaSet.
func K8SReplicaSetUID(val string) attribute.KeyValue {
	return K8SReplicaSetUIDKey.String(val)
}

// K8SReplicationControllerName returns an attribute KeyValue conforming to the
// "k8s.replicationcontroller.name" semantic conventions. It represents the name
// of the replication controller.
func K8SReplicationControllerName(val string) attribute.KeyValue {
	return K8SReplicationControllerNameKey.String(val)
}

// K8SReplicationControllerUID returns an attribute KeyValue conforming to the
// "k8s.replicationcontroller.uid" semantic conventions. It represents the UID of
// the replication controller.
func K8SReplicationControllerUID(val string) attribute.KeyValue {
	return K8SReplicationControllerUIDKey.String(val)
}

// K8SResourceQuotaName returns an attribute KeyValue conforming to the
// "k8s.resourcequota.name" semantic conventions. It represents the name of the
// resource quota.
func K8SResourceQuotaName(val string) attribute.KeyValue {
	return K8SResourceQuotaNameKey.String(val)
}

// K8SResourceQuotaResourceName returns an attribute KeyValue conforming to the
// "k8s.resourcequota.resource_name" semantic conventions. It represents the name
// of the K8s resource a resource quota defines.
func K8SResourceQuotaResourceName(val string) attribute.KeyValue {
	return K8SResourceQuotaResourceNameKey.String(val)
}

// K8SResourceQuotaUID returns an attribute KeyValue conforming to the
// "k8s.resourcequota.uid" semantic conventions. It represents the UID of the
// resource quota.
func K8SResourceQuotaUID(val string) attribute.KeyValue {
	return K8SResourceQuotaUIDKey.String(val)
}

// K8SStatefulSetAnnotation returns an attribute KeyValue conforming to the
// "k8s.statefulset.annotation" semantic conventions. It represents the
// annotation placed on the StatefulSet, the `<key>` being the annotation name,
// the value being the annotation value, even if the value is empty.
func K8SStatefulSetAnnotation(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.statefulset.annotation."+key, val)
}

// K8SStatefulSetLabel returns an attribute KeyValue conforming to the
// "k8s.statefulset.label" semantic conventions. It represents the label placed
// on the StatefulSet, the `<key>` being the label name, the value being the
// label value, even if the value is empty.
func K8SStatefulSetLabel(key string, val string) attribute.KeyValue {
	return attribute.String("k8s.statefulset.label."+key, val)
}

// K8SStatefulSetName returns an attribute KeyValue conforming to the
// "k8s.statefulset.name" semantic conventions. It represents the name of the
// StatefulSet.
func K8SStatefulSetName(val string) attribute.KeyValue {
	return K8SStatefulSetNameKey.String(val)
}

// K8SStatefulSetUID returns an attribute KeyValue conforming to the
// "k8s.statefulset.uid" semantic conventions. It represents the UID of the
// StatefulSet.
func K8SStatefulSetUID(val string) attribute.KeyValue {
	return K8SStatefulSetUIDKey.String(val)
}

// K8SStorageclassName returns an attribute KeyValue conforming to the
// "k8s.storageclass.name" semantic conventions. It represents the name of K8s
// [StorageClass] object.
//
// [StorageClass]: https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.30/#storageclass-v1-storage-k8s-io
func K8SStorageclassName(val string) attribute.KeyValue {
	return K8SStorageclassNameKey.String(val)
}

// K8SVolumeName returns an attribute KeyValue conforming to the
// "k8s.volume.name" semantic conventions. It represents the name of the K8s
// volume.
func K8SVolumeName(val string) attribute.KeyValue {
	return K8SVolumeNameKey.String(val)
}

// Enum values for k8s.container.status.reason
var (
	// The container is being created.
	// Stability: development
	K8SContainerStatusReasonContainerCreating = K8SContainerStatusReasonKey.String("ContainerCreating")
	// The container is in a crash loop back off state.
	// Stability: development
	K8SContainerStatusReasonCrashLoopBackOff = K8SContainerStatusReasonKey.String("CrashLoopBackOff")
	// There was an error creating the container configuration.
	// Stability: development
	K8SContainerStatusReasonCreateContainerConfigError = K8SContainerStatusReasonKey.String("CreateContainerConfigError")
	// There was an error pulling the container image.
	// Stability: development
	K8SContainerStatusReasonErrImagePull = K8SContainerStatusReasonKey.String("ErrImagePull")
	// The container image pull is in back off state.
	// Stability: development
	K8SContainerStatusReasonImagePullBackOff = K8SContainerStatusReasonKey.String("ImagePullBackOff")
	// The container was killed due to out of memory.
	// Stability: development
	K8SContainerStatusReasonOomKilled = K8SContainerStatusReasonKey.String("OOMKilled")
	// The container has completed execution.
	// Stability: development
	K8SContainerStatusReasonCompleted = K8SContainerStatusReasonKey.String("Completed")
	// There was an error with the container.
	// Stability: development
	K8SContainerStatusReasonError = K8SContainerStatusReasonKey.String("Error")
	// The container cannot run.
	// Stability: development
	K8SContainerStatusReasonContainerCannotRun = K8SContainerStatusReasonKey.String("ContainerCannotRun")
)

// Enum values for k8s.container.status.state
var (
	// The container has terminated.
	// Stability: development
	K8SContainerStatusStateTerminated = K8SContainerStatusStateKey.String("terminated")
	// The container is running.
	// Stability: development
	K8SContainerStatusStateRunning = K8SContainerStatusStateKey.String("running")
	// The container is waiting.
	// Stability: development
	K8SContainerStatusStateWaiting = K8SContainerStatusStateKey.String("waiting")
)

// Enum values for k8s.namespace.phase
var (
	// Active namespace phase as described by [K8s API]
	// Stability: development
	//
	// [K8s API]: https://pkg.go.dev/k8s.io/api@v0.31.3/core/v1#NamespacePhase
	K8SNamespacePhaseActive = K8SNamespacePhaseKey.String("active")
	// Terminating namespace phase as described by [K8s API]
	// Stability: development
	//
	// [K8s API]: https://pkg.go.dev/k8s.io/api@v0.31.3/core/v1#NamespacePhase
	K8SNamespacePhaseTerminating = K8SNamespacePhaseKey.String("terminating")
)

// Enum values for k8s.node.condition.status
var (
	// condition_true
	// Stability: development
	K8SNodeConditionStatusConditionTrue = K8SNodeConditionStatusKey.String("true")
	// condition_false
	// Stability: development
	K8SNodeConditionStatusConditionFalse = K8SNodeConditionStatusKey.String("false")
	// condition_unknown
	// Stability: development
	K8SNodeConditionStatusConditionUnknown = K8SNodeConditionStatusKey.String("unknown")
)

// Enum values for k8s.node.condition.type
var (
	// The node is healthy and ready to accept pods
	// Stability: development
	K8SNodeConditionTypeReady = K8SNodeConditionTypeKey.String("Ready")
	// Pressure exists on the disk size—that is, if the disk capacity is low
	// Stability: development
	K8SNodeConditionTypeDiskPressure = K8SNodeConditionTypeKey.String("DiskPressure")
	// Pressure exists on the node memory—that is, if the node memory is low
	// Stability: development
	K8SNodeConditionTypeMemoryPressure = K8SNodeConditionTypeKey.String("MemoryPressure")
	// Pressure exists on the processes—that is, if there are too many processes
	// on the node
	// Stability: development
	K8SNodeConditionTypePIDPressure = K8SNodeConditionTypeKey.String("PIDPressure")
	// The network for the node is not correctly configured
	// Stability: development
	K8SNodeConditionTypeNetworkUnavailable = K8SNodeConditionTypeKey.String("NetworkUnavailable")
)

// Enum values for k8s.volume.type
var (
	// A [persistentVolumeClaim] volume
	// Stability: development
	//
	// [persistentVolumeClaim]: https://v1-30.docs.kubernetes.io/docs/concepts/storage/volumes/#persistentvolumeclaim
	K8SVolumeTypePersistentVolumeClaim = K8SVolumeTypeKey.String("persistentVolumeClaim")
	// A [configMap] volume
	// Stability: development
	//
	// [configMap]: https://v1-30.docs.kubernetes.io/docs/concepts/storage/volumes/#configmap
	K8SVolumeTypeConfigMap = K8SVolumeTypeKey.String("configMap")
	// A [downwardAPI] volume
	// Stability: development
	//
	// [downwardAPI]: https://v1-30.docs.kubernetes.io/docs/concepts/storage/volumes/#downwardapi
	K8SVolumeTypeDownwardAPI = K8SVolumeTypeKey.String("downwardAPI")
	// An [emptyDir] volume
	// Stability: development
	//
	// [emptyDir]: https://v1-30.docs.kubernetes.io/docs/concepts/storage/volumes/#emptydir
	K8SVolumeTypeEmptyDir = K8SVolumeTypeKey.String("emptyDir")
	// A [secret] volume
	// Stability: development
	//
	// [secret]: https://v1-30.docs.kubernetes.io/docs/concepts/storage/volumes/#secret
	K8SVolumeTypeSecret = K8SVolumeTypeKey.String("secret")
	// A [local] volume
	// Stability: development
	//
	// [local]: https://v1-30.docs.kubernetes.io/docs/concepts/storage/volumes/#local
	K8SVolumeTypeLocal = K8SVolumeTypeKey.String("local")
)

// Namespace: linux
const (
	// LinuxMemorySlabStateKey is the attribute Key conforming to the
	// "linux.memory.slab.state" semantic conventions. It represents the Linux Slab
	// memory state.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "reclaimable", "unreclaimable"
	LinuxMemorySlabStateKey = attribute.Key("linux.memory.slab.state")
)

// Enum values for linux.memory.slab.state
var (
	// reclaimable
	// Stability: development
	LinuxMemorySlabStateReclaimable = LinuxMemorySlabStateKey.String("reclaimable")
	// unreclaimable
	// Stability: development
	LinuxMemorySlabStateUnreclaimable = LinuxMemorySlabStateKey.String("unreclaimable")
)

// Namespace: log
const (
	// LogFileNameKey is the attribute Key conforming to the "log.file.name"
	// semantic conventions. It represents the basename of the file.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "audit.log"
	LogFileNameKey = attribute.Key("log.file.name")

	// LogFileNameResolvedKey is the attribute Key conforming to the
	// "log.file.name_resolved" semantic conventions. It represents the basename of
	// the file, with symlinks resolved.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "uuid.log"
	LogFileNameResolvedKey = attribute.Key("log.file.name_resolved")

	// LogFilePathKey is the attribute Key conforming to the "log.file.path"
	// semantic conventions. It represents the full path to the file.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "/var/log/mysql/audit.log"
	LogFilePathKey = attribute.Key("log.file.path")

	// LogFilePathResolvedKey is the attribute Key conforming to the
	// "log.file.path_resolved" semantic conventions. It represents the full path to
	// the file, with symlinks resolved.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "/var/lib/docker/uuid.log"
	LogFilePathResolvedKey = attribute.Key("log.file.path_resolved")

	// LogIostreamKey is the attribute Key conforming to the "log.iostream" semantic
	// conventions. It represents the stream associated with the log. See below for
	// a list of well-known values.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	LogIostreamKey = attribute.Key("log.iostream")

	// LogRecordOriginalKey is the attribute Key conforming to the
	// "log.record.original" semantic conventions. It represents the complete
	// original Log Record.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "77 <86>1 2015-08-06T21:58:59.694Z 192.168.2.133 inactive - - -
	// Something happened", "[INFO] 8/3/24 12:34:56 Something happened"
	// Note: This value MAY be added when processing a Log Record which was
	// originally transmitted as a string or equivalent data type AND the Body field
	// of the Log Record does not contain the same value. (e.g. a syslog or a log
	// record read from a file.)
	LogRecordOriginalKey = attribute.Key("log.record.original")

	// LogRecordUIDKey is the attribute Key conforming to the "log.record.uid"
	// semantic conventions. It represents a unique identifier for the Log Record.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "01ARZ3NDEKTSV4RRFFQ69G5FAV"
	// Note: If an id is provided, other log records with the same id will be
	// considered duplicates and can be removed safely. This means, that two
	// distinguishable log records MUST have different values.
	// The id MAY be an
	// [Universally Unique Lexicographically Sortable Identifier (ULID)], but other
	// identifiers (e.g. UUID) may be used as needed.
	//
	// [Universally Unique Lexicographically Sortable Identifier (ULID)]: https://github.com/ulid/spec
	LogRecordUIDKey = attribute.Key("log.record.uid")
)

// LogFileName returns an attribute KeyValue conforming to the "log.file.name"
// semantic conventions. It represents the basename of the file.
func LogFileName(val string) attribute.KeyValue {
	return LogFileNameKey.String(val)
}

// LogFileNameResolved returns an attribute KeyValue conforming to the
// "log.file.name_resolved" semantic conventions. It represents the basename of
// the file, with symlinks resolved.
func LogFileNameResolved(val string) attribute.KeyValue {
	return LogFileNameResolvedKey.String(val)
}

// LogFilePath returns an attribute KeyValue conforming to the "log.file.path"
// semantic conventions. It represents the full path to the file.
func LogFilePath(val string) attribute.KeyValue {
	return LogFilePathKey.String(val)
}

// LogFilePathResolved returns an attribute KeyValue conforming to the
// "log.file.path_resolved" semantic conventions. It represents the full path to
// the file, with symlinks resolved.
func LogFilePathResolved(val string) attribute.KeyValue {
	return LogFilePathResolvedKey.String(val)
}

// LogRecordOriginal returns an attribute KeyValue conforming to the
// "log.record.original" semantic conventions. It represents the complete
// original Log Record.
func LogRecordOriginal(val string) attribute.KeyValue {
	return LogRecordOriginalKey.String(val)
}

// LogRecordUID returns an attribute KeyValue conforming to the "log.record.uid"
// semantic conventions. It represents a unique identifier for the Log Record.
func LogRecordUID(val string) attribute.KeyValue {
	return LogRecordUIDKey.String(val)
}

// Enum values for log.iostream
var (
	// Logs from stdout stream
	// Stability: development
	LogIostreamStdout = LogIostreamKey.String("stdout")
	// Events from stderr stream
	// Stability: development
	LogIostreamStderr = LogIostreamKey.String("stderr")
)

// Namespace: mainframe
const (
	// MainframeLparNameKey is the attribute Key conforming to the
	// "mainframe.lpar.name" semantic conventions. It represents the name of the
	// logical partition that hosts a systems with a mainframe operating system.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "LPAR01"
	MainframeLparNameKey = attribute.Key("mainframe.lpar.name")
)

// MainframeLparName returns an attribute KeyValue conforming to the
// "mainframe.lpar.name" semantic conventions. It represents the name of the
// logical partition that hosts a systems with a mainframe operating system.
func MainframeLparName(val string) attribute.KeyValue {
	return MainframeLparNameKey.String(val)
}

// Namespace: messaging
const (
	// MessagingBatchMessageCountKey is the attribute Key conforming to the
	// "messaging.batch.message_count" semantic conventions. It represents the
	// number of messages sent, received, or processed in the scope of the batching
	// operation.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 0, 1, 2
	// Note: Instrumentations SHOULD NOT set `messaging.batch.message_count` on
	// spans that operate with a single message. When a messaging client library
	// supports both batch and single-message API for the same operation,
	// instrumentations SHOULD use `messaging.batch.message_count` for batching APIs
	// and SHOULD NOT use it for single-message APIs.
	MessagingBatchMessageCountKey = attribute.Key("messaging.batch.message_count")

	// MessagingClientIDKey is the attribute Key conforming to the
	// "messaging.client.id" semantic conventions. It represents a unique identifier
	// for the client that consumes or produces a message.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "client-5", "myhost@8742@s8083jm"
	MessagingClientIDKey = attribute.Key("messaging.client.id")

	// MessagingConsumerGroupNameKey is the attribute Key conforming to the
	// "messaging.consumer.group.name" semantic conventions. It represents the name
	// of the consumer group with which a consumer is associated.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "my-group", "indexer"
	// Note: Semantic conventions for individual messaging systems SHOULD document
	// whether `messaging.consumer.group.name` is applicable and what it means in
	// the context of that system.
	MessagingConsumerGroupNameKey = attribute.Key("messaging.consumer.group.name")

	// MessagingDestinationAnonymousKey is the attribute Key conforming to the
	// "messaging.destination.anonymous" semantic conventions. It represents a
	// boolean that is true if the message destination is anonymous (could be
	// unnamed or have auto-generated name).
	//
	// Type: boolean
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	MessagingDestinationAnonymousKey = attribute.Key("messaging.destination.anonymous")

	// MessagingDestinationNameKey is the attribute Key conforming to the
	// "messaging.destination.name" semantic conventions. It represents the message
	// destination name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "MyQueue", "MyTopic"
	// Note: Destination name SHOULD uniquely identify a specific queue, topic or
	// other entity within the broker. If
	// the broker doesn't have such notion, the destination name SHOULD uniquely
	// identify the broker.
	MessagingDestinationNameKey = attribute.Key("messaging.destination.name")

	// MessagingDestinationPartitionIDKey is the attribute Key conforming to the
	// "messaging.destination.partition.id" semantic conventions. It represents the
	// identifier of the partition messages are sent to or received from, unique
	// within the `messaging.destination.name`.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1
	MessagingDestinationPartitionIDKey = attribute.Key("messaging.destination.partition.id")

	// MessagingDestinationSubscriptionNameKey is the attribute Key conforming to
	// the "messaging.destination.subscription.name" semantic conventions. It
	// represents the name of the destination subscription from which a message is
	// consumed.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "subscription-a"
	// Note: Semantic conventions for individual messaging systems SHOULD document
	// whether `messaging.destination.subscription.name` is applicable and what it
	// means in the context of that system.
	MessagingDestinationSubscriptionNameKey = attribute.Key("messaging.destination.subscription.name")

	// MessagingDestinationTemplateKey is the attribute Key conforming to the
	// "messaging.destination.template" semantic conventions. It represents the low
	// cardinality representation of the messaging destination name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "/customers/{customerId}"
	// Note: Destination names could be constructed from templates. An example would
	// be a destination name involving a user name or product id. Although the
	// destination name in this case is of high cardinality, the underlying template
	// is of low cardinality and can be effectively used for grouping and
	// aggregation.
	MessagingDestinationTemplateKey = attribute.Key("messaging.destination.template")

	// MessagingDestinationTemporaryKey is the attribute Key conforming to the
	// "messaging.destination.temporary" semantic conventions. It represents a
	// boolean that is true if the message destination is temporary and might not
	// exist anymore after messages are processed.
	//
	// Type: boolean
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	MessagingDestinationTemporaryKey = attribute.Key("messaging.destination.temporary")

	// MessagingEventHubsMessageEnqueuedTimeKey is the attribute Key conforming to
	// the "messaging.eventhubs.message.enqueued_time" semantic conventions. It
	// represents the UTC epoch seconds at which the message has been accepted and
	// stored in the entity.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	MessagingEventHubsMessageEnqueuedTimeKey = attribute.Key("messaging.eventhubs.message.enqueued_time")

	// MessagingGCPPubSubMessageAckDeadlineKey is the attribute Key conforming to
	// the "messaging.gcp_pubsub.message.ack_deadline" semantic conventions. It
	// represents the ack deadline in seconds set for the modify ack deadline
	// request.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	MessagingGCPPubSubMessageAckDeadlineKey = attribute.Key("messaging.gcp_pubsub.message.ack_deadline")

	// MessagingGCPPubSubMessageAckIDKey is the attribute Key conforming to the
	// "messaging.gcp_pubsub.message.ack_id" semantic conventions. It represents the
	// ack id for a given message.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: ack_id
	MessagingGCPPubSubMessageAckIDKey = attribute.Key("messaging.gcp_pubsub.message.ack_id")

	// MessagingGCPPubSubMessageDeliveryAttemptKey is the attribute Key conforming
	// to the "messaging.gcp_pubsub.message.delivery_attempt" semantic conventions.
	// It represents the delivery attempt for a given message.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	MessagingGCPPubSubMessageDeliveryAttemptKey = attribute.Key("messaging.gcp_pubsub.message.delivery_attempt")

	// MessagingGCPPubSubMessageOrderingKeyKey is the attribute Key conforming to
	// the "messaging.gcp_pubsub.message.ordering_key" semantic conventions. It
	// represents the ordering key for a given message. If the attribute is not
	// present, the message does not have an ordering key.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: ordering_key
	MessagingGCPPubSubMessageOrderingKeyKey = attribute.Key("messaging.gcp_pubsub.message.ordering_key")

	// MessagingKafkaMessageKeyKey is the attribute Key conforming to the
	// "messaging.kafka.message.key" semantic conventions. It represents the message
	// keys in Kafka are used for grouping alike messages to ensure they're
	// processed on the same partition. They differ from `messaging.message.id` in
	// that they're not unique. If the key is `null`, the attribute MUST NOT be set.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: myKey
	// Note: If the key type is not string, it's string representation has to be
	// supplied for the attribute. If the key has no unambiguous, canonical string
	// form, don't include its value.
	MessagingKafkaMessageKeyKey = attribute.Key("messaging.kafka.message.key")

	// MessagingKafkaMessageTombstoneKey is the attribute Key conforming to the
	// "messaging.kafka.message.tombstone" semantic conventions. It represents a
	// boolean that is true if the message is a tombstone.
	//
	// Type: boolean
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	MessagingKafkaMessageTombstoneKey = attribute.Key("messaging.kafka.message.tombstone")

	// MessagingKafkaOffsetKey is the attribute Key conforming to the
	// "messaging.kafka.offset" semantic conventions. It represents the offset of a
	// record in the corresponding Kafka partition.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	MessagingKafkaOffsetKey = attribute.Key("messaging.kafka.offset")

	// MessagingMessageBodySizeKey is the attribute Key conforming to the
	// "messaging.message.body.size" semantic conventions. It represents the size of
	// the message body in bytes.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Note: This can refer to both the compressed or uncompressed body size. If
	// both sizes are known, the uncompressed
	// body size should be used.
	MessagingMessageBodySizeKey = attribute.Key("messaging.message.body.size")

	// MessagingMessageConversationIDKey is the attribute Key conforming to the
	// "messaging.message.conversation_id" semantic conventions. It represents the
	// conversation ID identifying the conversation to which the message belongs,
	// represented as a string. Sometimes called "Correlation ID".
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: MyConversationId
	MessagingMessageConversationIDKey = attribute.Key("messaging.message.conversation_id")

	// MessagingMessageEnvelopeSizeKey is the attribute Key conforming to the
	// "messaging.message.envelope.size" semantic conventions. It represents the
	// size of the message body and metadata in bytes.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Note: This can refer to both the compressed or uncompressed size. If both
	// sizes are known, the uncompressed
	// size should be used.
	MessagingMessageEnvelopeSizeKey = attribute.Key("messaging.message.envelope.size")

	// MessagingMessageIDKey is the attribute Key conforming to the
	// "messaging.message.id" semantic conventions. It represents a value used by
	// the messaging system as an identifier for the message, represented as a
	// string.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 452a7c7c7c7048c2f887f61572b18fc2
	MessagingMessageIDKey = attribute.Key("messaging.message.id")

	// MessagingOperationNameKey is the attribute Key conforming to the
	// "messaging.operation.name" semantic conventions. It represents the
	// system-specific name of the messaging operation.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "ack", "nack", "send"
	MessagingOperationNameKey = attribute.Key("messaging.operation.name")

	// MessagingOperationTypeKey is the attribute Key conforming to the
	// "messaging.operation.type" semantic conventions. It represents a string
	// identifying the type of the messaging operation.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: If a custom value is used, it MUST be of low cardinality.
	MessagingOperationTypeKey = attribute.Key("messaging.operation.type")

	// MessagingRabbitMQDestinationRoutingKeyKey is the attribute Key conforming to
	// the "messaging.rabbitmq.destination.routing_key" semantic conventions. It
	// represents the rabbitMQ message routing key.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: myKey
	MessagingRabbitMQDestinationRoutingKeyKey = attribute.Key("messaging.rabbitmq.destination.routing_key")

	// MessagingRabbitMQMessageDeliveryTagKey is the attribute Key conforming to the
	// "messaging.rabbitmq.message.delivery_tag" semantic conventions. It represents
	// the rabbitMQ message delivery tag.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	MessagingRabbitMQMessageDeliveryTagKey = attribute.Key("messaging.rabbitmq.message.delivery_tag")

	// MessagingRocketMQConsumptionModelKey is the attribute Key conforming to the
	// "messaging.rocketmq.consumption_model" semantic conventions. It represents
	// the model of message consumption. This only applies to consumer spans.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	MessagingRocketMQConsumptionModelKey = attribute.Key("messaging.rocketmq.consumption_model")

	// MessagingRocketMQMessageDelayTimeLevelKey is the attribute Key conforming to
	// the "messaging.rocketmq.message.delay_time_level" semantic conventions. It
	// represents the delay time level for delay message, which determines the
	// message delay time.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	MessagingRocketMQMessageDelayTimeLevelKey = attribute.Key("messaging.rocketmq.message.delay_time_level")

	// MessagingRocketMQMessageDeliveryTimestampKey is the attribute Key conforming
	// to the "messaging.rocketmq.message.delivery_timestamp" semantic conventions.
	// It represents the timestamp in milliseconds that the delay message is
	// expected to be delivered to consumer.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	MessagingRocketMQMessageDeliveryTimestampKey = attribute.Key("messaging.rocketmq.message.delivery_timestamp")

	// MessagingRocketMQMessageGroupKey is the attribute Key conforming to the
	// "messaging.rocketmq.message.group" semantic conventions. It represents the it
	// is essential for FIFO message. Messages that belong to the same message group
	// are always processed one by one within the same consumer group.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: myMessageGroup
	MessagingRocketMQMessageGroupKey = attribute.Key("messaging.rocketmq.message.group")

	// MessagingRocketMQMessageKeysKey is the attribute Key conforming to the
	// "messaging.rocketmq.message.keys" semantic conventions. It represents the
	// key(s) of message, another way to mark message besides message id.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "keyA", "keyB"
	MessagingRocketMQMessageKeysKey = attribute.Key("messaging.rocketmq.message.keys")

	// MessagingRocketMQMessageTagKey is the attribute Key conforming to the
	// "messaging.rocketmq.message.tag" semantic conventions. It represents the
	// secondary classifier of message besides topic.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: tagA
	MessagingRocketMQMessageTagKey = attribute.Key("messaging.rocketmq.message.tag")

	// MessagingRocketMQMessageTypeKey is the attribute Key conforming to the
	// "messaging.rocketmq.message.type" semantic conventions. It represents the
	// type of message.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	MessagingRocketMQMessageTypeKey = attribute.Key("messaging.rocketmq.message.type")

	// MessagingRocketMQNamespaceKey is the attribute Key conforming to the
	// "messaging.rocketmq.namespace" semantic conventions. It represents the
	// namespace of RocketMQ resources, resources in different namespaces are
	// individual.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: myNamespace
	MessagingRocketMQNamespaceKey = attribute.Key("messaging.rocketmq.namespace")

	// MessagingServiceBusDispositionStatusKey is the attribute Key conforming to
	// the "messaging.servicebus.disposition_status" semantic conventions. It
	// represents the describes the [settlement type].
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	//
	// [settlement type]: https://learn.microsoft.com/azure/service-bus-messaging/message-transfers-locks-settlement#peeklock
	MessagingServiceBusDispositionStatusKey = attribute.Key("messaging.servicebus.disposition_status")

	// MessagingServiceBusMessageDeliveryCountKey is the attribute Key conforming to
	// the "messaging.servicebus.message.delivery_count" semantic conventions. It
	// represents the number of deliveries that have been attempted for this
	// message.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	MessagingServiceBusMessageDeliveryCountKey = attribute.Key("messaging.servicebus.message.delivery_count")

	// MessagingServiceBusMessageEnqueuedTimeKey is the attribute Key conforming to
	// the "messaging.servicebus.message.enqueued_time" semantic conventions. It
	// represents the UTC epoch seconds at which the message has been accepted and
	// stored in the entity.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	MessagingServiceBusMessageEnqueuedTimeKey = attribute.Key("messaging.servicebus.message.enqueued_time")

	// MessagingSystemKey is the attribute Key conforming to the "messaging.system"
	// semantic conventions. It represents the messaging system as identified by the
	// client instrumentation.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: The actual messaging system may differ from the one known by the
	// client. For example, when using Kafka client libraries to communicate with
	// Azure Event Hubs, the `messaging.system` is set to `kafka` based on the
	// instrumentation's best knowledge.
	MessagingSystemKey = attribute.Key("messaging.system")
)

// MessagingBatchMessageCount returns an attribute KeyValue conforming to the
// "messaging.batch.message_count" semantic conventions. It represents the number
// of messages sent, received, or processed in the scope of the batching
// operation.
func MessagingBatchMessageCount(val int) attribute.KeyValue {
	return MessagingBatchMessageCountKey.Int(val)
}

// MessagingClientID returns an attribute KeyValue conforming to the
// "messaging.client.id" semantic conventions. It represents a unique identifier
// for the client that consumes or produces a message.
func MessagingClientID(val string) attribute.KeyValue {
	return MessagingClientIDKey.String(val)
}

// MessagingConsumerGroupName returns an attribute KeyValue conforming to the
// "messaging.consumer.group.name" semantic conventions. It represents the name
// of the consumer group with which a consumer is associated.
func MessagingConsumerGroupName(val string) attribute.KeyValue {
	return MessagingConsumerGroupNameKey.String(val)
}

// MessagingDestinationAnonymous returns an attribute KeyValue conforming to the
// "messaging.destination.anonymous" semantic conventions. It represents a
// boolean that is true if the message destination is anonymous (could be unnamed
// or have auto-generated name).
func MessagingDestinationAnonymous(val bool) attribute.KeyValue {
	return MessagingDestinationAnonymousKey.Bool(val)
}

// MessagingDestinationName returns an attribute KeyValue conforming to the
// "messaging.destination.name" semantic conventions. It represents the message
// destination name.
func MessagingDestinationName(val string) attribute.KeyValue {
	return MessagingDestinationNameKey.String(val)
}

// MessagingDestinationPartitionID returns an attribute KeyValue conforming to
// the "messaging.destination.partition.id" semantic conventions. It represents
// the identifier of the partition messages are sent to or received from, unique
// within the `messaging.destination.name`.
func MessagingDestinationPartitionID(val string) attribute.KeyValue {
	return MessagingDestinationPartitionIDKey.String(val)
}

// MessagingDestinationSubscriptionName returns an attribute KeyValue conforming
// to the "messaging.destination.subscription.name" semantic conventions. It
// represents the name of the destination subscription from which a message is
// consumed.
func MessagingDestinationSubscriptionName(val string) attribute.KeyValue {
	return MessagingDestinationSubscriptionNameKey.String(val)
}

// MessagingDestinationTemplate returns an attribute KeyValue conforming to the
// "messaging.destination.template" semantic conventions. It represents the low
// cardinality representation of the messaging destination name.
func MessagingDestinationTemplate(val string) attribute.KeyValue {
	return MessagingDestinationTemplateKey.String(val)
}

// MessagingDestinationTemporary returns an attribute KeyValue conforming to the
// "messaging.destination.temporary" semantic conventions. It represents a
// boolean that is true if the message destination is temporary and might not
// exist anymore after messages are processed.
func MessagingDestinationTemporary(val bool) attribute.KeyValue {
	return MessagingDestinationTemporaryKey.Bool(val)
}

// MessagingEventHubsMessageEnqueuedTime returns an attribute KeyValue conforming
// to the "messaging.eventhubs.message.enqueued_time" semantic conventions. It
// represents the UTC epoch seconds at which the message has been accepted and
// stored in the entity.
func MessagingEventHubsMessageEnqueuedTime(val int) attribute.KeyValue {
	return MessagingEventHubsMessageEnqueuedTimeKey.Int(val)
}

// MessagingGCPPubSubMessageAckDeadline returns an attribute KeyValue conforming
// to the "messaging.gcp_pubsub.message.ack_deadline" semantic conventions. It
// represents the ack deadline in seconds set for the modify ack deadline
// request.
func MessagingGCPPubSubMessageAckDeadline(val int) attribute.KeyValue {
	return MessagingGCPPubSubMessageAckDeadlineKey.Int(val)
}

// MessagingGCPPubSubMessageAckID returns an attribute KeyValue conforming to the
// "messaging.gcp_pubsub.message.ack_id" semantic conventions. It represents the
// ack id for a given message.
func MessagingGCPPubSubMessageAckID(val string) attribute.KeyValue {
	return MessagingGCPPubSubMessageAckIDKey.String(val)
}

// MessagingGCPPubSubMessageDeliveryAttempt returns an attribute KeyValue
// conforming to the "messaging.gcp_pubsub.message.delivery_attempt" semantic
// conventions. It represents the delivery attempt for a given message.
func MessagingGCPPubSubMessageDeliveryAttempt(val int) attribute.KeyValue {
	return MessagingGCPPubSubMessageDeliveryAttemptKey.Int(val)
}

// MessagingGCPPubSubMessageOrderingKey returns an attribute KeyValue conforming
// to the "messaging.gcp_pubsub.message.ordering_key" semantic conventions. It
// represents the ordering key for a given message. If the attribute is not
// present, the message does not have an ordering key.
func MessagingGCPPubSubMessageOrderingKey(val string) attribute.KeyValue {
	return MessagingGCPPubSubMessageOrderingKeyKey.String(val)
}

// MessagingKafkaMessageKey returns an attribute KeyValue conforming to the
// "messaging.kafka.message.key" semantic conventions. It represents the message
// keys in Kafka are used for grouping alike messages to ensure they're processed
// on the same partition. They differ from `messaging.message.id` in that they're
// not unique. If the key is `null`, the attribute MUST NOT be set.
func MessagingKafkaMessageKey(val string) attribute.KeyValue {
	return MessagingKafkaMessageKeyKey.String(val)
}

// MessagingKafkaMessageTombstone returns an attribute KeyValue conforming to the
// "messaging.kafka.message.tombstone" semantic conventions. It represents a
// boolean that is true if the message is a tombstone.
func MessagingKafkaMessageTombstone(val bool) attribute.KeyValue {
	return MessagingKafkaMessageTombstoneKey.Bool(val)
}

// MessagingKafkaOffset returns an attribute KeyValue conforming to the
// "messaging.kafka.offset" semantic conventions. It represents the offset of a
// record in the corresponding Kafka partition.
func MessagingKafkaOffset(val int) attribute.KeyValue {
	return MessagingKafkaOffsetKey.Int(val)
}

// MessagingMessageBodySize returns an attribute KeyValue conforming to the
// "messaging.message.body.size" semantic conventions. It represents the size of
// the message body in bytes.
func MessagingMessageBodySize(val int) attribute.KeyValue {
	return MessagingMessageBodySizeKey.Int(val)
}

// MessagingMessageConversationID returns an attribute KeyValue conforming to the
// "messaging.message.conversation_id" semantic conventions. It represents the
// conversation ID identifying the conversation to which the message belongs,
// represented as a string. Sometimes called "Correlation ID".
func MessagingMessageConversationID(val string) attribute.KeyValue {
	return MessagingMessageConversationIDKey.String(val)
}

// MessagingMessageEnvelopeSize returns an attribute KeyValue conforming to the
// "messaging.message.envelope.size" semantic conventions. It represents the size
// of the message body and metadata in bytes.
func MessagingMessageEnvelopeSize(val int) attribute.KeyValue {
	return MessagingMessageEnvelopeSizeKey.Int(val)
}

// MessagingMessageID returns an attribute KeyValue conforming to the
// "messaging.message.id" semantic conventions. It represents a value used by the
// messaging system as an identifier for the message, represented as a string.
func MessagingMessageID(val string) attribute.KeyValue {
	return MessagingMessageIDKey.String(val)
}

// MessagingOperationName returns an attribute KeyValue conforming to the
// "messaging.operation.name" semantic conventions. It represents the
// system-specific name of the messaging operation.
func MessagingOperationName(val string) attribute.KeyValue {
	return MessagingOperationNameKey.String(val)
}

// MessagingRabbitMQDestinationRoutingKey returns an attribute KeyValue
// conforming to the "messaging.rabbitmq.destination.routing_key" semantic
// conventions. It represents the rabbitMQ message routing key.
func MessagingRabbitMQDestinationRoutingKey(val string) attribute.KeyValue {
	return MessagingRabbitMQDestinationRoutingKeyKey.String(val)
}

// MessagingRabbitMQMessageDeliveryTag returns an attribute KeyValue conforming
// to the "messaging.rabbitmq.message.delivery_tag" semantic conventions. It
// represents the rabbitMQ message delivery tag.
func MessagingRabbitMQMessageDeliveryTag(val int) attribute.KeyValue {
	return MessagingRabbitMQMessageDeliveryTagKey.Int(val)
}

// MessagingRocketMQMessageDelayTimeLevel returns an attribute KeyValue
// conforming to the "messaging.rocketmq.message.delay_time_level" semantic
// conventions. It represents the delay time level for delay message, which
// determines the message delay time.
func MessagingRocketMQMessageDelayTimeLevel(val int) attribute.KeyValue {
	return MessagingRocketMQMessageDelayTimeLevelKey.Int(val)
}

// MessagingRocketMQMessageDeliveryTimestamp returns an attribute KeyValue
// conforming to the "messaging.rocketmq.message.delivery_timestamp" semantic
// conventions. It represents the timestamp in milliseconds that the delay
// message is expected to be delivered to consumer.
func MessagingRocketMQMessageDeliveryTimestamp(val int) attribute.KeyValue {
	return MessagingRocketMQMessageDeliveryTimestampKey.Int(val)
}

// MessagingRocketMQMessageGroup returns an attribute KeyValue conforming to the
// "messaging.rocketmq.message.group" semantic conventions. It represents the it
// is essential for FIFO message. Messages that belong to the same message group
// are always processed one by one within the same consumer group.
func MessagingRocketMQMessageGroup(val string) attribute.KeyValue {
	return MessagingRocketMQMessageGroupKey.String(val)
}

// MessagingRocketMQMessageKeys returns an attribute KeyValue conforming to the
// "messaging.rocketmq.message.keys" semantic conventions. It represents the
// key(s) of message, another way to mark message besides message id.
func MessagingRocketMQMessageKeys(val ...string) attribute.KeyValue {
	return MessagingRocketMQMessageKeysKey.StringSlice(val)
}

// MessagingRocketMQMessageTag returns an attribute KeyValue conforming to the
// "messaging.rocketmq.message.tag" semantic conventions. It represents the
// secondary classifier of message besides topic.
func MessagingRocketMQMessageTag(val string) attribute.KeyValue {
	return MessagingRocketMQMessageTagKey.String(val)
}

// MessagingRocketMQNamespace returns an attribute KeyValue conforming to the
// "messaging.rocketmq.namespace" semantic conventions. It represents the
// namespace of RocketMQ resources, resources in different namespaces are
// individual.
func MessagingRocketMQNamespace(val string) attribute.KeyValue {
	return MessagingRocketMQNamespaceKey.String(val)
}

// MessagingServiceBusMessageDeliveryCount returns an attribute KeyValue
// conforming to the "messaging.servicebus.message.delivery_count" semantic
// conventions. It represents the number of deliveries that have been attempted
// for this message.
func MessagingServiceBusMessageDeliveryCount(val int) attribute.KeyValue {
	return MessagingServiceBusMessageDeliveryCountKey.Int(val)
}

// MessagingServiceBusMessageEnqueuedTime returns an attribute KeyValue
// conforming to the "messaging.servicebus.message.enqueued_time" semantic
// conventions. It represents the UTC epoch seconds at which the message has been
// accepted and stored in the entity.
func MessagingServiceBusMessageEnqueuedTime(val int) attribute.KeyValue {
	return MessagingServiceBusMessageEnqueuedTimeKey.Int(val)
}

// Enum values for messaging.operation.type
var (
	// A message is created. "Create" spans always refer to a single message and are
	// used to provide a unique creation context for messages in batch sending
	// scenarios.
	//
	// Stability: development
	MessagingOperationTypeCreate = MessagingOperationTypeKey.String("create")
	// One or more messages are provided for sending to an intermediary. If a single
	// message is sent, the context of the "Send" span can be used as the creation
	// context and no "Create" span needs to be created.
	//
	// Stability: development
	MessagingOperationTypeSend = MessagingOperationTypeKey.String("send")
	// One or more messages are requested by a consumer. This operation refers to
	// pull-based scenarios, where consumers explicitly call methods of messaging
	// SDKs to receive messages.
	//
	// Stability: development
	MessagingOperationTypeReceive = MessagingOperationTypeKey.String("receive")
	// One or more messages are processed by a consumer.
	//
	// Stability: development
	MessagingOperationTypeProcess = MessagingOperationTypeKey.String("process")
	// One or more messages are settled.
	//
	// Stability: development
	MessagingOperationTypeSettle = MessagingOperationTypeKey.String("settle")
)

// Enum values for messaging.rocketmq.consumption_model
var (
	// Clustering consumption model
	// Stability: development
	MessagingRocketMQConsumptionModelClustering = MessagingRocketMQConsumptionModelKey.String("clustering")
	// Broadcasting consumption model
	// Stability: development
	MessagingRocketMQConsumptionModelBroadcasting = MessagingRocketMQConsumptionModelKey.String("broadcasting")
)

// Enum values for messaging.rocketmq.message.type
var (
	// Normal message
	// Stability: development
	MessagingRocketMQMessageTypeNormal = MessagingRocketMQMessageTypeKey.String("normal")
	// FIFO message
	// Stability: development
	MessagingRocketMQMessageTypeFifo = MessagingRocketMQMessageTypeKey.String("fifo")
	// Delay message
	// Stability: development
	MessagingRocketMQMessageTypeDelay = MessagingRocketMQMessageTypeKey.String("delay")
	// Transaction message
	// Stability: development
	MessagingRocketMQMessageTypeTransaction = MessagingRocketMQMessageTypeKey.String("transaction")
)

// Enum values for messaging.servicebus.disposition_status
var (
	// Message is completed
	// Stability: development
	MessagingServiceBusDispositionStatusComplete = MessagingServiceBusDispositionStatusKey.String("complete")
	// Message is abandoned
	// Stability: development
	MessagingServiceBusDispositionStatusAbandon = MessagingServiceBusDispositionStatusKey.String("abandon")
	// Message is sent to dead letter queue
	// Stability: development
	MessagingServiceBusDispositionStatusDeadLetter = MessagingServiceBusDispositionStatusKey.String("dead_letter")
	// Message is deferred
	// Stability: development
	MessagingServiceBusDispositionStatusDefer = MessagingServiceBusDispositionStatusKey.String("defer")
)

// Enum values for messaging.system
var (
	// Apache ActiveMQ
	// Stability: development
	MessagingSystemActiveMQ = MessagingSystemKey.String("activemq")
	// Amazon Simple Notification Service (SNS)
	// Stability: development
	MessagingSystemAWSSNS = MessagingSystemKey.String("aws.sns")
	// Amazon Simple Queue Service (SQS)
	// Stability: development
	MessagingSystemAWSSQS = MessagingSystemKey.String("aws_sqs")
	// Azure Event Grid
	// Stability: development
	MessagingSystemEventGrid = MessagingSystemKey.String("eventgrid")
	// Azure Event Hubs
	// Stability: development
	MessagingSystemEventHubs = MessagingSystemKey.String("eventhubs")
	// Azure Service Bus
	// Stability: development
	MessagingSystemServiceBus = MessagingSystemKey.String("servicebus")
	// Google Cloud Pub/Sub
	// Stability: development
	MessagingSystemGCPPubSub = MessagingSystemKey.String("gcp_pubsub")
	// Java Message Service
	// Stability: development
	MessagingSystemJMS = MessagingSystemKey.String("jms")
	// Apache Kafka
	// Stability: development
	MessagingSystemKafka = MessagingSystemKey.String("kafka")
	// RabbitMQ
	// Stability: development
	MessagingSystemRabbitMQ = MessagingSystemKey.String("rabbitmq")
	// Apache RocketMQ
	// Stability: development
	MessagingSystemRocketMQ = MessagingSystemKey.String("rocketmq")
	// Apache Pulsar
	// Stability: development
	MessagingSystemPulsar = MessagingSystemKey.String("pulsar")
)

// Namespace: network
const (
	// NetworkCarrierICCKey is the attribute Key conforming to the
	// "network.carrier.icc" semantic conventions. It represents the ISO 3166-1
	// alpha-2 2-character country code associated with the mobile carrier network.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: DE
	NetworkCarrierICCKey = attribute.Key("network.carrier.icc")

	// NetworkCarrierMCCKey is the attribute Key conforming to the
	// "network.carrier.mcc" semantic conventions. It represents the mobile carrier
	// country code.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 310
	NetworkCarrierMCCKey = attribute.Key("network.carrier.mcc")

	// NetworkCarrierMNCKey is the attribute Key conforming to the
	// "network.carrier.mnc" semantic conventions. It represents the mobile carrier
	// network code.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 001
	NetworkCarrierMNCKey = attribute.Key("network.carrier.mnc")

	// NetworkCarrierNameKey is the attribute Key conforming to the
	// "network.carrier.name" semantic conventions. It represents the name of the
	// mobile carrier.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: sprint
	NetworkCarrierNameKey = attribute.Key("network.carrier.name")

	// NetworkConnectionStateKey is the attribute Key conforming to the
	// "network.connection.state" semantic conventions. It represents the state of
	// network connection.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "close_wait"
	// Note: Connection states are defined as part of the [rfc9293]
	//
	// [rfc9293]: https://datatracker.ietf.org/doc/html/rfc9293#section-3.3.2
	NetworkConnectionStateKey = attribute.Key("network.connection.state")

	// NetworkConnectionSubtypeKey is the attribute Key conforming to the
	// "network.connection.subtype" semantic conventions. It represents the this
	// describes more details regarding the connection.type. It may be the type of
	// cell technology connection, but it could be used for describing details about
	// a wifi connection.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: LTE
	NetworkConnectionSubtypeKey = attribute.Key("network.connection.subtype")

	// NetworkConnectionTypeKey is the attribute Key conforming to the
	// "network.connection.type" semantic conventions. It represents the internet
	// connection type.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: wifi
	NetworkConnectionTypeKey = attribute.Key("network.connection.type")

	// NetworkInterfaceNameKey is the attribute Key conforming to the
	// "network.interface.name" semantic conventions. It represents the network
	// interface name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "lo", "eth0"
	NetworkInterfaceNameKey = attribute.Key("network.interface.name")

	// NetworkIODirectionKey is the attribute Key conforming to the
	// "network.io.direction" semantic conventions. It represents the network IO
	// operation direction.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "transmit"
	NetworkIODirectionKey = attribute.Key("network.io.direction")

	// NetworkLocalAddressKey is the attribute Key conforming to the
	// "network.local.address" semantic conventions. It represents the local address
	// of the network connection - IP address or Unix domain socket name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "10.1.2.80", "/tmp/my.sock"
	NetworkLocalAddressKey = attribute.Key("network.local.address")

	// NetworkLocalPortKey is the attribute Key conforming to the
	// "network.local.port" semantic conventions. It represents the local port
	// number of the network connection.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: 65123
	NetworkLocalPortKey = attribute.Key("network.local.port")

	// NetworkPeerAddressKey is the attribute Key conforming to the
	// "network.peer.address" semantic conventions. It represents the peer address
	// of the network connection - IP address or Unix domain socket name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "10.1.2.80", "/tmp/my.sock"
	NetworkPeerAddressKey = attribute.Key("network.peer.address")

	// NetworkPeerPortKey is the attribute Key conforming to the "network.peer.port"
	// semantic conventions. It represents the peer port number of the network
	// connection.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: 65123
	NetworkPeerPortKey = attribute.Key("network.peer.port")

	// NetworkProtocolNameKey is the attribute Key conforming to the
	// "network.protocol.name" semantic conventions. It represents the
	// [OSI application layer] or non-OSI equivalent.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "amqp", "http", "mqtt"
	// Note: The value SHOULD be normalized to lowercase.
	//
	// [OSI application layer]: https://wikipedia.org/wiki/Application_layer
	NetworkProtocolNameKey = attribute.Key("network.protocol.name")

	// NetworkProtocolVersionKey is the attribute Key conforming to the
	// "network.protocol.version" semantic conventions. It represents the actual
	// version of the protocol used for network communication.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "1.1", "2"
	// Note: If protocol version is subject to negotiation (for example using [ALPN]
	// ), this attribute SHOULD be set to the negotiated version. If the actual
	// protocol version is not known, this attribute SHOULD NOT be set.
	//
	// [ALPN]: https://www.rfc-editor.org/rfc/rfc7301.html
	NetworkProtocolVersionKey = attribute.Key("network.protocol.version")

	// NetworkTransportKey is the attribute Key conforming to the
	// "network.transport" semantic conventions. It represents the
	// [OSI transport layer] or [inter-process communication method].
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "tcp", "udp"
	// Note: The value SHOULD be normalized to lowercase.
	//
	// Consider always setting the transport when setting a port number, since
	// a port number is ambiguous without knowing the transport. For example
	// different processes could be listening on TCP port 12345 and UDP port 12345.
	//
	// [OSI transport layer]: https://wikipedia.org/wiki/Transport_layer
	// [inter-process communication method]: https://wikipedia.org/wiki/Inter-process_communication
	NetworkTransportKey = attribute.Key("network.transport")

	// NetworkTypeKey is the attribute Key conforming to the "network.type" semantic
	// conventions. It represents the [OSI network layer] or non-OSI equivalent.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "ipv4", "ipv6"
	// Note: The value SHOULD be normalized to lowercase.
	//
	// [OSI network layer]: https://wikipedia.org/wiki/Network_layer
	NetworkTypeKey = attribute.Key("network.type")
)

// NetworkCarrierICC returns an attribute KeyValue conforming to the
// "network.carrier.icc" semantic conventions. It represents the ISO 3166-1
// alpha-2 2-character country code associated with the mobile carrier network.
func NetworkCarrierICC(val string) attribute.KeyValue {
	return NetworkCarrierICCKey.String(val)
}

// NetworkCarrierMCC returns an attribute KeyValue conforming to the
// "network.carrier.mcc" semantic conventions. It represents the mobile carrier
// country code.
func NetworkCarrierMCC(val string) attribute.KeyValue {
	return NetworkCarrierMCCKey.String(val)
}

// NetworkCarrierMNC returns an attribute KeyValue conforming to the
// "network.carrier.mnc" semantic conventions. It represents the mobile carrier
// network code.
func NetworkCarrierMNC(val string) attribute.KeyValue {
	return NetworkCarrierMNCKey.String(val)
}

// NetworkCarrierName returns an attribute KeyValue conforming to the
// "network.carrier.name" semantic conventions. It represents the name of the
// mobile carrier.
func NetworkCarrierName(val string) attribute.KeyValue {
	return NetworkCarrierNameKey.String(val)
}

// NetworkInterfaceName returns an attribute KeyValue conforming to the
// "network.interface.name" semantic conventions. It represents the network
// interface name.
func NetworkInterfaceName(val string) attribute.KeyValue {
	return NetworkInterfaceNameKey.String(val)
}

// NetworkLocalAddress returns an attribute KeyValue conforming to the
// "network.local.address" semantic conventions. It represents the local address
// of the network connection - IP address or Unix domain socket name.
func NetworkLocalAddress(val string) attribute.KeyValue {
	return NetworkLocalAddressKey.String(val)
}

// NetworkLocalPort returns an attribute KeyValue conforming to the
// "network.local.port" semantic conventions. It represents the local port number
// of the network connection.
func NetworkLocalPort(val int) attribute.KeyValue {
	return NetworkLocalPortKey.Int(val)
}

// NetworkPeerAddress returns an attribute KeyValue conforming to the
// "network.peer.address" semantic conventions. It represents the peer address of
// the network connection - IP address or Unix domain socket name.
func NetworkPeerAddress(val string) attribute.KeyValue {
	return NetworkPeerAddressKey.String(val)
}

// NetworkPeerPort returns an attribute KeyValue conforming to the
// "network.peer.port" semantic conventions. It represents the peer port number
// of the network connection.
func NetworkPeerPort(val int) attribute.KeyValue {
	return NetworkPeerPortKey.Int(val)
}

// NetworkProtocolName returns an attribute KeyValue conforming to the
// "network.protocol.name" semantic conventions. It represents the
// [OSI application layer] or non-OSI equivalent.
//
// [OSI application layer]: https://wikipedia.org/wiki/Application_layer
func NetworkProtocolName(val string) attribute.KeyValue {
	return NetworkProtocolNameKey.String(val)
}

// NetworkProtocolVersion returns an attribute KeyValue conforming to the
// "network.protocol.version" semantic conventions. It represents the actual
// version of the protocol used for network communication.
func NetworkProtocolVersion(val string) attribute.KeyValue {
	return NetworkProtocolVersionKey.String(val)
}

// Enum values for network.connection.state
var (
	// closed
	// Stability: development
	NetworkConnectionStateClosed = NetworkConnectionStateKey.String("closed")
	// close_wait
	// Stability: development
	NetworkConnectionStateCloseWait = NetworkConnectionStateKey.String("close_wait")
	// closing
	// Stability: development
	NetworkConnectionStateClosing = NetworkConnectionStateKey.String("closing")
	// established
	// Stability: development
	NetworkConnectionStateEstablished = NetworkConnectionStateKey.String("established")
	// fin_wait_1
	// Stability: development
	NetworkConnectionStateFinWait1 = NetworkConnectionStateKey.String("fin_wait_1")
	// fin_wait_2
	// Stability: development
	NetworkConnectionStateFinWait2 = NetworkConnectionStateKey.String("fin_wait_2")
	// last_ack
	// Stability: development
	NetworkConnectionStateLastAck = NetworkConnectionStateKey.String("last_ack")
	// listen
	// Stability: development
	NetworkConnectionStateListen = NetworkConnectionStateKey.String("listen")
	// syn_received
	// Stability: development
	NetworkConnectionStateSynReceived = NetworkConnectionStateKey.String("syn_received")
	// syn_sent
	// Stability: development
	NetworkConnectionStateSynSent = NetworkConnectionStateKey.String("syn_sent")
	// time_wait
	// Stability: development
	NetworkConnectionStateTimeWait = NetworkConnectionStateKey.String("time_wait")
)

// Enum values for network.connection.subtype
var (
	// GPRS
	// Stability: development
	NetworkConnectionSubtypeGprs = NetworkConnectionSubtypeKey.String("gprs")
	// EDGE
	// Stability: development
	NetworkConnectionSubtypeEdge = NetworkConnectionSubtypeKey.String("edge")
	// UMTS
	// Stability: development
	NetworkConnectionSubtypeUmts = NetworkConnectionSubtypeKey.String("umts")
	// CDMA
	// Stability: development
	NetworkConnectionSubtypeCdma = NetworkConnectionSubtypeKey.String("cdma")
	// EVDO Rel. 0
	// Stability: development
	NetworkConnectionSubtypeEvdo0 = NetworkConnectionSubtypeKey.String("evdo_0")
	// EVDO Rev. A
	// Stability: development
	NetworkConnectionSubtypeEvdoA = NetworkConnectionSubtypeKey.String("evdo_a")
	// CDMA2000 1XRTT
	// Stability: development
	NetworkConnectionSubtypeCdma20001xrtt = NetworkConnectionSubtypeKey.String("cdma2000_1xrtt")
	// HSDPA
	// Stability: development
	NetworkConnectionSubtypeHsdpa = NetworkConnectionSubtypeKey.String("hsdpa")
	// HSUPA
	// Stability: development
	NetworkConnectionSubtypeHsupa = NetworkConnectionSubtypeKey.String("hsupa")
	// HSPA
	// Stability: development
	NetworkConnectionSubtypeHspa = NetworkConnectionSubtypeKey.String("hspa")
	// IDEN
	// Stability: development
	NetworkConnectionSubtypeIden = NetworkConnectionSubtypeKey.String("iden")
	// EVDO Rev. B
	// Stability: development
	NetworkConnectionSubtypeEvdoB = NetworkConnectionSubtypeKey.String("evdo_b")
	// LTE
	// Stability: development
	NetworkConnectionSubtypeLte = NetworkConnectionSubtypeKey.String("lte")
	// EHRPD
	// Stability: development
	NetworkConnectionSubtypeEhrpd = NetworkConnectionSubtypeKey.String("ehrpd")
	// HSPAP
	// Stability: development
	NetworkConnectionSubtypeHspap = NetworkConnectionSubtypeKey.String("hspap")
	// GSM
	// Stability: development
	NetworkConnectionSubtypeGsm = NetworkConnectionSubtypeKey.String("gsm")
	// TD-SCDMA
	// Stability: development
	NetworkConnectionSubtypeTdScdma = NetworkConnectionSubtypeKey.String("td_scdma")
	// IWLAN
	// Stability: development
	NetworkConnectionSubtypeIwlan = NetworkConnectionSubtypeKey.String("iwlan")
	// 5G NR (New Radio)
	// Stability: development
	NetworkConnectionSubtypeNr = NetworkConnectionSubtypeKey.String("nr")
	// 5G NRNSA (New Radio Non-Standalone)
	// Stability: development
	NetworkConnectionSubtypeNrnsa = NetworkConnectionSubtypeKey.String("nrnsa")
	// LTE CA
	// Stability: development
	NetworkConnectionSubtypeLteCa = NetworkConnectionSubtypeKey.String("lte_ca")
)

// Enum values for network.connection.type
var (
	// wifi
	// Stability: development
	NetworkConnectionTypeWifi = NetworkConnectionTypeKey.String("wifi")
	// wired
	// Stability: development
	NetworkConnectionTypeWired = NetworkConnectionTypeKey.String("wired")
	// cell
	// Stability: development
	NetworkConnectionTypeCell = NetworkConnectionTypeKey.String("cell")
	// unavailable
	// Stability: development
	NetworkConnectionTypeUnavailable = NetworkConnectionTypeKey.String("unavailable")
	// unknown
	// Stability: development
	NetworkConnectionTypeUnknown = NetworkConnectionTypeKey.String("unknown")
)

// Enum values for network.io.direction
var (
	// transmit
	// Stability: development
	NetworkIODirectionTransmit = NetworkIODirectionKey.String("transmit")
	// receive
	// Stability: development
	NetworkIODirectionReceive = NetworkIODirectionKey.String("receive")
)

// Enum values for network.transport
var (
	// TCP
	// Stability: stable
	NetworkTransportTCP = NetworkTransportKey.String("tcp")
	// UDP
	// Stability: stable
	NetworkTransportUDP = NetworkTransportKey.String("udp")
	// Named or anonymous pipe.
	// Stability: stable
	NetworkTransportPipe = NetworkTransportKey.String("pipe")
	// Unix domain socket
	// Stability: stable
	NetworkTransportUnix = NetworkTransportKey.String("unix")
	// QUIC
	// Stability: stable
	NetworkTransportQUIC = NetworkTransportKey.String("quic")
)

// Enum values for network.type
var (
	// IPv4
	// Stability: stable
	NetworkTypeIPv4 = NetworkTypeKey.String("ipv4")
	// IPv6
	// Stability: stable
	NetworkTypeIPv6 = NetworkTypeKey.String("ipv6")
)

// Namespace: oci
const (
	// OCIManifestDigestKey is the attribute Key conforming to the
	// "oci.manifest.digest" semantic conventions. It represents the digest of the
	// OCI image manifest. For container images specifically is the digest by which
	// the container image is known.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// "sha256:e4ca62c0d62f3e886e684806dfe9d4e0cda60d54986898173c1083856cfda0f4"
	// Note: Follows [OCI Image Manifest Specification], and specifically the
	// [Digest property].
	// An example can be found in [Example Image Manifest].
	//
	// [OCI Image Manifest Specification]: https://github.com/opencontainers/image-spec/blob/main/manifest.md
	// [Digest property]: https://github.com/opencontainers/image-spec/blob/main/descriptor.md#digests
	// [Example Image Manifest]: https://github.com/opencontainers/image-spec/blob/main/manifest.md#example-image-manifest
	OCIManifestDigestKey = attribute.Key("oci.manifest.digest")
)

// OCIManifestDigest returns an attribute KeyValue conforming to the
// "oci.manifest.digest" semantic conventions. It represents the digest of the
// OCI image manifest. For container images specifically is the digest by which
// the container image is known.
func OCIManifestDigest(val string) attribute.KeyValue {
	return OCIManifestDigestKey.String(val)
}

// Namespace: openai
const (
	// OpenAIRequestServiceTierKey is the attribute Key conforming to the
	// "openai.request.service_tier" semantic conventions. It represents the service
	// tier requested. May be a specific tier, default, or auto.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "auto", "default"
	OpenAIRequestServiceTierKey = attribute.Key("openai.request.service_tier")

	// OpenAIResponseServiceTierKey is the attribute Key conforming to the
	// "openai.response.service_tier" semantic conventions. It represents the
	// service tier used for the response.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "scale", "default"
	OpenAIResponseServiceTierKey = attribute.Key("openai.response.service_tier")

	// OpenAIResponseSystemFingerprintKey is the attribute Key conforming to the
	// "openai.response.system_fingerprint" semantic conventions. It represents a
	// fingerprint to track any eventual change in the Generative AI environment.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "fp_44709d6fcb"
	OpenAIResponseSystemFingerprintKey = attribute.Key("openai.response.system_fingerprint")
)

// OpenAIResponseServiceTier returns an attribute KeyValue conforming to the
// "openai.response.service_tier" semantic conventions. It represents the service
// tier used for the response.
func OpenAIResponseServiceTier(val string) attribute.KeyValue {
	return OpenAIResponseServiceTierKey.String(val)
}

// OpenAIResponseSystemFingerprint returns an attribute KeyValue conforming to
// the "openai.response.system_fingerprint" semantic conventions. It represents a
// fingerprint to track any eventual change in the Generative AI environment.
func OpenAIResponseSystemFingerprint(val string) attribute.KeyValue {
	return OpenAIResponseSystemFingerprintKey.String(val)
}

// Enum values for openai.request.service_tier
var (
	// The system will utilize scale tier credits until they are exhausted.
	// Stability: development
	OpenAIRequestServiceTierAuto = OpenAIRequestServiceTierKey.String("auto")
	// The system will utilize the default scale tier.
	// Stability: development
	OpenAIRequestServiceTierDefault = OpenAIRequestServiceTierKey.String("default")
)

// Namespace: opentracing
const (
	// OpenTracingRefTypeKey is the attribute Key conforming to the
	// "opentracing.ref_type" semantic conventions. It represents the parent-child
	// Reference type.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: The causal relationship between a child Span and a parent Span.
	OpenTracingRefTypeKey = attribute.Key("opentracing.ref_type")
)

// Enum values for opentracing.ref_type
var (
	// The parent Span depends on the child Span in some capacity
	// Stability: development
	OpenTracingRefTypeChildOf = OpenTracingRefTypeKey.String("child_of")
	// The parent Span doesn't depend in any way on the result of the child Span
	// Stability: development
	OpenTracingRefTypeFollowsFrom = OpenTracingRefTypeKey.String("follows_from")
)

// Namespace: os
const (
	// OSBuildIDKey is the attribute Key conforming to the "os.build_id" semantic
	// conventions. It represents the unique identifier for a particular build or
	// compilation of the operating system.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "TQ3C.230805.001.B2", "20E247", "22621"
	OSBuildIDKey = attribute.Key("os.build_id")

	// OSDescriptionKey is the attribute Key conforming to the "os.description"
	// semantic conventions. It represents the human readable (not intended to be
	// parsed) OS version information, like e.g. reported by `ver` or
	// `lsb_release -a` commands.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Microsoft Windows [Version 10.0.18363.778]", "Ubuntu 18.04.1 LTS"
	OSDescriptionKey = attribute.Key("os.description")

	// OSNameKey is the attribute Key conforming to the "os.name" semantic
	// conventions. It represents the human readable operating system name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "iOS", "Android", "Ubuntu"
	OSNameKey = attribute.Key("os.name")

	// OSTypeKey is the attribute Key conforming to the "os.type" semantic
	// conventions. It represents the operating system type.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	OSTypeKey = attribute.Key("os.type")

	// OSVersionKey is the attribute Key conforming to the "os.version" semantic
	// conventions. It represents the version string of the operating system as
	// defined in [Version Attributes].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "14.2.1", "18.04.1"
	//
	// [Version Attributes]: /docs/resource/README.md#version-attributes
	OSVersionKey = attribute.Key("os.version")
)

// OSBuildID returns an attribute KeyValue conforming to the "os.build_id"
// semantic conventions. It represents the unique identifier for a particular
// build or compilation of the operating system.
func OSBuildID(val string) attribute.KeyValue {
	return OSBuildIDKey.String(val)
}

// OSDescription returns an attribute KeyValue conforming to the "os.description"
// semantic conventions. It represents the human readable (not intended to be
// parsed) OS version information, like e.g. reported by `ver` or
// `lsb_release -a` commands.
func OSDescription(val string) attribute.KeyValue {
	return OSDescriptionKey.String(val)
}

// OSName returns an attribute KeyValue conforming to the "os.name" semantic
// conventions. It represents the human readable operating system name.
func OSName(val string) attribute.KeyValue {
	return OSNameKey.String(val)
}

// OSVersion returns an attribute KeyValue conforming to the "os.version"
// semantic conventions. It represents the version string of the operating system
// as defined in [Version Attributes].
//
// [Version Attributes]: /docs/resource/README.md#version-attributes
func OSVersion(val string) attribute.KeyValue {
	return OSVersionKey.String(val)
}

// Enum values for os.type
var (
	// Microsoft Windows
	// Stability: development
	OSTypeWindows = OSTypeKey.String("windows")
	// Linux
	// Stability: development
	OSTypeLinux = OSTypeKey.String("linux")
	// Apple Darwin
	// Stability: development
	OSTypeDarwin = OSTypeKey.String("darwin")
	// FreeBSD
	// Stability: development
	OSTypeFreeBSD = OSTypeKey.String("freebsd")
	// NetBSD
	// Stability: development
	OSTypeNetBSD = OSTypeKey.String("netbsd")
	// OpenBSD
	// Stability: development
	OSTypeOpenBSD = OSTypeKey.String("openbsd")
	// DragonFly BSD
	// Stability: development
	OSTypeDragonflyBSD = OSTypeKey.String("dragonflybsd")
	// HP-UX (Hewlett Packard Unix)
	// Stability: development
	OSTypeHPUX = OSTypeKey.String("hpux")
	// AIX (Advanced Interactive eXecutive)
	// Stability: development
	OSTypeAIX = OSTypeKey.String("aix")
	// SunOS, Oracle Solaris
	// Stability: development
	OSTypeSolaris = OSTypeKey.String("solaris")
	// IBM z/OS
	// Stability: development
	OSTypeZOS = OSTypeKey.String("zos")
)

// Namespace: otel
const (
	// OTelComponentNameKey is the attribute Key conforming to the
	// "otel.component.name" semantic conventions. It represents a name uniquely
	// identifying the instance of the OpenTelemetry component within its containing
	// SDK instance.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "otlp_grpc_span_exporter/0", "custom-name"
	// Note: Implementations SHOULD ensure a low cardinality for this attribute,
	// even across application or SDK restarts.
	// E.g. implementations MUST NOT use UUIDs as values for this attribute.
	//
	// Implementations MAY achieve these goals by following a
	// `<otel.component.type>/<instance-counter>` pattern, e.g.
	// `batching_span_processor/0`.
	// Hereby `otel.component.type` refers to the corresponding attribute value of
	// the component.
	//
	// The value of `instance-counter` MAY be automatically assigned by the
	// component and uniqueness within the enclosing SDK instance MUST be
	// guaranteed.
	// For example, `<instance-counter>` MAY be implemented by using a monotonically
	// increasing counter (starting with `0`), which is incremented every time an
	// instance of the given component type is started.
	//
	// With this implementation, for example the first Batching Span Processor would
	// have `batching_span_processor/0`
	// as `otel.component.name`, the second one `batching_span_processor/1` and so
	// on.
	// These values will therefore be reused in the case of an application restart.
	OTelComponentNameKey = attribute.Key("otel.component.name")

	// OTelComponentTypeKey is the attribute Key conforming to the
	// "otel.component.type" semantic conventions. It represents a name identifying
	// the type of the OpenTelemetry component.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "batching_span_processor", "com.example.MySpanExporter"
	// Note: If none of the standardized values apply, implementations SHOULD use
	// the language-defined name of the type.
	// E.g. for Java the fully qualified classname SHOULD be used in this case.
	OTelComponentTypeKey = attribute.Key("otel.component.type")

	// OTelScopeNameKey is the attribute Key conforming to the "otel.scope.name"
	// semantic conventions. It represents the name of the instrumentation scope - (
	// `InstrumentationScope.Name` in OTLP).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "io.opentelemetry.contrib.mongodb"
	OTelScopeNameKey = attribute.Key("otel.scope.name")

	// OTelScopeSchemaURLKey is the attribute Key conforming to the
	// "otel.scope.schema_url" semantic conventions. It represents the schema URL of
	// the instrumentation scope.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "https://opentelemetry.io/schemas/1.31.0"
	OTelScopeSchemaURLKey = attribute.Key("otel.scope.schema_url")

	// OTelScopeVersionKey is the attribute Key conforming to the
	// "otel.scope.version" semantic conventions. It represents the version of the
	// instrumentation scope - (`InstrumentationScope.Version` in OTLP).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "1.0.0"
	OTelScopeVersionKey = attribute.Key("otel.scope.version")

	// OTelSpanParentOriginKey is the attribute Key conforming to the
	// "otel.span.parent.origin" semantic conventions. It represents the determines
	// whether the span has a parent span, and if so,
	// [whether it is a remote parent].
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	//
	// [whether it is a remote parent]: https://opentelemetry.io/docs/specs/otel/trace/api/#isremote
	OTelSpanParentOriginKey = attribute.Key("otel.span.parent.origin")

	// OTelSpanSamplingResultKey is the attribute Key conforming to the
	// "otel.span.sampling_result" semantic conventions. It represents the result
	// value of the sampler for this span.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	OTelSpanSamplingResultKey = attribute.Key("otel.span.sampling_result")

	// OTelStatusCodeKey is the attribute Key conforming to the "otel.status_code"
	// semantic conventions. It represents the name of the code, either "OK" or
	// "ERROR". MUST NOT be set if the status code is UNSET.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples:
	OTelStatusCodeKey = attribute.Key("otel.status_code")

	// OTelStatusDescriptionKey is the attribute Key conforming to the
	// "otel.status_description" semantic conventions. It represents the description
	// of the Status if it has a value, otherwise not set.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "resource not found"
	OTelStatusDescriptionKey = attribute.Key("otel.status_description")
)

// OTelComponentName returns an attribute KeyValue conforming to the
// "otel.component.name" semantic conventions. It represents a name uniquely
// identifying the instance of the OpenTelemetry component within its containing
// SDK instance.
func OTelComponentName(val string) attribute.KeyValue {
	return OTelComponentNameKey.String(val)
}

// OTelScopeName returns an attribute KeyValue conforming to the
// "otel.scope.name" semantic conventions. It represents the name of the
// instrumentation scope - (`InstrumentationScope.Name` in OTLP).
func OTelScopeName(val string) attribute.KeyValue {
	return OTelScopeNameKey.String(val)
}

// OTelScopeSchemaURL returns an attribute KeyValue conforming to the
// "otel.scope.schema_url" semantic conventions. It represents the schema URL of
// the instrumentation scope.
func OTelScopeSchemaURL(val string) attribute.KeyValue {
	return OTelScopeSchemaURLKey.String(val)
}

// OTelScopeVersion returns an attribute KeyValue conforming to the
// "otel.scope.version" semantic conventions. It represents the version of the
// instrumentation scope - (`InstrumentationScope.Version` in OTLP).
func OTelScopeVersion(val string) attribute.KeyValue {
	return OTelScopeVersionKey.String(val)
}

// OTelStatusDescription returns an attribute KeyValue conforming to the
// "otel.status_description" semantic conventions. It represents the description
// of the Status if it has a value, otherwise not set.
func OTelStatusDescription(val string) attribute.KeyValue {
	return OTelStatusDescriptionKey.String(val)
}

// Enum values for otel.component.type
var (
	// The builtin SDK batching span processor
	//
	// Stability: development
	OTelComponentTypeBatchingSpanProcessor = OTelComponentTypeKey.String("batching_span_processor")
	// The builtin SDK simple span processor
	//
	// Stability: development
	OTelComponentTypeSimpleSpanProcessor = OTelComponentTypeKey.String("simple_span_processor")
	// The builtin SDK batching log record processor
	//
	// Stability: development
	OTelComponentTypeBatchingLogProcessor = OTelComponentTypeKey.String("batching_log_processor")
	// The builtin SDK simple log record processor
	//
	// Stability: development
	OTelComponentTypeSimpleLogProcessor = OTelComponentTypeKey.String("simple_log_processor")
	// OTLP span exporter over gRPC with protobuf serialization
	//
	// Stability: development
	OTelComponentTypeOtlpGRPCSpanExporter = OTelComponentTypeKey.String("otlp_grpc_span_exporter")
	// OTLP span exporter over HTTP with protobuf serialization
	//
	// Stability: development
	OTelComponentTypeOtlpHTTPSpanExporter = OTelComponentTypeKey.String("otlp_http_span_exporter")
	// OTLP span exporter over HTTP with JSON serialization
	//
	// Stability: development
	OTelComponentTypeOtlpHTTPJSONSpanExporter = OTelComponentTypeKey.String("otlp_http_json_span_exporter")
	// Zipkin span exporter over HTTP
	//
	// Stability: development
	OTelComponentTypeZipkinHTTPSpanExporter = OTelComponentTypeKey.String("zipkin_http_span_exporter")
	// OTLP log record exporter over gRPC with protobuf serialization
	//
	// Stability: development
	OTelComponentTypeOtlpGRPCLogExporter = OTelComponentTypeKey.String("otlp_grpc_log_exporter")
	// OTLP log record exporter over HTTP with protobuf serialization
	//
	// Stability: development
	OTelComponentTypeOtlpHTTPLogExporter = OTelComponentTypeKey.String("otlp_http_log_exporter")
	// OTLP log record exporter over HTTP with JSON serialization
	//
	// Stability: development
	OTelComponentTypeOtlpHTTPJSONLogExporter = OTelComponentTypeKey.String("otlp_http_json_log_exporter")
	// The builtin SDK periodically exporting metric reader
	//
	// Stability: development
	OTelComponentTypePeriodicMetricReader = OTelComponentTypeKey.String("periodic_metric_reader")
	// OTLP metric exporter over gRPC with protobuf serialization
	//
	// Stability: development
	OTelComponentTypeOtlpGRPCMetricExporter = OTelComponentTypeKey.String("otlp_grpc_metric_exporter")
	// OTLP metric exporter over HTTP with protobuf serialization
	//
	// Stability: development
	OTelComponentTypeOtlpHTTPMetricExporter = OTelComponentTypeKey.String("otlp_http_metric_exporter")
	// OTLP metric exporter over HTTP with JSON serialization
	//
	// Stability: development
	OTelComponentTypeOtlpHTTPJSONMetricExporter = OTelComponentTypeKey.String("otlp_http_json_metric_exporter")
	// Prometheus metric exporter over HTTP with the default text-based format
	//
	// Stability: development
	OTelComponentTypePrometheusHTTPTextMetricExporter = OTelComponentTypeKey.String("prometheus_http_text_metric_exporter")
)

// Enum values for otel.span.parent.origin
var (
	// The span does not have a parent, it is a root span
	// Stability: development
	OTelSpanParentOriginNone = OTelSpanParentOriginKey.String("none")
	// The span has a parent and the parent's span context [isRemote()] is false
	// Stability: development
	//
	// [isRemote()]: https://opentelemetry.io/docs/specs/otel/trace/api/#isremote
	OTelSpanParentOriginLocal = OTelSpanParentOriginKey.String("local")
	// The span has a parent and the parent's span context [isRemote()] is true
	// Stability: development
	//
	// [isRemote()]: https://opentelemetry.io/docs/specs/otel/trace/api/#isremote
	OTelSpanParentOriginRemote = OTelSpanParentOriginKey.String("remote")
)

// Enum values for otel.span.sampling_result
var (
	// The span is not sampled and not recording
	// Stability: development
	OTelSpanSamplingResultDrop = OTelSpanSamplingResultKey.String("DROP")
	// The span is not sampled, but recording
	// Stability: development
	OTelSpanSamplingResultRecordOnly = OTelSpanSamplingResultKey.String("RECORD_ONLY")
	// The span is sampled and recording
	// Stability: development
	OTelSpanSamplingResultRecordAndSample = OTelSpanSamplingResultKey.String("RECORD_AND_SAMPLE")
)

// Enum values for otel.status_code
var (
	// The operation has been validated by an Application developer or Operator to
	// have completed successfully.
	// Stability: stable
	OTelStatusCodeOk = OTelStatusCodeKey.String("OK")
	// The operation contains an error.
	// Stability: stable
	OTelStatusCodeError = OTelStatusCodeKey.String("ERROR")
)

// Namespace: peer
const (
	// PeerServiceKey is the attribute Key conforming to the "peer.service" semantic
	// conventions. It represents the [`service.name`] of the remote service. SHOULD
	// be equal to the actual `service.name` resource attribute of the remote
	// service if any.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: AuthTokenCache
	//
	// [`service.name`]: /docs/resource/README.md#service
	PeerServiceKey = attribute.Key("peer.service")
)

// PeerService returns an attribute KeyValue conforming to the "peer.service"
// semantic conventions. It represents the [`service.name`] of the remote
// service. SHOULD be equal to the actual `service.name` resource attribute of
// the remote service if any.
//
// [`service.name`]: /docs/resource/README.md#service
func PeerService(val string) attribute.KeyValue {
	return PeerServiceKey.String(val)
}

// Namespace: process
const (
	// ProcessArgsCountKey is the attribute Key conforming to the
	// "process.args_count" semantic conventions. It represents the length of the
	// process.command_args array.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 4
	// Note: This field can be useful for querying or performing bucket analysis on
	// how many arguments were provided to start a process. More arguments may be an
	// indication of suspicious activity.
	ProcessArgsCountKey = attribute.Key("process.args_count")

	// ProcessCommandKey is the attribute Key conforming to the "process.command"
	// semantic conventions. It represents the command used to launch the process
	// (i.e. the command name). On Linux based systems, can be set to the zeroth
	// string in `proc/[pid]/cmdline`. On Windows, can be set to the first parameter
	// extracted from `GetCommandLineW`.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "cmd/otelcol"
	ProcessCommandKey = attribute.Key("process.command")

	// ProcessCommandArgsKey is the attribute Key conforming to the
	// "process.command_args" semantic conventions. It represents the all the
	// command arguments (including the command/executable itself) as received by
	// the process. On Linux-based systems (and some other Unixoid systems
	// supporting procfs), can be set according to the list of null-delimited
	// strings extracted from `proc/[pid]/cmdline`. For libc-based executables, this
	// would be the full argv vector passed to `main`. SHOULD NOT be collected by
	// default unless there is sanitization that excludes sensitive data.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "cmd/otecol", "--config=config.yaml"
	ProcessCommandArgsKey = attribute.Key("process.command_args")

	// ProcessCommandLineKey is the attribute Key conforming to the
	// "process.command_line" semantic conventions. It represents the full command
	// used to launch the process as a single string representing the full command.
	// On Windows, can be set to the result of `GetCommandLineW`. Do not set this if
	// you have to assemble it just for monitoring; use `process.command_args`
	// instead. SHOULD NOT be collected by default unless there is sanitization that
	// excludes sensitive data.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "C:\cmd\otecol --config="my directory\config.yaml""
	ProcessCommandLineKey = attribute.Key("process.command_line")

	// ProcessContextSwitchTypeKey is the attribute Key conforming to the
	// "process.context_switch_type" semantic conventions. It represents the
	// specifies whether the context switches for this data point were voluntary or
	// involuntary.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	ProcessContextSwitchTypeKey = attribute.Key("process.context_switch_type")

	// ProcessCreationTimeKey is the attribute Key conforming to the
	// "process.creation.time" semantic conventions. It represents the date and time
	// the process was created, in ISO 8601 format.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2023-11-21T09:25:34.853Z"
	ProcessCreationTimeKey = attribute.Key("process.creation.time")

	// ProcessExecutableBuildIDGNUKey is the attribute Key conforming to the
	// "process.executable.build_id.gnu" semantic conventions. It represents the GNU
	// build ID as found in the `.note.gnu.build-id` ELF section (hex string).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "c89b11207f6479603b0d49bf291c092c2b719293"
	ProcessExecutableBuildIDGNUKey = attribute.Key("process.executable.build_id.gnu")

	// ProcessExecutableBuildIDGoKey is the attribute Key conforming to the
	// "process.executable.build_id.go" semantic conventions. It represents the Go
	// build ID as retrieved by `go tool buildid <go executable>`.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// "foh3mEXu7BLZjsN9pOwG/kATcXlYVCDEFouRMQed_/WwRFB1hPo9LBkekthSPG/x8hMC8emW2cCjXD0_1aY"
	ProcessExecutableBuildIDGoKey = attribute.Key("process.executable.build_id.go")

	// ProcessExecutableBuildIDHtlhashKey is the attribute Key conforming to the
	// "process.executable.build_id.htlhash" semantic conventions. It represents the
	// profiling specific build ID for executables. See the OTel specification for
	// Profiles for more information.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "600DCAFE4A110000F2BF38C493F5FB92"
	ProcessExecutableBuildIDHtlhashKey = attribute.Key("process.executable.build_id.htlhash")

	// ProcessExecutableNameKey is the attribute Key conforming to the
	// "process.executable.name" semantic conventions. It represents the name of the
	// process executable. On Linux based systems, this SHOULD be set to the base
	// name of the target of `/proc/[pid]/exe`. On Windows, this SHOULD be set to
	// the base name of `GetProcessImageFileNameW`.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "otelcol"
	ProcessExecutableNameKey = attribute.Key("process.executable.name")

	// ProcessExecutablePathKey is the attribute Key conforming to the
	// "process.executable.path" semantic conventions. It represents the full path
	// to the process executable. On Linux based systems, can be set to the target
	// of `proc/[pid]/exe`. On Windows, can be set to the result of
	// `GetProcessImageFileNameW`.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "/usr/bin/cmd/otelcol"
	ProcessExecutablePathKey = attribute.Key("process.executable.path")

	// ProcessExitCodeKey is the attribute Key conforming to the "process.exit.code"
	// semantic conventions. It represents the exit code of the process.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 127
	ProcessExitCodeKey = attribute.Key("process.exit.code")

	// ProcessExitTimeKey is the attribute Key conforming to the "process.exit.time"
	// semantic conventions. It represents the date and time the process exited, in
	// ISO 8601 format.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2023-11-21T09:26:12.315Z"
	ProcessExitTimeKey = attribute.Key("process.exit.time")

	// ProcessGroupLeaderPIDKey is the attribute Key conforming to the
	// "process.group_leader.pid" semantic conventions. It represents the PID of the
	// process's group leader. This is also the process group ID (PGID) of the
	// process.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 23
	ProcessGroupLeaderPIDKey = attribute.Key("process.group_leader.pid")

	// ProcessInteractiveKey is the attribute Key conforming to the
	// "process.interactive" semantic conventions. It represents the whether the
	// process is connected to an interactive shell.
	//
	// Type: boolean
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	ProcessInteractiveKey = attribute.Key("process.interactive")

	// ProcessLinuxCgroupKey is the attribute Key conforming to the
	// "process.linux.cgroup" semantic conventions. It represents the control group
	// associated with the process.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "1:name=systemd:/user.slice/user-1000.slice/session-3.scope",
	// "0::/user.slice/user-1000.slice/user@1000.service/tmux-spawn-0267755b-4639-4a27-90ed-f19f88e53748.scope"
	// Note: Control groups (cgroups) are a kernel feature used to organize and
	// manage process resources. This attribute provides the path(s) to the
	// cgroup(s) associated with the process, which should match the contents of the
	// [/proc/[PID]/cgroup] file.
	//
	// [/proc/[PID]/cgroup]: https://man7.org/linux/man-pages/man7/cgroups.7.html
	ProcessLinuxCgroupKey = attribute.Key("process.linux.cgroup")

	// ProcessOwnerKey is the attribute Key conforming to the "process.owner"
	// semantic conventions. It represents the username of the user that owns the
	// process.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "root"
	ProcessOwnerKey = attribute.Key("process.owner")

	// ProcessPagingFaultTypeKey is the attribute Key conforming to the
	// "process.paging.fault_type" semantic conventions. It represents the type of
	// page fault for this data point. Type `major` is for major/hard page faults,
	// and `minor` is for minor/soft page faults.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	ProcessPagingFaultTypeKey = attribute.Key("process.paging.fault_type")

	// ProcessParentPIDKey is the attribute Key conforming to the
	// "process.parent_pid" semantic conventions. It represents the parent Process
	// identifier (PPID).
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 111
	ProcessParentPIDKey = attribute.Key("process.parent_pid")

	// ProcessPIDKey is the attribute Key conforming to the "process.pid" semantic
	// conventions. It represents the process identifier (PID).
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1234
	ProcessPIDKey = attribute.Key("process.pid")

	// ProcessRealUserIDKey is the attribute Key conforming to the
	// "process.real_user.id" semantic conventions. It represents the real user ID
	// (RUID) of the process.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1000
	ProcessRealUserIDKey = attribute.Key("process.real_user.id")

	// ProcessRealUserNameKey is the attribute Key conforming to the
	// "process.real_user.name" semantic conventions. It represents the username of
	// the real user of the process.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "operator"
	ProcessRealUserNameKey = attribute.Key("process.real_user.name")

	// ProcessRuntimeDescriptionKey is the attribute Key conforming to the
	// "process.runtime.description" semantic conventions. It represents an
	// additional description about the runtime of the process, for example a
	// specific vendor customization of the runtime environment.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: Eclipse OpenJ9 Eclipse OpenJ9 VM openj9-0.21.0
	ProcessRuntimeDescriptionKey = attribute.Key("process.runtime.description")

	// ProcessRuntimeNameKey is the attribute Key conforming to the
	// "process.runtime.name" semantic conventions. It represents the name of the
	// runtime of this process.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "OpenJDK Runtime Environment"
	ProcessRuntimeNameKey = attribute.Key("process.runtime.name")

	// ProcessRuntimeVersionKey is the attribute Key conforming to the
	// "process.runtime.version" semantic conventions. It represents the version of
	// the runtime of this process, as returned by the runtime without modification.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 14.0.2
	ProcessRuntimeVersionKey = attribute.Key("process.runtime.version")

	// ProcessSavedUserIDKey is the attribute Key conforming to the
	// "process.saved_user.id" semantic conventions. It represents the saved user ID
	// (SUID) of the process.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1002
	ProcessSavedUserIDKey = attribute.Key("process.saved_user.id")

	// ProcessSavedUserNameKey is the attribute Key conforming to the
	// "process.saved_user.name" semantic conventions. It represents the username of
	// the saved user.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "operator"
	ProcessSavedUserNameKey = attribute.Key("process.saved_user.name")

	// ProcessSessionLeaderPIDKey is the attribute Key conforming to the
	// "process.session_leader.pid" semantic conventions. It represents the PID of
	// the process's session leader. This is also the session ID (SID) of the
	// process.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 14
	ProcessSessionLeaderPIDKey = attribute.Key("process.session_leader.pid")

	// ProcessTitleKey is the attribute Key conforming to the "process.title"
	// semantic conventions. It represents the process title (proctitle).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "cat /etc/hostname", "xfce4-session", "bash"
	// Note: In many Unix-like systems, process title (proctitle), is the string
	// that represents the name or command line of a running process, displayed by
	// system monitoring tools like ps, top, and htop.
	ProcessTitleKey = attribute.Key("process.title")

	// ProcessUserIDKey is the attribute Key conforming to the "process.user.id"
	// semantic conventions. It represents the effective user ID (EUID) of the
	// process.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1001
	ProcessUserIDKey = attribute.Key("process.user.id")

	// ProcessUserNameKey is the attribute Key conforming to the "process.user.name"
	// semantic conventions. It represents the username of the effective user of the
	// process.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "root"
	ProcessUserNameKey = attribute.Key("process.user.name")

	// ProcessVpidKey is the attribute Key conforming to the "process.vpid" semantic
	// conventions. It represents the virtual process identifier.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 12
	// Note: The process ID within a PID namespace. This is not necessarily unique
	// across all processes on the host but it is unique within the process
	// namespace that the process exists within.
	ProcessVpidKey = attribute.Key("process.vpid")

	// ProcessWorkingDirectoryKey is the attribute Key conforming to the
	// "process.working_directory" semantic conventions. It represents the working
	// directory of the process.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "/root"
	ProcessWorkingDirectoryKey = attribute.Key("process.working_directory")
)

// ProcessArgsCount returns an attribute KeyValue conforming to the
// "process.args_count" semantic conventions. It represents the length of the
// process.command_args array.
func ProcessArgsCount(val int) attribute.KeyValue {
	return ProcessArgsCountKey.Int(val)
}

// ProcessCommand returns an attribute KeyValue conforming to the
// "process.command" semantic conventions. It represents the command used to
// launch the process (i.e. the command name). On Linux based systems, can be set
// to the zeroth string in `proc/[pid]/cmdline`. On Windows, can be set to the
// first parameter extracted from `GetCommandLineW`.
func ProcessCommand(val string) attribute.KeyValue {
	return ProcessCommandKey.String(val)
}

// ProcessCommandArgs returns an attribute KeyValue conforming to the
// "process.command_args" semantic conventions. It represents the all the command
// arguments (including the command/executable itself) as received by the
// process. On Linux-based systems (and some other Unixoid systems supporting
// procfs), can be set according to the list of null-delimited strings extracted
// from `proc/[pid]/cmdline`. For libc-based executables, this would be the full
// argv vector passed to `main`. SHOULD NOT be collected by default unless there
// is sanitization that excludes sensitive data.
func ProcessCommandArgs(val ...string) attribute.KeyValue {
	return ProcessCommandArgsKey.StringSlice(val)
}

// ProcessCommandLine returns an attribute KeyValue conforming to the
// "process.command_line" semantic conventions. It represents the full command
// used to launch the process as a single string representing the full command.
// On Windows, can be set to the result of `GetCommandLineW`. Do not set this if
// you have to assemble it just for monitoring; use `process.command_args`
// instead. SHOULD NOT be collected by default unless there is sanitization that
// excludes sensitive data.
func ProcessCommandLine(val string) attribute.KeyValue {
	return ProcessCommandLineKey.String(val)
}

// ProcessCreationTime returns an attribute KeyValue conforming to the
// "process.creation.time" semantic conventions. It represents the date and time
// the process was created, in ISO 8601 format.
func ProcessCreationTime(val string) attribute.KeyValue {
	return ProcessCreationTimeKey.String(val)
}

// ProcessEnvironmentVariable returns an attribute KeyValue conforming to the
// "process.environment_variable" semantic conventions. It represents the process
// environment variables, `<key>` being the environment variable name, the value
// being the environment variable value.
func ProcessEnvironmentVariable(key string, val string) attribute.KeyValue {
	return attribute.String("process.environment_variable."+key, val)
}

// ProcessExecutableBuildIDGNU returns an attribute KeyValue conforming to the
// "process.executable.build_id.gnu" semantic conventions. It represents the GNU
// build ID as found in the `.note.gnu.build-id` ELF section (hex string).
func ProcessExecutableBuildIDGNU(val string) attribute.KeyValue {
	return ProcessExecutableBuildIDGNUKey.String(val)
}

// ProcessExecutableBuildIDGo returns an attribute KeyValue conforming to the
// "process.executable.build_id.go" semantic conventions. It represents the Go
// build ID as retrieved by `go tool buildid <go executable>`.
func ProcessExecutableBuildIDGo(val string) attribute.KeyValue {
	return ProcessExecutableBuildIDGoKey.String(val)
}

// ProcessExecutableBuildIDHtlhash returns an attribute KeyValue conforming to
// the "process.executable.build_id.htlhash" semantic conventions. It represents
// the profiling specific build ID for executables. See the OTel specification
// for Profiles for more information.
func ProcessExecutableBuildIDHtlhash(val string) attribute.KeyValue {
	return ProcessExecutableBuildIDHtlhashKey.String(val)
}

// ProcessExecutableName returns an attribute KeyValue conforming to the
// "process.executable.name" semantic conventions. It represents the name of the
// process executable. On Linux based systems, this SHOULD be set to the base
// name of the target of `/proc/[pid]/exe`. On Windows, this SHOULD be set to the
// base name of `GetProcessImageFileNameW`.
func ProcessExecutableName(val string) attribute.KeyValue {
	return ProcessExecutableNameKey.String(val)
}

// ProcessExecutablePath returns an attribute KeyValue conforming to the
// "process.executable.path" semantic conventions. It represents the full path to
// the process executable. On Linux based systems, can be set to the target of
// `proc/[pid]/exe`. On Windows, can be set to the result of
// `GetProcessImageFileNameW`.
func ProcessExecutablePath(val string) attribute.KeyValue {
	return ProcessExecutablePathKey.String(val)
}

// ProcessExitCode returns an attribute KeyValue conforming to the
// "process.exit.code" semantic conventions. It represents the exit code of the
// process.
func ProcessExitCode(val int) attribute.KeyValue {
	return ProcessExitCodeKey.Int(val)
}

// ProcessExitTime returns an attribute KeyValue conforming to the
// "process.exit.time" semantic conventions. It represents the date and time the
// process exited, in ISO 8601 format.
func ProcessExitTime(val string) attribute.KeyValue {
	return ProcessExitTimeKey.String(val)
}

// ProcessGroupLeaderPID returns an attribute KeyValue conforming to the
// "process.group_leader.pid" semantic conventions. It represents the PID of the
// process's group leader. This is also the process group ID (PGID) of the
// process.
func ProcessGroupLeaderPID(val int) attribute.KeyValue {
	return ProcessGroupLeaderPIDKey.Int(val)
}

// ProcessInteractive returns an attribute KeyValue conforming to the
// "process.interactive" semantic conventions. It represents the whether the
// process is connected to an interactive shell.
func ProcessInteractive(val bool) attribute.KeyValue {
	return ProcessInteractiveKey.Bool(val)
}

// ProcessLinuxCgroup returns an attribute KeyValue conforming to the
// "process.linux.cgroup" semantic conventions. It represents the control group
// associated with the process.
func ProcessLinuxCgroup(val string) attribute.KeyValue {
	return ProcessLinuxCgroupKey.String(val)
}

// ProcessOwner returns an attribute KeyValue conforming to the "process.owner"
// semantic conventions. It represents the username of the user that owns the
// process.
func ProcessOwner(val string) attribute.KeyValue {
	return ProcessOwnerKey.String(val)
}

// ProcessParentPID returns an attribute KeyValue conforming to the
// "process.parent_pid" semantic conventions. It represents the parent Process
// identifier (PPID).
func ProcessParentPID(val int) attribute.KeyValue {
	return ProcessParentPIDKey.Int(val)
}

// ProcessPID returns an attribute KeyValue conforming to the "process.pid"
// semantic conventions. It represents the process identifier (PID).
func ProcessPID(val int) attribute.KeyValue {
	return ProcessPIDKey.Int(val)
}

// ProcessRealUserID returns an attribute KeyValue conforming to the
// "process.real_user.id" semantic conventions. It represents the real user ID
// (RUID) of the process.
func ProcessRealUserID(val int) attribute.KeyValue {
	return ProcessRealUserIDKey.Int(val)
}

// ProcessRealUserName returns an attribute KeyValue conforming to the
// "process.real_user.name" semantic conventions. It represents the username of
// the real user of the process.
func ProcessRealUserName(val string) attribute.KeyValue {
	return ProcessRealUserNameKey.String(val)
}

// ProcessRuntimeDescription returns an attribute KeyValue conforming to the
// "process.runtime.description" semantic conventions. It represents an
// additional description about the runtime of the process, for example a
// specific vendor customization of the runtime environment.
func ProcessRuntimeDescription(val string) attribute.KeyValue {
	return ProcessRuntimeDescriptionKey.String(val)
}

// ProcessRuntimeName returns an attribute KeyValue conforming to the
// "process.runtime.name" semantic conventions. It represents the name of the
// runtime of this process.
func ProcessRuntimeName(val string) attribute.KeyValue {
	return ProcessRuntimeNameKey.String(val)
}

// ProcessRuntimeVersion returns an attribute KeyValue conforming to the
// "process.runtime.version" semantic conventions. It represents the version of
// the runtime of this process, as returned by the runtime without modification.
func ProcessRuntimeVersion(val string) attribute.KeyValue {
	return ProcessRuntimeVersionKey.String(val)
}

// ProcessSavedUserID returns an attribute KeyValue conforming to the
// "process.saved_user.id" semantic conventions. It represents the saved user ID
// (SUID) of the process.
func ProcessSavedUserID(val int) attribute.KeyValue {
	return ProcessSavedUserIDKey.Int(val)
}

// ProcessSavedUserName returns an attribute KeyValue conforming to the
// "process.saved_user.name" semantic conventions. It represents the username of
// the saved user.
func ProcessSavedUserName(val string) attribute.KeyValue {
	return ProcessSavedUserNameKey.String(val)
}

// ProcessSessionLeaderPID returns an attribute KeyValue conforming to the
// "process.session_leader.pid" semantic conventions. It represents the PID of
// the process's session leader. This is also the session ID (SID) of the
// process.
func ProcessSessionLeaderPID(val int) attribute.KeyValue {
	return ProcessSessionLeaderPIDKey.Int(val)
}

// ProcessTitle returns an attribute KeyValue conforming to the "process.title"
// semantic conventions. It represents the process title (proctitle).
func ProcessTitle(val string) attribute.KeyValue {
	return ProcessTitleKey.String(val)
}

// ProcessUserID returns an attribute KeyValue conforming to the
// "process.user.id" semantic conventions. It represents the effective user ID
// (EUID) of the process.
func ProcessUserID(val int) attribute.KeyValue {
	return ProcessUserIDKey.Int(val)
}

// ProcessUserName returns an attribute KeyValue conforming to the
// "process.user.name" semantic conventions. It represents the username of the
// effective user of the process.
func ProcessUserName(val string) attribute.KeyValue {
	return ProcessUserNameKey.String(val)
}

// ProcessVpid returns an attribute KeyValue conforming to the "process.vpid"
// semantic conventions. It represents the virtual process identifier.
func ProcessVpid(val int) attribute.KeyValue {
	return ProcessVpidKey.Int(val)
}

// ProcessWorkingDirectory returns an attribute KeyValue conforming to the
// "process.working_directory" semantic conventions. It represents the working
// directory of the process.
func ProcessWorkingDirectory(val string) attribute.KeyValue {
	return ProcessWorkingDirectoryKey.String(val)
}

// Enum values for process.context_switch_type
var (
	// voluntary
	// Stability: development
	ProcessContextSwitchTypeVoluntary = ProcessContextSwitchTypeKey.String("voluntary")
	// involuntary
	// Stability: development
	ProcessContextSwitchTypeInvoluntary = ProcessContextSwitchTypeKey.String("involuntary")
)

// Enum values for process.paging.fault_type
var (
	// major
	// Stability: development
	ProcessPagingFaultTypeMajor = ProcessPagingFaultTypeKey.String("major")
	// minor
	// Stability: development
	ProcessPagingFaultTypeMinor = ProcessPagingFaultTypeKey.String("minor")
)

// Namespace: profile
const (
	// ProfileFrameTypeKey is the attribute Key conforming to the
	// "profile.frame.type" semantic conventions. It represents the describes the
	// interpreter or compiler of a single frame.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "cpython"
	ProfileFrameTypeKey = attribute.Key("profile.frame.type")
)

// Enum values for profile.frame.type
var (
	// [.NET]
	//
	// Stability: development
	//
	// [.NET]: https://wikipedia.org/wiki/.NET
	ProfileFrameTypeDotnet = ProfileFrameTypeKey.String("dotnet")
	// [JVM]
	//
	// Stability: development
	//
	// [JVM]: https://wikipedia.org/wiki/Java_virtual_machine
	ProfileFrameTypeJVM = ProfileFrameTypeKey.String("jvm")
	// [Kernel]
	//
	// Stability: development
	//
	// [Kernel]: https://wikipedia.org/wiki/Kernel_(operating_system)
	ProfileFrameTypeKernel = ProfileFrameTypeKey.String("kernel")
	// Can be one of but not limited to [C], [C++], [Go] or [Rust]. If possible, a
	// more precise value MUST be used.
	//
	// Stability: development
	//
	// [C]: https://wikipedia.org/wiki/C_(programming_language)
	// [C++]: https://wikipedia.org/wiki/C%2B%2B
	// [Go]: https://wikipedia.org/wiki/Go_(programming_language)
	// [Rust]: https://wikipedia.org/wiki/Rust_(programming_language)
	ProfileFrameTypeNative = ProfileFrameTypeKey.String("native")
	// [Perl]
	//
	// Stability: development
	//
	// [Perl]: https://wikipedia.org/wiki/Perl
	ProfileFrameTypePerl = ProfileFrameTypeKey.String("perl")
	// [PHP]
	//
	// Stability: development
	//
	// [PHP]: https://wikipedia.org/wiki/PHP
	ProfileFrameTypePHP = ProfileFrameTypeKey.String("php")
	// [Python]
	//
	// Stability: development
	//
	// [Python]: https://wikipedia.org/wiki/Python_(programming_language)
	ProfileFrameTypeCpython = ProfileFrameTypeKey.String("cpython")
	// [Ruby]
	//
	// Stability: development
	//
	// [Ruby]: https://wikipedia.org/wiki/Ruby_(programming_language)
	ProfileFrameTypeRuby = ProfileFrameTypeKey.String("ruby")
	// [V8JS]
	//
	// Stability: development
	//
	// [V8JS]: https://wikipedia.org/wiki/V8_(JavaScript_engine)
	ProfileFrameTypeV8JS = ProfileFrameTypeKey.String("v8js")
	// [Erlang]
	//
	// Stability: development
	//
	// [Erlang]: https://en.wikipedia.org/wiki/BEAM_(Erlang_virtual_machine)
	ProfileFrameTypeBeam = ProfileFrameTypeKey.String("beam")
	// [Go],
	//
	// Stability: development
	//
	// [Go]: https://wikipedia.org/wiki/Go_(programming_language)
	ProfileFrameTypeGo = ProfileFrameTypeKey.String("go")
	// [Rust]
	//
	// Stability: development
	//
	// [Rust]: https://wikipedia.org/wiki/Rust_(programming_language)
	ProfileFrameTypeRust = ProfileFrameTypeKey.String("rust")
)

// Namespace: rpc
const (
	// RPCConnectRPCErrorCodeKey is the attribute Key conforming to the
	// "rpc.connect_rpc.error_code" semantic conventions. It represents the
	// [error codes] of the Connect request. Error codes are always string values.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	//
	// [error codes]: https://connectrpc.com//docs/protocol/#error-codes
	RPCConnectRPCErrorCodeKey = attribute.Key("rpc.connect_rpc.error_code")

	// RPCGRPCStatusCodeKey is the attribute Key conforming to the
	// "rpc.grpc.status_code" semantic conventions. It represents the
	// [numeric status code] of the gRPC request.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	//
	// [numeric status code]: https://github.com/grpc/grpc/blob/v1.33.2/doc/statuscodes.md
	RPCGRPCStatusCodeKey = attribute.Key("rpc.grpc.status_code")

	// RPCJSONRPCErrorCodeKey is the attribute Key conforming to the
	// "rpc.jsonrpc.error_code" semantic conventions. It represents the `error.code`
	//  property of response if it is an error response.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: -32700, 100
	RPCJSONRPCErrorCodeKey = attribute.Key("rpc.jsonrpc.error_code")

	// RPCJSONRPCErrorMessageKey is the attribute Key conforming to the
	// "rpc.jsonrpc.error_message" semantic conventions. It represents the
	// `error.message` property of response if it is an error response.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Parse error", "User already exists"
	RPCJSONRPCErrorMessageKey = attribute.Key("rpc.jsonrpc.error_message")

	// RPCJSONRPCRequestIDKey is the attribute Key conforming to the
	// "rpc.jsonrpc.request_id" semantic conventions. It represents the `id`
	// property of request or response. Since protocol allows id to be int, string,
	// `null` or missing (for notifications), value is expected to be cast to string
	// for simplicity. Use empty string in case of `null` value. Omit entirely if
	// this is a notification.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "10", "request-7", ""
	RPCJSONRPCRequestIDKey = attribute.Key("rpc.jsonrpc.request_id")

	// RPCJSONRPCVersionKey is the attribute Key conforming to the
	// "rpc.jsonrpc.version" semantic conventions. It represents the protocol
	// version as in `jsonrpc` property of request/response. Since JSON-RPC 1.0
	// doesn't specify this, the value can be omitted.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2.0", "1.0"
	RPCJSONRPCVersionKey = attribute.Key("rpc.jsonrpc.version")

	// RPCMessageCompressedSizeKey is the attribute Key conforming to the
	// "rpc.message.compressed_size" semantic conventions. It represents the
	// compressed size of the message in bytes.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	RPCMessageCompressedSizeKey = attribute.Key("rpc.message.compressed_size")

	// RPCMessageIDKey is the attribute Key conforming to the "rpc.message.id"
	// semantic conventions. It MUST be calculated as two different counters
	// starting from `1` one for sent messages and one for received message..
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: This way we guarantee that the values will be consistent between
	// different implementations.
	RPCMessageIDKey = attribute.Key("rpc.message.id")

	// RPCMessageTypeKey is the attribute Key conforming to the "rpc.message.type"
	// semantic conventions. It represents the whether this is a received or sent
	// message.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	RPCMessageTypeKey = attribute.Key("rpc.message.type")

	// RPCMessageUncompressedSizeKey is the attribute Key conforming to the
	// "rpc.message.uncompressed_size" semantic conventions. It represents the
	// uncompressed size of the message in bytes.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	RPCMessageUncompressedSizeKey = attribute.Key("rpc.message.uncompressed_size")

	// RPCMethodKey is the attribute Key conforming to the "rpc.method" semantic
	// conventions. It represents the name of the (logical) method being called,
	// must be equal to the $method part in the span name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: exampleMethod
	// Note: This is the logical name of the method from the RPC interface
	// perspective, which can be different from the name of any implementing
	// method/function. The `code.function.name` attribute may be used to store the
	// latter (e.g., method actually executing the call on the server side, RPC
	// client stub method on the client side).
	RPCMethodKey = attribute.Key("rpc.method")

	// RPCServiceKey is the attribute Key conforming to the "rpc.service" semantic
	// conventions. It represents the full (logical) name of the service being
	// called, including its package name, if applicable.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: myservice.EchoService
	// Note: This is the logical name of the service from the RPC interface
	// perspective, which can be different from the name of any implementing class.
	// The `code.namespace` attribute may be used to store the latter (despite the
	// attribute name, it may include a class name; e.g., class with method actually
	// executing the call on the server side, RPC client stub class on the client
	// side).
	RPCServiceKey = attribute.Key("rpc.service")

	// RPCSystemKey is the attribute Key conforming to the "rpc.system" semantic
	// conventions. It represents a string identifying the remoting system. See
	// below for a list of well-known identifiers.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	RPCSystemKey = attribute.Key("rpc.system")
)

// RPCConnectRPCRequestMetadata returns an attribute KeyValue conforming to the
// "rpc.connect_rpc.request.metadata" semantic conventions. It represents the
// connect request metadata, `<key>` being the normalized Connect Metadata key
// (lowercase), the value being the metadata values.
func RPCConnectRPCRequestMetadata(key string, val ...string) attribute.KeyValue {
	return attribute.StringSlice("rpc.connect_rpc.request.metadata."+key, val)
}

// RPCConnectRPCResponseMetadata returns an attribute KeyValue conforming to the
// "rpc.connect_rpc.response.metadata" semantic conventions. It represents the
// connect response metadata, `<key>` being the normalized Connect Metadata key
// (lowercase), the value being the metadata values.
func RPCConnectRPCResponseMetadata(key string, val ...string) attribute.KeyValue {
	return attribute.StringSlice("rpc.connect_rpc.response.metadata."+key, val)
}

// RPCGRPCRequestMetadata returns an attribute KeyValue conforming to the
// "rpc.grpc.request.metadata" semantic conventions. It represents the gRPC
// request metadata, `<key>` being the normalized gRPC Metadata key (lowercase),
// the value being the metadata values.
func RPCGRPCRequestMetadata(key string, val ...string) attribute.KeyValue {
	return attribute.StringSlice("rpc.grpc.request.metadata."+key, val)
}

// RPCGRPCResponseMetadata returns an attribute KeyValue conforming to the
// "rpc.grpc.response.metadata" semantic conventions. It represents the gRPC
// response metadata, `<key>` being the normalized gRPC Metadata key (lowercase),
// the value being the metadata values.
func RPCGRPCResponseMetadata(key string, val ...string) attribute.KeyValue {
	return attribute.StringSlice("rpc.grpc.response.metadata."+key, val)
}

// RPCJSONRPCErrorCode returns an attribute KeyValue conforming to the
// "rpc.jsonrpc.error_code" semantic conventions. It represents the `error.code`
// property of response if it is an error response.
func RPCJSONRPCErrorCode(val int) attribute.KeyValue {
	return RPCJSONRPCErrorCodeKey.Int(val)
}

// RPCJSONRPCErrorMessage returns an attribute KeyValue conforming to the
// "rpc.jsonrpc.error_message" semantic conventions. It represents the
// `error.message` property of response if it is an error response.
func RPCJSONRPCErrorMessage(val string) attribute.KeyValue {
	return RPCJSONRPCErrorMessageKey.String(val)
}

// RPCJSONRPCRequestID returns an attribute KeyValue conforming to the
// "rpc.jsonrpc.request_id" semantic conventions. It represents the `id` property
// of request or response. Since protocol allows id to be int, string, `null` or
// missing (for notifications), value is expected to be cast to string for
// simplicity. Use empty string in case of `null` value. Omit entirely if this is
// a notification.
func RPCJSONRPCRequestID(val string) attribute.KeyValue {
	return RPCJSONRPCRequestIDKey.String(val)
}

// RPCJSONRPCVersion returns an attribute KeyValue conforming to the
// "rpc.jsonrpc.version" semantic conventions. It represents the protocol version
// as in `jsonrpc` property of request/response. Since JSON-RPC 1.0 doesn't
// specify this, the value can be omitted.
func RPCJSONRPCVersion(val string) attribute.KeyValue {
	return RPCJSONRPCVersionKey.String(val)
}

// RPCMessageCompressedSize returns an attribute KeyValue conforming to the
// "rpc.message.compressed_size" semantic conventions. It represents the
// compressed size of the message in bytes.
func RPCMessageCompressedSize(val int) attribute.KeyValue {
	return RPCMessageCompressedSizeKey.Int(val)
}

// RPCMessageID returns an attribute KeyValue conforming to the "rpc.message.id"
// semantic conventions. It MUST be calculated as two different counters starting
// from `1` one for sent messages and one for received message..
func RPCMessageID(val int) attribute.KeyValue {
	return RPCMessageIDKey.Int(val)
}

// RPCMessageUncompressedSize returns an attribute KeyValue conforming to the
// "rpc.message.uncompressed_size" semantic conventions. It represents the
// uncompressed size of the message in bytes.
func RPCMessageUncompressedSize(val int) attribute.KeyValue {
	return RPCMessageUncompressedSizeKey.Int(val)
}

// RPCMethod returns an attribute KeyValue conforming to the "rpc.method"
// semantic conventions. It represents the name of the (logical) method being
// called, must be equal to the $method part in the span name.
func RPCMethod(val string) attribute.KeyValue {
	return RPCMethodKey.String(val)
}

// RPCService returns an attribute KeyValue conforming to the "rpc.service"
// semantic conventions. It represents the full (logical) name of the service
// being called, including its package name, if applicable.
func RPCService(val string) attribute.KeyValue {
	return RPCServiceKey.String(val)
}

// Enum values for rpc.connect_rpc.error_code
var (
	// cancelled
	// Stability: development
	RPCConnectRPCErrorCodeCancelled = RPCConnectRPCErrorCodeKey.String("cancelled")
	// unknown
	// Stability: development
	RPCConnectRPCErrorCodeUnknown = RPCConnectRPCErrorCodeKey.String("unknown")
	// invalid_argument
	// Stability: development
	RPCConnectRPCErrorCodeInvalidArgument = RPCConnectRPCErrorCodeKey.String("invalid_argument")
	// deadline_exceeded
	// Stability: development
	RPCConnectRPCErrorCodeDeadlineExceeded = RPCConnectRPCErrorCodeKey.String("deadline_exceeded")
	// not_found
	// Stability: development
	RPCConnectRPCErrorCodeNotFound = RPCConnectRPCErrorCodeKey.String("not_found")
	// already_exists
	// Stability: development
	RPCConnectRPCErrorCodeAlreadyExists = RPCConnectRPCErrorCodeKey.String("already_exists")
	// permission_denied
	// Stability: development
	RPCConnectRPCErrorCodePermissionDenied = RPCConnectRPCErrorCodeKey.String("permission_denied")
	// resource_exhausted
	// Stability: development
	RPCConnectRPCErrorCodeResourceExhausted = RPCConnectRPCErrorCodeKey.String("resource_exhausted")
	// failed_precondition
	// Stability: development
	RPCConnectRPCErrorCodeFailedPrecondition = RPCConnectRPCErrorCodeKey.String("failed_precondition")
	// aborted
	// Stability: development
	RPCConnectRPCErrorCodeAborted = RPCConnectRPCErrorCodeKey.String("aborted")
	// out_of_range
	// Stability: development
	RPCConnectRPCErrorCodeOutOfRange = RPCConnectRPCErrorCodeKey.String("out_of_range")
	// unimplemented
	// Stability: development
	RPCConnectRPCErrorCodeUnimplemented = RPCConnectRPCErrorCodeKey.String("unimplemented")
	// internal
	// Stability: development
	RPCConnectRPCErrorCodeInternal = RPCConnectRPCErrorCodeKey.String("internal")
	// unavailable
	// Stability: development
	RPCConnectRPCErrorCodeUnavailable = RPCConnectRPCErrorCodeKey.String("unavailable")
	// data_loss
	// Stability: development
	RPCConnectRPCErrorCodeDataLoss = RPCConnectRPCErrorCodeKey.String("data_loss")
	// unauthenticated
	// Stability: development
	RPCConnectRPCErrorCodeUnauthenticated = RPCConnectRPCErrorCodeKey.String("unauthenticated")
)

// Enum values for rpc.grpc.status_code
var (
	// OK
	// Stability: development
	RPCGRPCStatusCodeOk = RPCGRPCStatusCodeKey.Int(0)
	// CANCELLED
	// Stability: development
	RPCGRPCStatusCodeCancelled = RPCGRPCStatusCodeKey.Int(1)
	// UNKNOWN
	// Stability: development
	RPCGRPCStatusCodeUnknown = RPCGRPCStatusCodeKey.Int(2)
	// INVALID_ARGUMENT
	// Stability: development
	RPCGRPCStatusCodeInvalidArgument = RPCGRPCStatusCodeKey.Int(3)
	// DEADLINE_EXCEEDED
	// Stability: development
	RPCGRPCStatusCodeDeadlineExceeded = RPCGRPCStatusCodeKey.Int(4)
	// NOT_FOUND
	// Stability: development
	RPCGRPCStatusCodeNotFound = RPCGRPCStatusCodeKey.Int(5)
	// ALREADY_EXISTS
	// Stability: development
	RPCGRPCStatusCodeAlreadyExists = RPCGRPCStatusCodeKey.Int(6)
	// PERMISSION_DENIED
	// Stability: development
	RPCGRPCStatusCodePermissionDenied = RPCGRPCStatusCodeKey.Int(7)
	// RESOURCE_EXHAUSTED
	// Stability: development
	RPCGRPCStatusCodeResourceExhausted = RPCGRPCStatusCodeKey.Int(8)
	// FAILED_PRECONDITION
	// Stability: development
	RPCGRPCStatusCodeFailedPrecondition = RPCGRPCStatusCodeKey.Int(9)
	// ABORTED
	// Stability: development
	RPCGRPCStatusCodeAborted = RPCGRPCStatusCodeKey.Int(10)
	// OUT_OF_RANGE
	// Stability: development
	RPCGRPCStatusCodeOutOfRange = RPCGRPCStatusCodeKey.Int(11)
	// UNIMPLEMENTED
	// Stability: development
	RPCGRPCStatusCodeUnimplemented = RPCGRPCStatusCodeKey.Int(12)
	// INTERNAL
	// Stability: development
	RPCGRPCStatusCodeInternal = RPCGRPCStatusCodeKey.Int(13)
	// UNAVAILABLE
	// Stability: development
	RPCGRPCStatusCodeUnavailable = RPCGRPCStatusCodeKey.Int(14)
	// DATA_LOSS
	// Stability: development
	RPCGRPCStatusCodeDataLoss = RPCGRPCStatusCodeKey.Int(15)
	// UNAUTHENTICATED
	// Stability: development
	RPCGRPCStatusCodeUnauthenticated = RPCGRPCStatusCodeKey.Int(16)
)

// Enum values for rpc.message.type
var (
	// sent
	// Stability: development
	RPCMessageTypeSent = RPCMessageTypeKey.String("SENT")
	// received
	// Stability: development
	RPCMessageTypeReceived = RPCMessageTypeKey.String("RECEIVED")
)

// Enum values for rpc.system
var (
	// gRPC
	// Stability: development
	RPCSystemGRPC = RPCSystemKey.String("grpc")
	// Java RMI
	// Stability: development
	RPCSystemJavaRmi = RPCSystemKey.String("java_rmi")
	// .NET WCF
	// Stability: development
	RPCSystemDotnetWcf = RPCSystemKey.String("dotnet_wcf")
	// Apache Dubbo
	// Stability: development
	RPCSystemApacheDubbo = RPCSystemKey.String("apache_dubbo")
	// Connect RPC
	// Stability: development
	RPCSystemConnectRPC = RPCSystemKey.String("connect_rpc")
)

// Namespace: security_rule
const (
	// SecurityRuleCategoryKey is the attribute Key conforming to the
	// "security_rule.category" semantic conventions. It represents a categorization
	// value keyword used by the entity using the rule for detection of this event.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Attempted Information Leak"
	SecurityRuleCategoryKey = attribute.Key("security_rule.category")

	// SecurityRuleDescriptionKey is the attribute Key conforming to the
	// "security_rule.description" semantic conventions. It represents the
	// description of the rule generating the event.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Block requests to public DNS over HTTPS / TLS protocols"
	SecurityRuleDescriptionKey = attribute.Key("security_rule.description")

	// SecurityRuleLicenseKey is the attribute Key conforming to the
	// "security_rule.license" semantic conventions. It represents the name of the
	// license under which the rule used to generate this event is made available.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Apache 2.0"
	SecurityRuleLicenseKey = attribute.Key("security_rule.license")

	// SecurityRuleNameKey is the attribute Key conforming to the
	// "security_rule.name" semantic conventions. It represents the name of the rule
	// or signature generating the event.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "BLOCK_DNS_over_TLS"
	SecurityRuleNameKey = attribute.Key("security_rule.name")

	// SecurityRuleReferenceKey is the attribute Key conforming to the
	// "security_rule.reference" semantic conventions. It represents the reference
	// URL to additional information about the rule used to generate this event.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "https://en.wikipedia.org/wiki/DNS_over_TLS"
	// Note: The URL can point to the vendor’s documentation about the rule. If
	// that’s not available, it can also be a link to a more general page
	// describing this type of alert.
	SecurityRuleReferenceKey = attribute.Key("security_rule.reference")

	// SecurityRuleRulesetNameKey is the attribute Key conforming to the
	// "security_rule.ruleset.name" semantic conventions. It represents the name of
	// the ruleset, policy, group, or parent category in which the rule used to
	// generate this event is a member.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Standard_Protocol_Filters"
	SecurityRuleRulesetNameKey = attribute.Key("security_rule.ruleset.name")

	// SecurityRuleUUIDKey is the attribute Key conforming to the
	// "security_rule.uuid" semantic conventions. It represents a rule ID that is
	// unique within the scope of a set or group of agents, observers, or other
	// entities using the rule for detection of this event.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "550e8400-e29b-41d4-a716-446655440000", "1100110011"
	SecurityRuleUUIDKey = attribute.Key("security_rule.uuid")

	// SecurityRuleVersionKey is the attribute Key conforming to the
	// "security_rule.version" semantic conventions. It represents the version /
	// revision of the rule being used for analysis.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "1.0.0"
	SecurityRuleVersionKey = attribute.Key("security_rule.version")
)

// SecurityRuleCategory returns an attribute KeyValue conforming to the
// "security_rule.category" semantic conventions. It represents a categorization
// value keyword used by the entity using the rule for detection of this event.
func SecurityRuleCategory(val string) attribute.KeyValue {
	return SecurityRuleCategoryKey.String(val)
}

// SecurityRuleDescription returns an attribute KeyValue conforming to the
// "security_rule.description" semantic conventions. It represents the
// description of the rule generating the event.
func SecurityRuleDescription(val string) attribute.KeyValue {
	return SecurityRuleDescriptionKey.String(val)
}

// SecurityRuleLicense returns an attribute KeyValue conforming to the
// "security_rule.license" semantic conventions. It represents the name of the
// license under which the rule used to generate this event is made available.
func SecurityRuleLicense(val string) attribute.KeyValue {
	return SecurityRuleLicenseKey.String(val)
}

// SecurityRuleName returns an attribute KeyValue conforming to the
// "security_rule.name" semantic conventions. It represents the name of the rule
// or signature generating the event.
func SecurityRuleName(val string) attribute.KeyValue {
	return SecurityRuleNameKey.String(val)
}

// SecurityRuleReference returns an attribute KeyValue conforming to the
// "security_rule.reference" semantic conventions. It represents the reference
// URL to additional information about the rule used to generate this event.
func SecurityRuleReference(val string) attribute.KeyValue {
	return SecurityRuleReferenceKey.String(val)
}

// SecurityRuleRulesetName returns an attribute KeyValue conforming to the
// "security_rule.ruleset.name" semantic conventions. It represents the name of
// the ruleset, policy, group, or parent category in which the rule used to
// generate this event is a member.
func SecurityRuleRulesetName(val string) attribute.KeyValue {
	return SecurityRuleRulesetNameKey.String(val)
}

// SecurityRuleUUID returns an attribute KeyValue conforming to the
// "security_rule.uuid" semantic conventions. It represents a rule ID that is
// unique within the scope of a set or group of agents, observers, or other
// entities using the rule for detection of this event.
func SecurityRuleUUID(val string) attribute.KeyValue {
	return SecurityRuleUUIDKey.String(val)
}

// SecurityRuleVersion returns an attribute KeyValue conforming to the
// "security_rule.version" semantic conventions. It represents the version /
// revision of the rule being used for analysis.
func SecurityRuleVersion(val string) attribute.KeyValue {
	return SecurityRuleVersionKey.String(val)
}

// Namespace: server
const (
	// ServerAddressKey is the attribute Key conforming to the "server.address"
	// semantic conventions. It represents the server domain name if available
	// without reverse DNS lookup; otherwise, IP address or Unix domain socket name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "example.com", "10.1.2.80", "/tmp/my.sock"
	// Note: When observed from the client side, and when communicating through an
	// intermediary, `server.address` SHOULD represent the server address behind any
	// intermediaries, for example proxies, if it's available.
	ServerAddressKey = attribute.Key("server.address")

	// ServerPortKey is the attribute Key conforming to the "server.port" semantic
	// conventions. It represents the server port number.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: 80, 8080, 443
	// Note: When observed from the client side, and when communicating through an
	// intermediary, `server.port` SHOULD represent the server port behind any
	// intermediaries, for example proxies, if it's available.
	ServerPortKey = attribute.Key("server.port")
)

// ServerAddress returns an attribute KeyValue conforming to the "server.address"
// semantic conventions. It represents the server domain name if available
// without reverse DNS lookup; otherwise, IP address or Unix domain socket name.
func ServerAddress(val string) attribute.KeyValue {
	return ServerAddressKey.String(val)
}

// ServerPort returns an attribute KeyValue conforming to the "server.port"
// semantic conventions. It represents the server port number.
func ServerPort(val int) attribute.KeyValue {
	return ServerPortKey.Int(val)
}

// Namespace: service
const (
	// ServiceInstanceIDKey is the attribute Key conforming to the
	// "service.instance.id" semantic conventions. It represents the string ID of
	// the service instance.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "627cc493-f310-47de-96bd-71410b7dec09"
	// Note: MUST be unique for each instance of the same
	// `service.namespace,service.name` pair (in other words
	// `service.namespace,service.name,service.instance.id` triplet MUST be globally
	// unique). The ID helps to
	// distinguish instances of the same service that exist at the same time (e.g.
	// instances of a horizontally scaled
	// service).
	//
	// Implementations, such as SDKs, are recommended to generate a random Version 1
	// or Version 4 [RFC
	// 4122] UUID, but are free to use an inherent unique ID as
	// the source of
	// this value if stability is desirable. In that case, the ID SHOULD be used as
	// source of a UUID Version 5 and
	// SHOULD use the following UUID as the namespace:
	// `4d63009a-8d0f-11ee-aad7-4c796ed8e320`.
	//
	// UUIDs are typically recommended, as only an opaque value for the purposes of
	// identifying a service instance is
	// needed. Similar to what can be seen in the man page for the
	// [`/etc/machine-id`] file, the underlying
	// data, such as pod name and namespace should be treated as confidential, being
	// the user's choice to expose it
	// or not via another resource attribute.
	//
	// For applications running behind an application server (like unicorn), we do
	// not recommend using one identifier
	// for all processes participating in the application. Instead, it's recommended
	// each division (e.g. a worker
	// thread in unicorn) to have its own instance.id.
	//
	// It's not recommended for a Collector to set `service.instance.id` if it can't
	// unambiguously determine the
	// service instance that is generating that telemetry. For instance, creating an
	// UUID based on `pod.name` will
	// likely be wrong, as the Collector might not know from which container within
	// that pod the telemetry originated.
	// However, Collectors can set the `service.instance.id` if they can
	// unambiguously determine the service instance
	// for that telemetry. This is typically the case for scraping receivers, as
	// they know the target address and
	// port.
	//
	// [RFC
	// 4122]: https://www.ietf.org/rfc/rfc4122.txt
	// [`/etc/machine-id`]: https://www.freedesktop.org/software/systemd/man/latest/machine-id.html
	ServiceInstanceIDKey = attribute.Key("service.instance.id")

	// ServiceNameKey is the attribute Key conforming to the "service.name" semantic
	// conventions. It represents the logical name of the service.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "shoppingcart"
	// Note: MUST be the same for all instances of horizontally scaled services. If
	// the value was not specified, SDKs MUST fallback to `unknown_service:`
	// concatenated with [`process.executable.name`], e.g. `unknown_service:bash`.
	// If `process.executable.name` is not available, the value MUST be set to
	// `unknown_service`.
	//
	// [`process.executable.name`]: process.md
	ServiceNameKey = attribute.Key("service.name")

	// ServiceNamespaceKey is the attribute Key conforming to the
	// "service.namespace" semantic conventions. It represents a namespace for
	// `service.name`.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Shop"
	// Note: A string value having a meaning that helps to distinguish a group of
	// services, for example the team name that owns a group of services.
	// `service.name` is expected to be unique within the same namespace. If
	// `service.namespace` is not specified in the Resource then `service.name` is
	// expected to be unique for all services that have no explicit namespace
	// defined (so the empty/unspecified namespace is simply one more valid
	// namespace). Zero-length namespace string is assumed equal to unspecified
	// namespace.
	ServiceNamespaceKey = attribute.Key("service.namespace")

	// ServiceVersionKey is the attribute Key conforming to the "service.version"
	// semantic conventions. It represents the version string of the service API or
	// implementation. The format is not defined by these conventions.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "2.0.0", "a01dbef8a"
	ServiceVersionKey = attribute.Key("service.version")
)

// ServiceInstanceID returns an attribute KeyValue conforming to the
// "service.instance.id" semantic conventions. It represents the string ID of the
// service instance.
func ServiceInstanceID(val string) attribute.KeyValue {
	return ServiceInstanceIDKey.String(val)
}

// ServiceName returns an attribute KeyValue conforming to the "service.name"
// semantic conventions. It represents the logical name of the service.
func ServiceName(val string) attribute.KeyValue {
	return ServiceNameKey.String(val)
}

// ServiceNamespace returns an attribute KeyValue conforming to the
// "service.namespace" semantic conventions. It represents a namespace for
// `service.name`.
func ServiceNamespace(val string) attribute.KeyValue {
	return ServiceNamespaceKey.String(val)
}

// ServiceVersion returns an attribute KeyValue conforming to the
// "service.version" semantic conventions. It represents the version string of
// the service API or implementation. The format is not defined by these
// conventions.
func ServiceVersion(val string) attribute.KeyValue {
	return ServiceVersionKey.String(val)
}

// Namespace: session
const (
	// SessionIDKey is the attribute Key conforming to the "session.id" semantic
	// conventions. It represents a unique id to identify a session.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 00112233-4455-6677-8899-aabbccddeeff
	SessionIDKey = attribute.Key("session.id")

	// SessionPreviousIDKey is the attribute Key conforming to the
	// "session.previous_id" semantic conventions. It represents the previous
	// `session.id` for this user, when known.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 00112233-4455-6677-8899-aabbccddeeff
	SessionPreviousIDKey = attribute.Key("session.previous_id")
)

// SessionID returns an attribute KeyValue conforming to the "session.id"
// semantic conventions. It represents a unique id to identify a session.
func SessionID(val string) attribute.KeyValue {
	return SessionIDKey.String(val)
}

// SessionPreviousID returns an attribute KeyValue conforming to the
// "session.previous_id" semantic conventions. It represents the previous
// `session.id` for this user, when known.
func SessionPreviousID(val string) attribute.KeyValue {
	return SessionPreviousIDKey.String(val)
}

// Namespace: signalr
const (
	// SignalRConnectionStatusKey is the attribute Key conforming to the
	// "signalr.connection.status" semantic conventions. It represents the signalR
	// HTTP connection closure status.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "app_shutdown", "timeout"
	SignalRConnectionStatusKey = attribute.Key("signalr.connection.status")

	// SignalRTransportKey is the attribute Key conforming to the
	// "signalr.transport" semantic conventions. It represents the
	// [SignalR transport type].
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "web_sockets", "long_polling"
	//
	// [SignalR transport type]: https://github.com/dotnet/aspnetcore/blob/main/src/SignalR/docs/specs/TransportProtocols.md
	SignalRTransportKey = attribute.Key("signalr.transport")
)

// Enum values for signalr.connection.status
var (
	// The connection was closed normally.
	// Stability: stable
	SignalRConnectionStatusNormalClosure = SignalRConnectionStatusKey.String("normal_closure")
	// The connection was closed due to a timeout.
	// Stability: stable
	SignalRConnectionStatusTimeout = SignalRConnectionStatusKey.String("timeout")
	// The connection was closed because the app is shutting down.
	// Stability: stable
	SignalRConnectionStatusAppShutdown = SignalRConnectionStatusKey.String("app_shutdown")
)

// Enum values for signalr.transport
var (
	// ServerSentEvents protocol
	// Stability: stable
	SignalRTransportServerSentEvents = SignalRTransportKey.String("server_sent_events")
	// LongPolling protocol
	// Stability: stable
	SignalRTransportLongPolling = SignalRTransportKey.String("long_polling")
	// WebSockets protocol
	// Stability: stable
	SignalRTransportWebSockets = SignalRTransportKey.String("web_sockets")
)

// Namespace: source
const (
	// SourceAddressKey is the attribute Key conforming to the "source.address"
	// semantic conventions. It represents the source address - domain name if
	// available without reverse DNS lookup; otherwise, IP address or Unix domain
	// socket name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "source.example.com", "10.1.2.80", "/tmp/my.sock"
	// Note: When observed from the destination side, and when communicating through
	// an intermediary, `source.address` SHOULD represent the source address behind
	// any intermediaries, for example proxies, if it's available.
	SourceAddressKey = attribute.Key("source.address")

	// SourcePortKey is the attribute Key conforming to the "source.port" semantic
	// conventions. It represents the source port number.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 3389, 2888
	SourcePortKey = attribute.Key("source.port")
)

// SourceAddress returns an attribute KeyValue conforming to the "source.address"
// semantic conventions. It represents the source address - domain name if
// available without reverse DNS lookup; otherwise, IP address or Unix domain
// socket name.
func SourceAddress(val string) attribute.KeyValue {
	return SourceAddressKey.String(val)
}

// SourcePort returns an attribute KeyValue conforming to the "source.port"
// semantic conventions. It represents the source port number.
func SourcePort(val int) attribute.KeyValue {
	return SourcePortKey.Int(val)
}

// Namespace: system
const (
	// SystemCPULogicalNumberKey is the attribute Key conforming to the
	// "system.cpu.logical_number" semantic conventions. It represents the
	// deprecated, use `cpu.logical_number` instead.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 1
	SystemCPULogicalNumberKey = attribute.Key("system.cpu.logical_number")

	// SystemDeviceKey is the attribute Key conforming to the "system.device"
	// semantic conventions. It represents the device identifier.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "(identifier)"
	SystemDeviceKey = attribute.Key("system.device")

	// SystemFilesystemModeKey is the attribute Key conforming to the
	// "system.filesystem.mode" semantic conventions. It represents the filesystem
	// mode.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "rw, ro"
	SystemFilesystemModeKey = attribute.Key("system.filesystem.mode")

	// SystemFilesystemMountpointKey is the attribute Key conforming to the
	// "system.filesystem.mountpoint" semantic conventions. It represents the
	// filesystem mount path.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "/mnt/data"
	SystemFilesystemMountpointKey = attribute.Key("system.filesystem.mountpoint")

	// SystemFilesystemStateKey is the attribute Key conforming to the
	// "system.filesystem.state" semantic conventions. It represents the filesystem
	// state.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "used"
	SystemFilesystemStateKey = attribute.Key("system.filesystem.state")

	// SystemFilesystemTypeKey is the attribute Key conforming to the
	// "system.filesystem.type" semantic conventions. It represents the filesystem
	// type.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "ext4"
	SystemFilesystemTypeKey = attribute.Key("system.filesystem.type")

	// SystemMemoryStateKey is the attribute Key conforming to the
	// "system.memory.state" semantic conventions. It represents the memory state.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "free", "cached"
	SystemMemoryStateKey = attribute.Key("system.memory.state")

	// SystemPagingDirectionKey is the attribute Key conforming to the
	// "system.paging.direction" semantic conventions. It represents the paging
	// access direction.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "in"
	SystemPagingDirectionKey = attribute.Key("system.paging.direction")

	// SystemPagingStateKey is the attribute Key conforming to the
	// "system.paging.state" semantic conventions. It represents the memory paging
	// state.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "free"
	SystemPagingStateKey = attribute.Key("system.paging.state")

	// SystemPagingTypeKey is the attribute Key conforming to the
	// "system.paging.type" semantic conventions. It represents the memory paging
	// type.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "minor"
	SystemPagingTypeKey = attribute.Key("system.paging.type")

	// SystemProcessStatusKey is the attribute Key conforming to the
	// "system.process.status" semantic conventions. It represents the process
	// state, e.g., [Linux Process State Codes].
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "running"
	//
	// [Linux Process State Codes]: https://man7.org/linux/man-pages/man1/ps.1.html#PROCESS_STATE_CODES
	SystemProcessStatusKey = attribute.Key("system.process.status")
)

// SystemCPULogicalNumber returns an attribute KeyValue conforming to the
// "system.cpu.logical_number" semantic conventions. It represents the
// deprecated, use `cpu.logical_number` instead.
func SystemCPULogicalNumber(val int) attribute.KeyValue {
	return SystemCPULogicalNumberKey.Int(val)
}

// SystemDevice returns an attribute KeyValue conforming to the "system.device"
// semantic conventions. It represents the device identifier.
func SystemDevice(val string) attribute.KeyValue {
	return SystemDeviceKey.String(val)
}

// SystemFilesystemMode returns an attribute KeyValue conforming to the
// "system.filesystem.mode" semantic conventions. It represents the filesystem
// mode.
func SystemFilesystemMode(val string) attribute.KeyValue {
	return SystemFilesystemModeKey.String(val)
}

// SystemFilesystemMountpoint returns an attribute KeyValue conforming to the
// "system.filesystem.mountpoint" semantic conventions. It represents the
// filesystem mount path.
func SystemFilesystemMountpoint(val string) attribute.KeyValue {
	return SystemFilesystemMountpointKey.String(val)
}

// Enum values for system.filesystem.state
var (
	// used
	// Stability: development
	SystemFilesystemStateUsed = SystemFilesystemStateKey.String("used")
	// free
	// Stability: development
	SystemFilesystemStateFree = SystemFilesystemStateKey.String("free")
	// reserved
	// Stability: development
	SystemFilesystemStateReserved = SystemFilesystemStateKey.String("reserved")
)

// Enum values for system.filesystem.type
var (
	// fat32
	// Stability: development
	SystemFilesystemTypeFat32 = SystemFilesystemTypeKey.String("fat32")
	// exfat
	// Stability: development
	SystemFilesystemTypeExfat = SystemFilesystemTypeKey.String("exfat")
	// ntfs
	// Stability: development
	SystemFilesystemTypeNtfs = SystemFilesystemTypeKey.String("ntfs")
	// refs
	// Stability: development
	SystemFilesystemTypeRefs = SystemFilesystemTypeKey.String("refs")
	// hfsplus
	// Stability: development
	SystemFilesystemTypeHfsplus = SystemFilesystemTypeKey.String("hfsplus")
	// ext4
	// Stability: development
	SystemFilesystemTypeExt4 = SystemFilesystemTypeKey.String("ext4")
)

// Enum values for system.memory.state
var (
	// Actual used virtual memory in bytes.
	// Stability: development
	SystemMemoryStateUsed = SystemMemoryStateKey.String("used")
	// free
	// Stability: development
	SystemMemoryStateFree = SystemMemoryStateKey.String("free")
	// buffers
	// Stability: development
	SystemMemoryStateBuffers = SystemMemoryStateKey.String("buffers")
	// cached
	// Stability: development
	SystemMemoryStateCached = SystemMemoryStateKey.String("cached")
)

// Enum values for system.paging.direction
var (
	// in
	// Stability: development
	SystemPagingDirectionIn = SystemPagingDirectionKey.String("in")
	// out
	// Stability: development
	SystemPagingDirectionOut = SystemPagingDirectionKey.String("out")
)

// Enum values for system.paging.state
var (
	// used
	// Stability: development
	SystemPagingStateUsed = SystemPagingStateKey.String("used")
	// free
	// Stability: development
	SystemPagingStateFree = SystemPagingStateKey.String("free")
)

// Enum values for system.paging.type
var (
	// major
	// Stability: development
	SystemPagingTypeMajor = SystemPagingTypeKey.String("major")
	// minor
	// Stability: development
	SystemPagingTypeMinor = SystemPagingTypeKey.String("minor")
)

// Enum values for system.process.status
var (
	// running
	// Stability: development
	SystemProcessStatusRunning = SystemProcessStatusKey.String("running")
	// sleeping
	// Stability: development
	SystemProcessStatusSleeping = SystemProcessStatusKey.String("sleeping")
	// stopped
	// Stability: development
	SystemProcessStatusStopped = SystemProcessStatusKey.String("stopped")
	// defunct
	// Stability: development
	SystemProcessStatusDefunct = SystemProcessStatusKey.String("defunct")
)

// Namespace: telemetry
const (
	// TelemetryDistroNameKey is the attribute Key conforming to the
	// "telemetry.distro.name" semantic conventions. It represents the name of the
	// auto instrumentation agent or distribution, if used.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "parts-unlimited-java"
	// Note: Official auto instrumentation agents and distributions SHOULD set the
	// `telemetry.distro.name` attribute to
	// a string starting with `opentelemetry-`, e.g.
	// `opentelemetry-java-instrumentation`.
	TelemetryDistroNameKey = attribute.Key("telemetry.distro.name")

	// TelemetryDistroVersionKey is the attribute Key conforming to the
	// "telemetry.distro.version" semantic conventions. It represents the version
	// string of the auto instrumentation agent or distribution, if used.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "1.2.3"
	TelemetryDistroVersionKey = attribute.Key("telemetry.distro.version")

	// TelemetrySDKLanguageKey is the attribute Key conforming to the
	// "telemetry.sdk.language" semantic conventions. It represents the language of
	// the telemetry SDK.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples:
	TelemetrySDKLanguageKey = attribute.Key("telemetry.sdk.language")

	// TelemetrySDKNameKey is the attribute Key conforming to the
	// "telemetry.sdk.name" semantic conventions. It represents the name of the
	// telemetry SDK as defined above.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "opentelemetry"
	// Note: The OpenTelemetry SDK MUST set the `telemetry.sdk.name` attribute to
	// `opentelemetry`.
	// If another SDK, like a fork or a vendor-provided implementation, is used,
	// this SDK MUST set the
	// `telemetry.sdk.name` attribute to the fully-qualified class or module name of
	// this SDK's main entry point
	// or another suitable identifier depending on the language.
	// The identifier `opentelemetry` is reserved and MUST NOT be used in this case.
	// All custom identifiers SHOULD be stable across different versions of an
	// implementation.
	TelemetrySDKNameKey = attribute.Key("telemetry.sdk.name")

	// TelemetrySDKVersionKey is the attribute Key conforming to the
	// "telemetry.sdk.version" semantic conventions. It represents the version
	// string of the telemetry SDK.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "1.2.3"
	TelemetrySDKVersionKey = attribute.Key("telemetry.sdk.version")
)

// TelemetryDistroName returns an attribute KeyValue conforming to the
// "telemetry.distro.name" semantic conventions. It represents the name of the
// auto instrumentation agent or distribution, if used.
func TelemetryDistroName(val string) attribute.KeyValue {
	return TelemetryDistroNameKey.String(val)
}

// TelemetryDistroVersion returns an attribute KeyValue conforming to the
// "telemetry.distro.version" semantic conventions. It represents the version
// string of the auto instrumentation agent or distribution, if used.
func TelemetryDistroVersion(val string) attribute.KeyValue {
	return TelemetryDistroVersionKey.String(val)
}

// TelemetrySDKName returns an attribute KeyValue conforming to the
// "telemetry.sdk.name" semantic conventions. It represents the name of the
// telemetry SDK as defined above.
func TelemetrySDKName(val string) attribute.KeyValue {
	return TelemetrySDKNameKey.String(val)
}

// TelemetrySDKVersion returns an attribute KeyValue conforming to the
// "telemetry.sdk.version" semantic conventions. It represents the version string
// of the telemetry SDK.
func TelemetrySDKVersion(val string) attribute.KeyValue {
	return TelemetrySDKVersionKey.String(val)
}

// Enum values for telemetry.sdk.language
var (
	// cpp
	// Stability: stable
	TelemetrySDKLanguageCPP = TelemetrySDKLanguageKey.String("cpp")
	// dotnet
	// Stability: stable
	TelemetrySDKLanguageDotnet = TelemetrySDKLanguageKey.String("dotnet")
	// erlang
	// Stability: stable
	TelemetrySDKLanguageErlang = TelemetrySDKLanguageKey.String("erlang")
	// go
	// Stability: stable
	TelemetrySDKLanguageGo = TelemetrySDKLanguageKey.String("go")
	// java
	// Stability: stable
	TelemetrySDKLanguageJava = TelemetrySDKLanguageKey.String("java")
	// nodejs
	// Stability: stable
	TelemetrySDKLanguageNodejs = TelemetrySDKLanguageKey.String("nodejs")
	// php
	// Stability: stable
	TelemetrySDKLanguagePHP = TelemetrySDKLanguageKey.String("php")
	// python
	// Stability: stable
	TelemetrySDKLanguagePython = TelemetrySDKLanguageKey.String("python")
	// ruby
	// Stability: stable
	TelemetrySDKLanguageRuby = TelemetrySDKLanguageKey.String("ruby")
	// rust
	// Stability: stable
	TelemetrySDKLanguageRust = TelemetrySDKLanguageKey.String("rust")
	// swift
	// Stability: stable
	TelemetrySDKLanguageSwift = TelemetrySDKLanguageKey.String("swift")
	// webjs
	// Stability: stable
	TelemetrySDKLanguageWebJS = TelemetrySDKLanguageKey.String("webjs")
)

// Namespace: test
const (
	// TestCaseNameKey is the attribute Key conforming to the "test.case.name"
	// semantic conventions. It represents the fully qualified human readable name
	// of the [test case].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "org.example.TestCase1.test1", "example/tests/TestCase1.test1",
	// "ExampleTestCase1_test1"
	//
	// [test case]: https://wikipedia.org/wiki/Test_case
	TestCaseNameKey = attribute.Key("test.case.name")

	// TestCaseResultStatusKey is the attribute Key conforming to the
	// "test.case.result.status" semantic conventions. It represents the status of
	// the actual test case result from test execution.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "pass", "fail"
	TestCaseResultStatusKey = attribute.Key("test.case.result.status")

	// TestSuiteNameKey is the attribute Key conforming to the "test.suite.name"
	// semantic conventions. It represents the human readable name of a [test suite]
	// .
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "TestSuite1"
	//
	// [test suite]: https://wikipedia.org/wiki/Test_suite
	TestSuiteNameKey = attribute.Key("test.suite.name")

	// TestSuiteRunStatusKey is the attribute Key conforming to the
	// "test.suite.run.status" semantic conventions. It represents the status of the
	// test suite run.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "success", "failure", "skipped", "aborted", "timed_out",
	// "in_progress"
	TestSuiteRunStatusKey = attribute.Key("test.suite.run.status")
)

// TestCaseName returns an attribute KeyValue conforming to the "test.case.name"
// semantic conventions. It represents the fully qualified human readable name of
// the [test case].
//
// [test case]: https://wikipedia.org/wiki/Test_case
func TestCaseName(val string) attribute.KeyValue {
	return TestCaseNameKey.String(val)
}

// TestSuiteName returns an attribute KeyValue conforming to the
// "test.suite.name" semantic conventions. It represents the human readable name
// of a [test suite].
//
// [test suite]: https://wikipedia.org/wiki/Test_suite
func TestSuiteName(val string) attribute.KeyValue {
	return TestSuiteNameKey.String(val)
}

// Enum values for test.case.result.status
var (
	// pass
	// Stability: development
	TestCaseResultStatusPass = TestCaseResultStatusKey.String("pass")
	// fail
	// Stability: development
	TestCaseResultStatusFail = TestCaseResultStatusKey.String("fail")
)

// Enum values for test.suite.run.status
var (
	// success
	// Stability: development
	TestSuiteRunStatusSuccess = TestSuiteRunStatusKey.String("success")
	// failure
	// Stability: development
	TestSuiteRunStatusFailure = TestSuiteRunStatusKey.String("failure")
	// skipped
	// Stability: development
	TestSuiteRunStatusSkipped = TestSuiteRunStatusKey.String("skipped")
	// aborted
	// Stability: development
	TestSuiteRunStatusAborted = TestSuiteRunStatusKey.String("aborted")
	// timed_out
	// Stability: development
	TestSuiteRunStatusTimedOut = TestSuiteRunStatusKey.String("timed_out")
	// in_progress
	// Stability: development
	TestSuiteRunStatusInProgress = TestSuiteRunStatusKey.String("in_progress")
)

// Namespace: thread
const (
	// ThreadIDKey is the attribute Key conforming to the "thread.id" semantic
	// conventions. It represents the current "managed" thread ID (as opposed to OS
	// thread ID).
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	ThreadIDKey = attribute.Key("thread.id")

	// ThreadNameKey is the attribute Key conforming to the "thread.name" semantic
	// conventions. It represents the current thread name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: main
	ThreadNameKey = attribute.Key("thread.name")
)

// ThreadID returns an attribute KeyValue conforming to the "thread.id" semantic
// conventions. It represents the current "managed" thread ID (as opposed to OS
// thread ID).
func ThreadID(val int) attribute.KeyValue {
	return ThreadIDKey.Int(val)
}

// ThreadName returns an attribute KeyValue conforming to the "thread.name"
// semantic conventions. It represents the current thread name.
func ThreadName(val string) attribute.KeyValue {
	return ThreadNameKey.String(val)
}

// Namespace: tls
const (
	// TLSCipherKey is the attribute Key conforming to the "tls.cipher" semantic
	// conventions. It represents the string indicating the [cipher] used during the
	// current connection.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "TLS_RSA_WITH_3DES_EDE_CBC_SHA",
	// "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256"
	// Note: The values allowed for `tls.cipher` MUST be one of the `Descriptions`
	// of the [registered TLS Cipher Suits].
	//
	// [cipher]: https://datatracker.ietf.org/doc/html/rfc5246#appendix-A.5
	// [registered TLS Cipher Suits]: https://www.iana.org/assignments/tls-parameters/tls-parameters.xhtml#table-tls-parameters-4
	TLSCipherKey = attribute.Key("tls.cipher")

	// TLSClientCertificateKey is the attribute Key conforming to the
	// "tls.client.certificate" semantic conventions. It represents the PEM-encoded
	// stand-alone certificate offered by the client. This is usually
	// mutually-exclusive of `client.certificate_chain` since this value also exists
	// in that list.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "MII..."
	TLSClientCertificateKey = attribute.Key("tls.client.certificate")

	// TLSClientCertificateChainKey is the attribute Key conforming to the
	// "tls.client.certificate_chain" semantic conventions. It represents the array
	// of PEM-encoded certificates that make up the certificate chain offered by the
	// client. This is usually mutually-exclusive of `client.certificate` since that
	// value should be the first certificate in the chain.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "MII...", "MI..."
	TLSClientCertificateChainKey = attribute.Key("tls.client.certificate_chain")

	// TLSClientHashMd5Key is the attribute Key conforming to the
	// "tls.client.hash.md5" semantic conventions. It represents the certificate
	// fingerprint using the MD5 digest of DER-encoded version of certificate
	// offered by the client. For consistency with other hash values, this value
	// should be formatted as an uppercase hash.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "0F76C7F2C55BFD7D8E8B8F4BFBF0C9EC"
	TLSClientHashMd5Key = attribute.Key("tls.client.hash.md5")

	// TLSClientHashSha1Key is the attribute Key conforming to the
	// "tls.client.hash.sha1" semantic conventions. It represents the certificate
	// fingerprint using the SHA1 digest of DER-encoded version of certificate
	// offered by the client. For consistency with other hash values, this value
	// should be formatted as an uppercase hash.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "9E393D93138888D288266C2D915214D1D1CCEB2A"
	TLSClientHashSha1Key = attribute.Key("tls.client.hash.sha1")

	// TLSClientHashSha256Key is the attribute Key conforming to the
	// "tls.client.hash.sha256" semantic conventions. It represents the certificate
	// fingerprint using the SHA256 digest of DER-encoded version of certificate
	// offered by the client. For consistency with other hash values, this value
	// should be formatted as an uppercase hash.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "0687F666A054EF17A08E2F2162EAB4CBC0D265E1D7875BE74BF3C712CA92DAF0"
	TLSClientHashSha256Key = attribute.Key("tls.client.hash.sha256")

	// TLSClientIssuerKey is the attribute Key conforming to the "tls.client.issuer"
	// semantic conventions. It represents the distinguished name of [subject] of
	// the issuer of the x.509 certificate presented by the client.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "CN=Example Root CA, OU=Infrastructure Team, DC=example, DC=com"
	//
	// [subject]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.6
	TLSClientIssuerKey = attribute.Key("tls.client.issuer")

	// TLSClientJa3Key is the attribute Key conforming to the "tls.client.ja3"
	// semantic conventions. It represents a hash that identifies clients based on
	// how they perform an SSL/TLS handshake.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "d4e5b18d6b55c71272893221c96ba240"
	TLSClientJa3Key = attribute.Key("tls.client.ja3")

	// TLSClientNotAfterKey is the attribute Key conforming to the
	// "tls.client.not_after" semantic conventions. It represents the date/Time
	// indicating when client certificate is no longer considered valid.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2021-01-01T00:00:00.000Z"
	TLSClientNotAfterKey = attribute.Key("tls.client.not_after")

	// TLSClientNotBeforeKey is the attribute Key conforming to the
	// "tls.client.not_before" semantic conventions. It represents the date/Time
	// indicating when client certificate is first considered valid.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "1970-01-01T00:00:00.000Z"
	TLSClientNotBeforeKey = attribute.Key("tls.client.not_before")

	// TLSClientSubjectKey is the attribute Key conforming to the
	// "tls.client.subject" semantic conventions. It represents the distinguished
	// name of subject of the x.509 certificate presented by the client.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "CN=myclient, OU=Documentation Team, DC=example, DC=com"
	TLSClientSubjectKey = attribute.Key("tls.client.subject")

	// TLSClientSupportedCiphersKey is the attribute Key conforming to the
	// "tls.client.supported_ciphers" semantic conventions. It represents the array
	// of ciphers offered by the client during the client hello.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
	// "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384"
	TLSClientSupportedCiphersKey = attribute.Key("tls.client.supported_ciphers")

	// TLSCurveKey is the attribute Key conforming to the "tls.curve" semantic
	// conventions. It represents the string indicating the curve used for the given
	// cipher, when applicable.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "secp256r1"
	TLSCurveKey = attribute.Key("tls.curve")

	// TLSEstablishedKey is the attribute Key conforming to the "tls.established"
	// semantic conventions. It represents the boolean flag indicating if the TLS
	// negotiation was successful and transitioned to an encrypted tunnel.
	//
	// Type: boolean
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: true
	TLSEstablishedKey = attribute.Key("tls.established")

	// TLSNextProtocolKey is the attribute Key conforming to the "tls.next_protocol"
	// semantic conventions. It represents the string indicating the protocol being
	// tunneled. Per the values in the [IANA registry], this string should be lower
	// case.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "http/1.1"
	//
	// [IANA registry]: https://www.iana.org/assignments/tls-extensiontype-values/tls-extensiontype-values.xhtml#alpn-protocol-ids
	TLSNextProtocolKey = attribute.Key("tls.next_protocol")

	// TLSProtocolNameKey is the attribute Key conforming to the "tls.protocol.name"
	// semantic conventions. It represents the normalized lowercase protocol name
	// parsed from original string of the negotiated [SSL/TLS protocol version].
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	//
	// [SSL/TLS protocol version]: https://docs.openssl.org/1.1.1/man3/SSL_get_version/#return-values
	TLSProtocolNameKey = attribute.Key("tls.protocol.name")

	// TLSProtocolVersionKey is the attribute Key conforming to the
	// "tls.protocol.version" semantic conventions. It represents the numeric part
	// of the version parsed from the original string of the negotiated
	// [SSL/TLS protocol version].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "1.2", "3"
	//
	// [SSL/TLS protocol version]: https://docs.openssl.org/1.1.1/man3/SSL_get_version/#return-values
	TLSProtocolVersionKey = attribute.Key("tls.protocol.version")

	// TLSResumedKey is the attribute Key conforming to the "tls.resumed" semantic
	// conventions. It represents the boolean flag indicating if this TLS connection
	// was resumed from an existing TLS negotiation.
	//
	// Type: boolean
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: true
	TLSResumedKey = attribute.Key("tls.resumed")

	// TLSServerCertificateKey is the attribute Key conforming to the
	// "tls.server.certificate" semantic conventions. It represents the PEM-encoded
	// stand-alone certificate offered by the server. This is usually
	// mutually-exclusive of `server.certificate_chain` since this value also exists
	// in that list.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "MII..."
	TLSServerCertificateKey = attribute.Key("tls.server.certificate")

	// TLSServerCertificateChainKey is the attribute Key conforming to the
	// "tls.server.certificate_chain" semantic conventions. It represents the array
	// of PEM-encoded certificates that make up the certificate chain offered by the
	// server. This is usually mutually-exclusive of `server.certificate` since that
	// value should be the first certificate in the chain.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "MII...", "MI..."
	TLSServerCertificateChainKey = attribute.Key("tls.server.certificate_chain")

	// TLSServerHashMd5Key is the attribute Key conforming to the
	// "tls.server.hash.md5" semantic conventions. It represents the certificate
	// fingerprint using the MD5 digest of DER-encoded version of certificate
	// offered by the server. For consistency with other hash values, this value
	// should be formatted as an uppercase hash.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "0F76C7F2C55BFD7D8E8B8F4BFBF0C9EC"
	TLSServerHashMd5Key = attribute.Key("tls.server.hash.md5")

	// TLSServerHashSha1Key is the attribute Key conforming to the
	// "tls.server.hash.sha1" semantic conventions. It represents the certificate
	// fingerprint using the SHA1 digest of DER-encoded version of certificate
	// offered by the server. For consistency with other hash values, this value
	// should be formatted as an uppercase hash.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "9E393D93138888D288266C2D915214D1D1CCEB2A"
	TLSServerHashSha1Key = attribute.Key("tls.server.hash.sha1")

	// TLSServerHashSha256Key is the attribute Key conforming to the
	// "tls.server.hash.sha256" semantic conventions. It represents the certificate
	// fingerprint using the SHA256 digest of DER-encoded version of certificate
	// offered by the server. For consistency with other hash values, this value
	// should be formatted as an uppercase hash.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "0687F666A054EF17A08E2F2162EAB4CBC0D265E1D7875BE74BF3C712CA92DAF0"
	TLSServerHashSha256Key = attribute.Key("tls.server.hash.sha256")

	// TLSServerIssuerKey is the attribute Key conforming to the "tls.server.issuer"
	// semantic conventions. It represents the distinguished name of [subject] of
	// the issuer of the x.509 certificate presented by the client.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "CN=Example Root CA, OU=Infrastructure Team, DC=example, DC=com"
	//
	// [subject]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.6
	TLSServerIssuerKey = attribute.Key("tls.server.issuer")

	// TLSServerJa3sKey is the attribute Key conforming to the "tls.server.ja3s"
	// semantic conventions. It represents a hash that identifies servers based on
	// how they perform an SSL/TLS handshake.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "d4e5b18d6b55c71272893221c96ba240"
	TLSServerJa3sKey = attribute.Key("tls.server.ja3s")

	// TLSServerNotAfterKey is the attribute Key conforming to the
	// "tls.server.not_after" semantic conventions. It represents the date/Time
	// indicating when server certificate is no longer considered valid.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "2021-01-01T00:00:00.000Z"
	TLSServerNotAfterKey = attribute.Key("tls.server.not_after")

	// TLSServerNotBeforeKey is the attribute Key conforming to the
	// "tls.server.not_before" semantic conventions. It represents the date/Time
	// indicating when server certificate is first considered valid.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "1970-01-01T00:00:00.000Z"
	TLSServerNotBeforeKey = attribute.Key("tls.server.not_before")

	// TLSServerSubjectKey is the attribute Key conforming to the
	// "tls.server.subject" semantic conventions. It represents the distinguished
	// name of subject of the x.509 certificate presented by the server.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "CN=myserver, OU=Documentation Team, DC=example, DC=com"
	TLSServerSubjectKey = attribute.Key("tls.server.subject")
)

// TLSCipher returns an attribute KeyValue conforming to the "tls.cipher"
// semantic conventions. It represents the string indicating the [cipher] used
// during the current connection.
//
// [cipher]: https://datatracker.ietf.org/doc/html/rfc5246#appendix-A.5
func TLSCipher(val string) attribute.KeyValue {
	return TLSCipherKey.String(val)
}

// TLSClientCertificate returns an attribute KeyValue conforming to the
// "tls.client.certificate" semantic conventions. It represents the PEM-encoded
// stand-alone certificate offered by the client. This is usually
// mutually-exclusive of `client.certificate_chain` since this value also exists
// in that list.
func TLSClientCertificate(val string) attribute.KeyValue {
	return TLSClientCertificateKey.String(val)
}

// TLSClientCertificateChain returns an attribute KeyValue conforming to the
// "tls.client.certificate_chain" semantic conventions. It represents the array
// of PEM-encoded certificates that make up the certificate chain offered by the
// client. This is usually mutually-exclusive of `client.certificate` since that
// value should be the first certificate in the chain.
func TLSClientCertificateChain(val ...string) attribute.KeyValue {
	return TLSClientCertificateChainKey.StringSlice(val)
}

// TLSClientHashMd5 returns an attribute KeyValue conforming to the
// "tls.client.hash.md5" semantic conventions. It represents the certificate
// fingerprint using the MD5 digest of DER-encoded version of certificate offered
// by the client. For consistency with other hash values, this value should be
// formatted as an uppercase hash.
func TLSClientHashMd5(val string) attribute.KeyValue {
	return TLSClientHashMd5Key.String(val)
}

// TLSClientHashSha1 returns an attribute KeyValue conforming to the
// "tls.client.hash.sha1" semantic conventions. It represents the certificate
// fingerprint using the SHA1 digest of DER-encoded version of certificate
// offered by the client. For consistency with other hash values, this value
// should be formatted as an uppercase hash.
func TLSClientHashSha1(val string) attribute.KeyValue {
	return TLSClientHashSha1Key.String(val)
}

// TLSClientHashSha256 returns an attribute KeyValue conforming to the
// "tls.client.hash.sha256" semantic conventions. It represents the certificate
// fingerprint using the SHA256 digest of DER-encoded version of certificate
// offered by the client. For consistency with other hash values, this value
// should be formatted as an uppercase hash.
func TLSClientHashSha256(val string) attribute.KeyValue {
	return TLSClientHashSha256Key.String(val)
}

// TLSClientIssuer returns an attribute KeyValue conforming to the
// "tls.client.issuer" semantic conventions. It represents the distinguished name
// of [subject] of the issuer of the x.509 certificate presented by the client.
//
// [subject]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.6
func TLSClientIssuer(val string) attribute.KeyValue {
	return TLSClientIssuerKey.String(val)
}

// TLSClientJa3 returns an attribute KeyValue conforming to the "tls.client.ja3"
// semantic conventions. It represents a hash that identifies clients based on
// how they perform an SSL/TLS handshake.
func TLSClientJa3(val string) attribute.KeyValue {
	return TLSClientJa3Key.String(val)
}

// TLSClientNotAfter returns an attribute KeyValue conforming to the
// "tls.client.not_after" semantic conventions. It represents the date/Time
// indicating when client certificate is no longer considered valid.
func TLSClientNotAfter(val string) attribute.KeyValue {
	return TLSClientNotAfterKey.String(val)
}

// TLSClientNotBefore returns an attribute KeyValue conforming to the
// "tls.client.not_before" semantic conventions. It represents the date/Time
// indicating when client certificate is first considered valid.
func TLSClientNotBefore(val string) attribute.KeyValue {
	return TLSClientNotBeforeKey.String(val)
}

// TLSClientSubject returns an attribute KeyValue conforming to the
// "tls.client.subject" semantic conventions. It represents the distinguished
// name of subject of the x.509 certificate presented by the client.
func TLSClientSubject(val string) attribute.KeyValue {
	return TLSClientSubjectKey.String(val)
}

// TLSClientSupportedCiphers returns an attribute KeyValue conforming to the
// "tls.client.supported_ciphers" semantic conventions. It represents the array
// of ciphers offered by the client during the client hello.
func TLSClientSupportedCiphers(val ...string) attribute.KeyValue {
	return TLSClientSupportedCiphersKey.StringSlice(val)
}

// TLSCurve returns an attribute KeyValue conforming to the "tls.curve" semantic
// conventions. It represents the string indicating the curve used for the given
// cipher, when applicable.
func TLSCurve(val string) attribute.KeyValue {
	return TLSCurveKey.String(val)
}

// TLSEstablished returns an attribute KeyValue conforming to the
// "tls.established" semantic conventions. It represents the boolean flag
// indicating if the TLS negotiation was successful and transitioned to an
// encrypted tunnel.
func TLSEstablished(val bool) attribute.KeyValue {
	return TLSEstablishedKey.Bool(val)
}

// TLSNextProtocol returns an attribute KeyValue conforming to the
// "tls.next_protocol" semantic conventions. It represents the string indicating
// the protocol being tunneled. Per the values in the [IANA registry], this
// string should be lower case.
//
// [IANA registry]: https://www.iana.org/assignments/tls-extensiontype-values/tls-extensiontype-values.xhtml#alpn-protocol-ids
func TLSNextProtocol(val string) attribute.KeyValue {
	return TLSNextProtocolKey.String(val)
}

// TLSProtocolVersion returns an attribute KeyValue conforming to the
// "tls.protocol.version" semantic conventions. It represents the numeric part of
// the version parsed from the original string of the negotiated
// [SSL/TLS protocol version].
//
// [SSL/TLS protocol version]: https://docs.openssl.org/1.1.1/man3/SSL_get_version/#return-values
func TLSProtocolVersion(val string) attribute.KeyValue {
	return TLSProtocolVersionKey.String(val)
}

// TLSResumed returns an attribute KeyValue conforming to the "tls.resumed"
// semantic conventions. It represents the boolean flag indicating if this TLS
// connection was resumed from an existing TLS negotiation.
func TLSResumed(val bool) attribute.KeyValue {
	return TLSResumedKey.Bool(val)
}

// TLSServerCertificate returns an attribute KeyValue conforming to the
// "tls.server.certificate" semantic conventions. It represents the PEM-encoded
// stand-alone certificate offered by the server. This is usually
// mutually-exclusive of `server.certificate_chain` since this value also exists
// in that list.
func TLSServerCertificate(val string) attribute.KeyValue {
	return TLSServerCertificateKey.String(val)
}

// TLSServerCertificateChain returns an attribute KeyValue conforming to the
// "tls.server.certificate_chain" semantic conventions. It represents the array
// of PEM-encoded certificates that make up the certificate chain offered by the
// server. This is usually mutually-exclusive of `server.certificate` since that
// value should be the first certificate in the chain.
func TLSServerCertificateChain(val ...string) attribute.KeyValue {
	return TLSServerCertificateChainKey.StringSlice(val)
}

// TLSServerHashMd5 returns an attribute KeyValue conforming to the
// "tls.server.hash.md5" semantic conventions. It represents the certificate
// fingerprint using the MD5 digest of DER-encoded version of certificate offered
// by the server. For consistency with other hash values, this value should be
// formatted as an uppercase hash.
func TLSServerHashMd5(val string) attribute.KeyValue {
	return TLSServerHashMd5Key.String(val)
}

// TLSServerHashSha1 returns an attribute KeyValue conforming to the
// "tls.server.hash.sha1" semantic conventions. It represents the certificate
// fingerprint using the SHA1 digest of DER-encoded version of certificate
// offered by the server. For consistency with other hash values, this value
// should be formatted as an uppercase hash.
func TLSServerHashSha1(val string) attribute.KeyValue {
	return TLSServerHashSha1Key.String(val)
}

// TLSServerHashSha256 returns an attribute KeyValue conforming to the
// "tls.server.hash.sha256" semantic conventions. It represents the certificate
// fingerprint using the SHA256 digest of DER-encoded version of certificate
// offered by the server. For consistency with other hash values, this value
// should be formatted as an uppercase hash.
func TLSServerHashSha256(val string) attribute.KeyValue {
	return TLSServerHashSha256Key.String(val)
}

// TLSServerIssuer returns an attribute KeyValue conforming to the
// "tls.server.issuer" semantic conventions. It represents the distinguished name
// of [subject] of the issuer of the x.509 certificate presented by the client.
//
// [subject]: https://datatracker.ietf.org/doc/html/rfc5280#section-4.1.2.6
func TLSServerIssuer(val string) attribute.KeyValue {
	return TLSServerIssuerKey.String(val)
}

// TLSServerJa3s returns an attribute KeyValue conforming to the
// "tls.server.ja3s" semantic conventions. It represents a hash that identifies
// servers based on how they perform an SSL/TLS handshake.
func TLSServerJa3s(val string) attribute.KeyValue {
	return TLSServerJa3sKey.String(val)
}

// TLSServerNotAfter returns an attribute KeyValue conforming to the
// "tls.server.not_after" semantic conventions. It represents the date/Time
// indicating when server certificate is no longer considered valid.
func TLSServerNotAfter(val string) attribute.KeyValue {
	return TLSServerNotAfterKey.String(val)
}

// TLSServerNotBefore returns an attribute KeyValue conforming to the
// "tls.server.not_before" semantic conventions. It represents the date/Time
// indicating when server certificate is first considered valid.
func TLSServerNotBefore(val string) attribute.KeyValue {
	return TLSServerNotBeforeKey.String(val)
}

// TLSServerSubject returns an attribute KeyValue conforming to the
// "tls.server.subject" semantic conventions. It represents the distinguished
// name of subject of the x.509 certificate presented by the server.
func TLSServerSubject(val string) attribute.KeyValue {
	return TLSServerSubjectKey.String(val)
}

// Enum values for tls.protocol.name
var (
	// ssl
	// Stability: development
	TLSProtocolNameSsl = TLSProtocolNameKey.String("ssl")
	// tls
	// Stability: development
	TLSProtocolNameTLS = TLSProtocolNameKey.String("tls")
)

// Namespace: url
const (
	// URLDomainKey is the attribute Key conforming to the "url.domain" semantic
	// conventions. It represents the domain extracted from the `url.full`, such as
	// "opentelemetry.io".
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "www.foo.bar", "opentelemetry.io", "3.12.167.2",
	// "[1080:0:0:0:8:800:200C:417A]"
	// Note: In some cases a URL may refer to an IP and/or port directly, without a
	// domain name. In this case, the IP address would go to the domain field. If
	// the URL contains a [literal IPv6 address] enclosed by `[` and `]`, the `[`
	// and `]` characters should also be captured in the domain field.
	//
	// [literal IPv6 address]: https://www.rfc-editor.org/rfc/rfc2732#section-2
	URLDomainKey = attribute.Key("url.domain")

	// URLExtensionKey is the attribute Key conforming to the "url.extension"
	// semantic conventions. It represents the file extension extracted from the
	// `url.full`, excluding the leading dot.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "png", "gz"
	// Note: The file extension is only set if it exists, as not every url has a
	// file extension. When the file name has multiple extensions `example.tar.gz`,
	// only the last one should be captured `gz`, not `tar.gz`.
	URLExtensionKey = attribute.Key("url.extension")

	// URLFragmentKey is the attribute Key conforming to the "url.fragment" semantic
	// conventions. It represents the [URI fragment] component.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "SemConv"
	//
	// [URI fragment]: https://www.rfc-editor.org/rfc/rfc3986#section-3.5
	URLFragmentKey = attribute.Key("url.fragment")

	// URLFullKey is the attribute Key conforming to the "url.full" semantic
	// conventions. It represents the absolute URL describing a network resource
	// according to [RFC3986].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "https://www.foo.bar/search?q=OpenTelemetry#SemConv", "//localhost"
	// Note: For network calls, URL usually has
	// `scheme://host[:port][path][?query][#fragment]` format, where the fragment
	// is not transmitted over HTTP, but if it is known, it SHOULD be included
	// nevertheless.
	//
	// `url.full` MUST NOT contain credentials passed via URL in form of
	// `https://username:password@www.example.com/`.
	// In such case username and password SHOULD be redacted and attribute's value
	// SHOULD be `https://REDACTED:REDACTED@www.example.com/`.
	//
	// `url.full` SHOULD capture the absolute URL when it is available (or can be
	// reconstructed).
	//
	// Sensitive content provided in `url.full` SHOULD be scrubbed when
	// instrumentations can identify it.
	//
	//
	// Query string values for the following keys SHOULD be redacted by default and
	// replaced by the
	// value `REDACTED`:
	//
	//   - [`AWSAccessKeyId`]
	//   - [`Signature`]
	//   - [`sig`]
	//   - [`X-Goog-Signature`]
	//
	// This list is subject to change over time.
	//
	// When a query string value is redacted, the query string key SHOULD still be
	// preserved, e.g.
	// `https://www.example.com/path?color=blue&sig=REDACTED`.
	//
	// [RFC3986]: https://www.rfc-editor.org/rfc/rfc3986
	// [`AWSAccessKeyId`]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/RESTAuthentication.html#RESTAuthenticationQueryStringAuth
	// [`Signature`]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/RESTAuthentication.html#RESTAuthenticationQueryStringAuth
	// [`sig`]: https://learn.microsoft.com/azure/storage/common/storage-sas-overview#sas-token
	// [`X-Goog-Signature`]: https://cloud.google.com/storage/docs/access-control/signed-urls
	URLFullKey = attribute.Key("url.full")

	// URLOriginalKey is the attribute Key conforming to the "url.original" semantic
	// conventions. It represents the unmodified original URL as seen in the event
	// source.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "https://www.foo.bar/search?q=OpenTelemetry#SemConv",
	// "search?q=OpenTelemetry"
	// Note: In network monitoring, the observed URL may be a full URL, whereas in
	// access logs, the URL is often just represented as a path. This field is meant
	// to represent the URL as it was observed, complete or not.
	// `url.original` might contain credentials passed via URL in form of
	// `https://username:password@www.example.com/`. In such case password and
	// username SHOULD NOT be redacted and attribute's value SHOULD remain the same.
	URLOriginalKey = attribute.Key("url.original")

	// URLPathKey is the attribute Key conforming to the "url.path" semantic
	// conventions. It represents the [URI path] component.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "/search"
	// Note: Sensitive content provided in `url.path` SHOULD be scrubbed when
	// instrumentations can identify it.
	//
	// [URI path]: https://www.rfc-editor.org/rfc/rfc3986#section-3.3
	URLPathKey = attribute.Key("url.path")

	// URLPortKey is the attribute Key conforming to the "url.port" semantic
	// conventions. It represents the port extracted from the `url.full`.
	//
	// Type: int
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: 443
	URLPortKey = attribute.Key("url.port")

	// URLQueryKey is the attribute Key conforming to the "url.query" semantic
	// conventions. It represents the [URI query] component.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "q=OpenTelemetry"
	// Note: Sensitive content provided in `url.query` SHOULD be scrubbed when
	// instrumentations can identify it.
	//
	//
	// Query string values for the following keys SHOULD be redacted by default and
	// replaced by the value `REDACTED`:
	//
	//   - [`AWSAccessKeyId`]
	//   - [`Signature`]
	//   - [`sig`]
	//   - [`X-Goog-Signature`]
	//
	// This list is subject to change over time.
	//
	// When a query string value is redacted, the query string key SHOULD still be
	// preserved, e.g.
	// `q=OpenTelemetry&sig=REDACTED`.
	//
	// [URI query]: https://www.rfc-editor.org/rfc/rfc3986#section-3.4
	// [`AWSAccessKeyId`]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/RESTAuthentication.html#RESTAuthenticationQueryStringAuth
	// [`Signature`]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/RESTAuthentication.html#RESTAuthenticationQueryStringAuth
	// [`sig`]: https://learn.microsoft.com/azure/storage/common/storage-sas-overview#sas-token
	// [`X-Goog-Signature`]: https://cloud.google.com/storage/docs/access-control/signed-urls
	URLQueryKey = attribute.Key("url.query")

	// URLRegisteredDomainKey is the attribute Key conforming to the
	// "url.registered_domain" semantic conventions. It represents the highest
	// registered url domain, stripped of the subdomain.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "example.com", "foo.co.uk"
	// Note: This value can be determined precisely with the [public suffix list].
	// For example, the registered domain for `foo.example.com` is `example.com`.
	// Trying to approximate this by simply taking the last two labels will not work
	// well for TLDs such as `co.uk`.
	//
	// [public suffix list]: https://publicsuffix.org/
	URLRegisteredDomainKey = attribute.Key("url.registered_domain")

	// URLSchemeKey is the attribute Key conforming to the "url.scheme" semantic
	// conventions. It represents the [URI scheme] component identifying the used
	// protocol.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "https", "ftp", "telnet"
	//
	// [URI scheme]: https://www.rfc-editor.org/rfc/rfc3986#section-3.1
	URLSchemeKey = attribute.Key("url.scheme")

	// URLSubdomainKey is the attribute Key conforming to the "url.subdomain"
	// semantic conventions. It represents the subdomain portion of a fully
	// qualified domain name includes all of the names except the host name under
	// the registered_domain. In a partially qualified domain, or if the
	// qualification level of the full name cannot be determined, subdomain contains
	// all of the names below the registered domain.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "east", "sub2.sub1"
	// Note: The subdomain portion of `www.east.mydomain.co.uk` is `east`. If the
	// domain has multiple levels of subdomain, such as `sub2.sub1.example.com`, the
	// subdomain field should contain `sub2.sub1`, with no trailing period.
	URLSubdomainKey = attribute.Key("url.subdomain")

	// URLTemplateKey is the attribute Key conforming to the "url.template" semantic
	// conventions. It represents the low-cardinality template of an
	// [absolute path reference].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "/users/{id}", "/users/:id", "/users?id={id}"
	//
	// [absolute path reference]: https://www.rfc-editor.org/rfc/rfc3986#section-4.2
	URLTemplateKey = attribute.Key("url.template")

	// URLTopLevelDomainKey is the attribute Key conforming to the
	// "url.top_level_domain" semantic conventions. It represents the effective top
	// level domain (eTLD), also known as the domain suffix, is the last part of the
	// domain name. For example, the top level domain for example.com is `com`.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "com", "co.uk"
	// Note: This value can be determined precisely with the [public suffix list].
	//
	// [public suffix list]: https://publicsuffix.org/
	URLTopLevelDomainKey = attribute.Key("url.top_level_domain")
)

// URLDomain returns an attribute KeyValue conforming to the "url.domain"
// semantic conventions. It represents the domain extracted from the `url.full`,
// such as "opentelemetry.io".
func URLDomain(val string) attribute.KeyValue {
	return URLDomainKey.String(val)
}

// URLExtension returns an attribute KeyValue conforming to the "url.extension"
// semantic conventions. It represents the file extension extracted from the
// `url.full`, excluding the leading dot.
func URLExtension(val string) attribute.KeyValue {
	return URLExtensionKey.String(val)
}

// URLFragment returns an attribute KeyValue conforming to the "url.fragment"
// semantic conventions. It represents the [URI fragment] component.
//
// [URI fragment]: https://www.rfc-editor.org/rfc/rfc3986#section-3.5
func URLFragment(val string) attribute.KeyValue {
	return URLFragmentKey.String(val)
}

// URLFull returns an attribute KeyValue conforming to the "url.full" semantic
// conventions. It represents the absolute URL describing a network resource
// according to [RFC3986].
//
// [RFC3986]: https://www.rfc-editor.org/rfc/rfc3986
func URLFull(val string) attribute.KeyValue {
	return URLFullKey.String(val)
}

// URLOriginal returns an attribute KeyValue conforming to the "url.original"
// semantic conventions. It represents the unmodified original URL as seen in the
// event source.
func URLOriginal(val string) attribute.KeyValue {
	return URLOriginalKey.String(val)
}

// URLPath returns an attribute KeyValue conforming to the "url.path" semantic
// conventions. It represents the [URI path] component.
//
// [URI path]: https://www.rfc-editor.org/rfc/rfc3986#section-3.3
func URLPath(val string) attribute.KeyValue {
	return URLPathKey.String(val)
}

// URLPort returns an attribute KeyValue conforming to the "url.port" semantic
// conventions. It represents the port extracted from the `url.full`.
func URLPort(val int) attribute.KeyValue {
	return URLPortKey.Int(val)
}

// URLQuery returns an attribute KeyValue conforming to the "url.query" semantic
// conventions. It represents the [URI query] component.
//
// [URI query]: https://www.rfc-editor.org/rfc/rfc3986#section-3.4
func URLQuery(val string) attribute.KeyValue {
	return URLQueryKey.String(val)
}

// URLRegisteredDomain returns an attribute KeyValue conforming to the
// "url.registered_domain" semantic conventions. It represents the highest
// registered url domain, stripped of the subdomain.
func URLRegisteredDomain(val string) attribute.KeyValue {
	return URLRegisteredDomainKey.String(val)
}

// URLScheme returns an attribute KeyValue conforming to the "url.scheme"
// semantic conventions. It represents the [URI scheme] component identifying the
// used protocol.
//
// [URI scheme]: https://www.rfc-editor.org/rfc/rfc3986#section-3.1
func URLScheme(val string) attribute.KeyValue {
	return URLSchemeKey.String(val)
}

// URLSubdomain returns an attribute KeyValue conforming to the "url.subdomain"
// semantic conventions. It represents the subdomain portion of a fully qualified
// domain name includes all of the names except the host name under the
// registered_domain. In a partially qualified domain, or if the qualification
// level of the full name cannot be determined, subdomain contains all of the
// names below the registered domain.
func URLSubdomain(val string) attribute.KeyValue {
	return URLSubdomainKey.String(val)
}

// URLTemplate returns an attribute KeyValue conforming to the "url.template"
// semantic conventions. It represents the low-cardinality template of an
// [absolute path reference].
//
// [absolute path reference]: https://www.rfc-editor.org/rfc/rfc3986#section-4.2
func URLTemplate(val string) attribute.KeyValue {
	return URLTemplateKey.String(val)
}

// URLTopLevelDomain returns an attribute KeyValue conforming to the
// "url.top_level_domain" semantic conventions. It represents the effective top
// level domain (eTLD), also known as the domain suffix, is the last part of the
// domain name. For example, the top level domain for example.com is `com`.
func URLTopLevelDomain(val string) attribute.KeyValue {
	return URLTopLevelDomainKey.String(val)
}

// Namespace: user
const (
	// UserEmailKey is the attribute Key conforming to the "user.email" semantic
	// conventions. It represents the user email address.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "a.einstein@example.com"
	UserEmailKey = attribute.Key("user.email")

	// UserFullNameKey is the attribute Key conforming to the "user.full_name"
	// semantic conventions. It represents the user's full name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Albert Einstein"
	UserFullNameKey = attribute.Key("user.full_name")

	// UserHashKey is the attribute Key conforming to the "user.hash" semantic
	// conventions. It represents the unique user hash to correlate information for
	// a user in anonymized form.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "364fc68eaf4c8acec74a4e52d7d1feaa"
	// Note: Useful if `user.id` or `user.name` contain confidential information and
	// cannot be used.
	UserHashKey = attribute.Key("user.hash")

	// UserIDKey is the attribute Key conforming to the "user.id" semantic
	// conventions. It represents the unique identifier of the user.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "S-1-5-21-202424912787-2692429404-2351956786-1000"
	UserIDKey = attribute.Key("user.id")

	// UserNameKey is the attribute Key conforming to the "user.name" semantic
	// conventions. It represents the short name or login/username of the user.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "a.einstein"
	UserNameKey = attribute.Key("user.name")

	// UserRolesKey is the attribute Key conforming to the "user.roles" semantic
	// conventions. It represents the array of user roles at the time of the event.
	//
	// Type: string[]
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "admin", "reporting_user"
	UserRolesKey = attribute.Key("user.roles")
)

// UserEmail returns an attribute KeyValue conforming to the "user.email"
// semantic conventions. It represents the user email address.
func UserEmail(val string) attribute.KeyValue {
	return UserEmailKey.String(val)
}

// UserFullName returns an attribute KeyValue conforming to the "user.full_name"
// semantic conventions. It represents the user's full name.
func UserFullName(val string) attribute.KeyValue {
	return UserFullNameKey.String(val)
}

// UserHash returns an attribute KeyValue conforming to the "user.hash" semantic
// conventions. It represents the unique user hash to correlate information for a
// user in anonymized form.
func UserHash(val string) attribute.KeyValue {
	return UserHashKey.String(val)
}

// UserID returns an attribute KeyValue conforming to the "user.id" semantic
// conventions. It represents the unique identifier of the user.
func UserID(val string) attribute.KeyValue {
	return UserIDKey.String(val)
}

// UserName returns an attribute KeyValue conforming to the "user.name" semantic
// conventions. It represents the short name or login/username of the user.
func UserName(val string) attribute.KeyValue {
	return UserNameKey.String(val)
}

// UserRoles returns an attribute KeyValue conforming to the "user.roles"
// semantic conventions. It represents the array of user roles at the time of the
// event.
func UserRoles(val ...string) attribute.KeyValue {
	return UserRolesKey.StringSlice(val)
}

// Namespace: user_agent
const (
	// UserAgentNameKey is the attribute Key conforming to the "user_agent.name"
	// semantic conventions. It represents the name of the user-agent extracted from
	// original. Usually refers to the browser's name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Safari", "YourApp"
	// Note: [Example] of extracting browser's name from original string. In the
	// case of using a user-agent for non-browser products, such as microservices
	// with multiple names/versions inside the `user_agent.original`, the most
	// significant name SHOULD be selected. In such a scenario it should align with
	// `user_agent.version`
	//
	// [Example]: https://www.whatsmyua.info
	UserAgentNameKey = attribute.Key("user_agent.name")

	// UserAgentOriginalKey is the attribute Key conforming to the
	// "user_agent.original" semantic conventions. It represents the value of the
	// [HTTP User-Agent] header sent by the client.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Stable
	//
	// Examples: "CERN-LineMode/2.15 libwww/2.17b3", "Mozilla/5.0 (iPhone; CPU
	// iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko)
	// Version/14.1.2 Mobile/15E148 Safari/604.1", "YourApp/1.0.0
	// grpc-java-okhttp/1.27.2"
	//
	// [HTTP User-Agent]: https://www.rfc-editor.org/rfc/rfc9110.html#field.user-agent
	UserAgentOriginalKey = attribute.Key("user_agent.original")

	// UserAgentOSNameKey is the attribute Key conforming to the
	// "user_agent.os.name" semantic conventions. It represents the human readable
	// operating system name.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "iOS", "Android", "Ubuntu"
	// Note: For mapping user agent strings to OS names, libraries such as
	// [ua-parser] can be utilized.
	//
	// [ua-parser]: https://github.com/ua-parser
	UserAgentOSNameKey = attribute.Key("user_agent.os.name")

	// UserAgentOSVersionKey is the attribute Key conforming to the
	// "user_agent.os.version" semantic conventions. It represents the version
	// string of the operating system as defined in [Version Attributes].
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "14.2.1", "18.04.1"
	// Note: For mapping user agent strings to OS versions, libraries such as
	// [ua-parser] can be utilized.
	//
	// [Version Attributes]: /docs/resource/README.md#version-attributes
	// [ua-parser]: https://github.com/ua-parser
	UserAgentOSVersionKey = attribute.Key("user_agent.os.version")

	// UserAgentSyntheticTypeKey is the attribute Key conforming to the
	// "user_agent.synthetic.type" semantic conventions. It represents the specifies
	// the category of synthetic traffic, such as tests or bots.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// Note: This attribute MAY be derived from the contents of the
	// `user_agent.original` attribute. Components that populate the attribute are
	// responsible for determining what they consider to be synthetic bot or test
	// traffic. This attribute can either be set for self-identification purposes,
	// or on telemetry detected to be generated as a result of a synthetic request.
	// This attribute is useful for distinguishing between genuine client traffic
	// and synthetic traffic generated by bots or tests.
	UserAgentSyntheticTypeKey = attribute.Key("user_agent.synthetic.type")

	// UserAgentVersionKey is the attribute Key conforming to the
	// "user_agent.version" semantic conventions. It represents the version of the
	// user-agent extracted from original. Usually refers to the browser's version.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "14.1.2", "1.0.0"
	// Note: [Example] of extracting browser's version from original string. In the
	// case of using a user-agent for non-browser products, such as microservices
	// with multiple names/versions inside the `user_agent.original`, the most
	// significant version SHOULD be selected. In such a scenario it should align
	// with `user_agent.name`
	//
	// [Example]: https://www.whatsmyua.info
	UserAgentVersionKey = attribute.Key("user_agent.version")
)

// UserAgentName returns an attribute KeyValue conforming to the
// "user_agent.name" semantic conventions. It represents the name of the
// user-agent extracted from original. Usually refers to the browser's name.
func UserAgentName(val string) attribute.KeyValue {
	return UserAgentNameKey.String(val)
}

// UserAgentOriginal returns an attribute KeyValue conforming to the
// "user_agent.original" semantic conventions. It represents the value of the
// [HTTP User-Agent] header sent by the client.
//
// [HTTP User-Agent]: https://www.rfc-editor.org/rfc/rfc9110.html#field.user-agent
func UserAgentOriginal(val string) attribute.KeyValue {
	return UserAgentOriginalKey.String(val)
}

// UserAgentOSName returns an attribute KeyValue conforming to the
// "user_agent.os.name" semantic conventions. It represents the human readable
// operating system name.
func UserAgentOSName(val string) attribute.KeyValue {
	return UserAgentOSNameKey.String(val)
}

// UserAgentOSVersion returns an attribute KeyValue conforming to the
// "user_agent.os.version" semantic conventions. It represents the version string
// of the operating system as defined in [Version Attributes].
//
// [Version Attributes]: /docs/resource/README.md#version-attributes
func UserAgentOSVersion(val string) attribute.KeyValue {
	return UserAgentOSVersionKey.String(val)
}

// UserAgentVersion returns an attribute KeyValue conforming to the
// "user_agent.version" semantic conventions. It represents the version of the
// user-agent extracted from original. Usually refers to the browser's version.
func UserAgentVersion(val string) attribute.KeyValue {
	return UserAgentVersionKey.String(val)
}

// Enum values for user_agent.synthetic.type
var (
	// Bot source.
	// Stability: development
	UserAgentSyntheticTypeBot = UserAgentSyntheticTypeKey.String("bot")
	// Synthetic test source.
	// Stability: development
	UserAgentSyntheticTypeTest = UserAgentSyntheticTypeKey.String("test")
)

// Namespace: vcs
const (
	// VCSChangeIDKey is the attribute Key conforming to the "vcs.change.id"
	// semantic conventions. It represents the ID of the change (pull request/merge
	// request/changelist) if applicable. This is usually a unique (within
	// repository) identifier generated by the VCS system.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "123"
	VCSChangeIDKey = attribute.Key("vcs.change.id")

	// VCSChangeStateKey is the attribute Key conforming to the "vcs.change.state"
	// semantic conventions. It represents the state of the change (pull
	// request/merge request/changelist).
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "open", "closed", "merged"
	VCSChangeStateKey = attribute.Key("vcs.change.state")

	// VCSChangeTitleKey is the attribute Key conforming to the "vcs.change.title"
	// semantic conventions. It represents the human readable title of the change
	// (pull request/merge request/changelist). This title is often a brief summary
	// of the change and may get merged in to a ref as the commit summary.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "Fixes broken thing", "feat: add my new feature", "[chore] update
	// dependency"
	VCSChangeTitleKey = attribute.Key("vcs.change.title")

	// VCSLineChangeTypeKey is the attribute Key conforming to the
	// "vcs.line_change.type" semantic conventions. It represents the type of line
	// change being measured on a branch or change.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "added", "removed"
	VCSLineChangeTypeKey = attribute.Key("vcs.line_change.type")

	// VCSOwnerNameKey is the attribute Key conforming to the "vcs.owner.name"
	// semantic conventions. It represents the group owner within the version
	// control system.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "my-org", "myteam", "business-unit"
	VCSOwnerNameKey = attribute.Key("vcs.owner.name")

	// VCSProviderNameKey is the attribute Key conforming to the "vcs.provider.name"
	// semantic conventions. It represents the name of the version control system
	// provider.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "github", "gitlab", "gitea", "bitbucket"
	VCSProviderNameKey = attribute.Key("vcs.provider.name")

	// VCSRefBaseNameKey is the attribute Key conforming to the "vcs.ref.base.name"
	// semantic conventions. It represents the name of the [reference] such as
	// **branch** or **tag** in the repository.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "my-feature-branch", "tag-1-test"
	// Note: `base` refers to the starting point of a change. For example, `main`
	// would be the base reference of type branch if you've created a new
	// reference of type branch from it and created new commits.
	//
	// [reference]: https://git-scm.com/docs/gitglossary#def_ref
	VCSRefBaseNameKey = attribute.Key("vcs.ref.base.name")

	// VCSRefBaseRevisionKey is the attribute Key conforming to the
	// "vcs.ref.base.revision" semantic conventions. It represents the revision,
	// literally [revised version], The revision most often refers to a commit
	// object in Git, or a revision number in SVN.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "9d59409acf479dfa0df1aa568182e43e43df8bbe28d60fcf2bc52e30068802cc",
	// "main", "123", "HEAD"
	// Note: `base` refers to the starting point of a change. For example, `main`
	// would be the base reference of type branch if you've created a new
	// reference of type branch from it and created new commits. The
	// revision can be a full [hash value (see
	// glossary)],
	// of the recorded change to a ref within a repository pointing to a
	// commit [commit] object. It does
	// not necessarily have to be a hash; it can simply define a [revision
	// number]
	// which is an integer that is monotonically increasing. In cases where
	// it is identical to the `ref.base.name`, it SHOULD still be included.
	// It is up to the implementer to decide which value to set as the
	// revision based on the VCS system and situational context.
	//
	// [revised version]: https://www.merriam-webster.com/dictionary/revision
	// [hash value (see
	// glossary)]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-5.pdf
	// [commit]: https://git-scm.com/docs/git-commit
	// [revision
	// number]: https://svnbook.red-bean.com/en/1.7/svn.tour.revs.specifiers.html
	VCSRefBaseRevisionKey = attribute.Key("vcs.ref.base.revision")

	// VCSRefBaseTypeKey is the attribute Key conforming to the "vcs.ref.base.type"
	// semantic conventions. It represents the type of the [reference] in the
	// repository.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "branch", "tag"
	// Note: `base` refers to the starting point of a change. For example, `main`
	// would be the base reference of type branch if you've created a new
	// reference of type branch from it and created new commits.
	//
	// [reference]: https://git-scm.com/docs/gitglossary#def_ref
	VCSRefBaseTypeKey = attribute.Key("vcs.ref.base.type")

	// VCSRefHeadNameKey is the attribute Key conforming to the "vcs.ref.head.name"
	// semantic conventions. It represents the name of the [reference] such as
	// **branch** or **tag** in the repository.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "my-feature-branch", "tag-1-test"
	// Note: `head` refers to where you are right now; the current reference at a
	// given time.
	//
	// [reference]: https://git-scm.com/docs/gitglossary#def_ref
	VCSRefHeadNameKey = attribute.Key("vcs.ref.head.name")

	// VCSRefHeadRevisionKey is the attribute Key conforming to the
	// "vcs.ref.head.revision" semantic conventions. It represents the revision,
	// literally [revised version], The revision most often refers to a commit
	// object in Git, or a revision number in SVN.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "9d59409acf479dfa0df1aa568182e43e43df8bbe28d60fcf2bc52e30068802cc",
	// "main", "123", "HEAD"
	// Note: `head` refers to where you are right now; the current reference at a
	// given time.The revision can be a full [hash value (see
	// glossary)],
	// of the recorded change to a ref within a repository pointing to a
	// commit [commit] object. It does
	// not necessarily have to be a hash; it can simply define a [revision
	// number]
	// which is an integer that is monotonically increasing. In cases where
	// it is identical to the `ref.head.name`, it SHOULD still be included.
	// It is up to the implementer to decide which value to set as the
	// revision based on the VCS system and situational context.
	//
	// [revised version]: https://www.merriam-webster.com/dictionary/revision
	// [hash value (see
	// glossary)]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-5.pdf
	// [commit]: https://git-scm.com/docs/git-commit
	// [revision
	// number]: https://svnbook.red-bean.com/en/1.7/svn.tour.revs.specifiers.html
	VCSRefHeadRevisionKey = attribute.Key("vcs.ref.head.revision")

	// VCSRefHeadTypeKey is the attribute Key conforming to the "vcs.ref.head.type"
	// semantic conventions. It represents the type of the [reference] in the
	// repository.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "branch", "tag"
	// Note: `head` refers to where you are right now; the current reference at a
	// given time.
	//
	// [reference]: https://git-scm.com/docs/gitglossary#def_ref
	VCSRefHeadTypeKey = attribute.Key("vcs.ref.head.type")

	// VCSRefTypeKey is the attribute Key conforming to the "vcs.ref.type" semantic
	// conventions. It represents the type of the [reference] in the repository.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "branch", "tag"
	//
	// [reference]: https://git-scm.com/docs/gitglossary#def_ref
	VCSRefTypeKey = attribute.Key("vcs.ref.type")

	// VCSRepositoryNameKey is the attribute Key conforming to the
	// "vcs.repository.name" semantic conventions. It represents the human readable
	// name of the repository. It SHOULD NOT include any additional identifier like
	// Group/SubGroup in GitLab or organization in GitHub.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "semantic-conventions", "my-cool-repo"
	// Note: Due to it only being the name, it can clash with forks of the same
	// repository if collecting telemetry across multiple orgs or groups in
	// the same backends.
	VCSRepositoryNameKey = attribute.Key("vcs.repository.name")

	// VCSRepositoryURLFullKey is the attribute Key conforming to the
	// "vcs.repository.url.full" semantic conventions. It represents the
	// [canonical URL] of the repository providing the complete HTTP(S) address in
	// order to locate and identify the repository through a browser.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples:
	// "https://github.com/opentelemetry/open-telemetry-collector-contrib",
	// "https://gitlab.com/my-org/my-project/my-projects-project/repo"
	// Note: In Git Version Control Systems, the canonical URL SHOULD NOT include
	// the `.git` extension.
	//
	// [canonical URL]: https://support.google.com/webmasters/answer/10347851?hl=en#:~:text=A%20canonical%20URL%20is%20the,Google%20chooses%20one%20as%20canonical.
	VCSRepositoryURLFullKey = attribute.Key("vcs.repository.url.full")

	// VCSRevisionDeltaDirectionKey is the attribute Key conforming to the
	// "vcs.revision_delta.direction" semantic conventions. It represents the type
	// of revision comparison.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "ahead", "behind"
	VCSRevisionDeltaDirectionKey = attribute.Key("vcs.revision_delta.direction")
)

// VCSChangeID returns an attribute KeyValue conforming to the "vcs.change.id"
// semantic conventions. It represents the ID of the change (pull request/merge
// request/changelist) if applicable. This is usually a unique (within
// repository) identifier generated by the VCS system.
func VCSChangeID(val string) attribute.KeyValue {
	return VCSChangeIDKey.String(val)
}

// VCSChangeTitle returns an attribute KeyValue conforming to the
// "vcs.change.title" semantic conventions. It represents the human readable
// title of the change (pull request/merge request/changelist). This title is
// often a brief summary of the change and may get merged in to a ref as the
// commit summary.
func VCSChangeTitle(val string) attribute.KeyValue {
	return VCSChangeTitleKey.String(val)
}

// VCSOwnerName returns an attribute KeyValue conforming to the "vcs.owner.name"
// semantic conventions. It represents the group owner within the version control
// system.
func VCSOwnerName(val string) attribute.KeyValue {
	return VCSOwnerNameKey.String(val)
}

// VCSRefBaseName returns an attribute KeyValue conforming to the
// "vcs.ref.base.name" semantic conventions. It represents the name of the
// [reference] such as **branch** or **tag** in the repository.
//
// [reference]: https://git-scm.com/docs/gitglossary#def_ref
func VCSRefBaseName(val string) attribute.KeyValue {
	return VCSRefBaseNameKey.String(val)
}

// VCSRefBaseRevision returns an attribute KeyValue conforming to the
// "vcs.ref.base.revision" semantic conventions. It represents the revision,
// literally [revised version], The revision most often refers to a commit object
// in Git, or a revision number in SVN.
//
// [revised version]: https://www.merriam-webster.com/dictionary/revision
func VCSRefBaseRevision(val string) attribute.KeyValue {
	return VCSRefBaseRevisionKey.String(val)
}

// VCSRefHeadName returns an attribute KeyValue conforming to the
// "vcs.ref.head.name" semantic conventions. It represents the name of the
// [reference] such as **branch** or **tag** in the repository.
//
// [reference]: https://git-scm.com/docs/gitglossary#def_ref
func VCSRefHeadName(val string) attribute.KeyValue {
	return VCSRefHeadNameKey.String(val)
}

// VCSRefHeadRevision returns an attribute KeyValue conforming to the
// "vcs.ref.head.revision" semantic conventions. It represents the revision,
// literally [revised version], The revision most often refers to a commit object
// in Git, or a revision number in SVN.
//
// [revised version]: https://www.merriam-webster.com/dictionary/revision
func VCSRefHeadRevision(val string) attribute.KeyValue {
	return VCSRefHeadRevisionKey.String(val)
}

// VCSRepositoryName returns an attribute KeyValue conforming to the
// "vcs.repository.name" semantic conventions. It represents the human readable
// name of the repository. It SHOULD NOT include any additional identifier like
// Group/SubGroup in GitLab or organization in GitHub.
func VCSRepositoryName(val string) attribute.KeyValue {
	return VCSRepositoryNameKey.String(val)
}

// VCSRepositoryURLFull returns an attribute KeyValue conforming to the
// "vcs.repository.url.full" semantic conventions. It represents the
// [canonical URL] of the repository providing the complete HTTP(S) address in
// order to locate and identify the repository through a browser.
//
// [canonical URL]: https://support.google.com/webmasters/answer/10347851?hl=en#:~:text=A%20canonical%20URL%20is%20the,Google%20chooses%20one%20as%20canonical.
func VCSRepositoryURLFull(val string) attribute.KeyValue {
	return VCSRepositoryURLFullKey.String(val)
}

// Enum values for vcs.change.state
var (
	// Open means the change is currently active and under review. It hasn't been
	// merged into the target branch yet, and it's still possible to make changes or
	// add comments.
	// Stability: development
	VCSChangeStateOpen = VCSChangeStateKey.String("open")
	// WIP (work-in-progress, draft) means the change is still in progress and not
	// yet ready for a full review. It might still undergo significant changes.
	// Stability: development
	VCSChangeStateWip = VCSChangeStateKey.String("wip")
	// Closed means the merge request has been closed without merging. This can
	// happen for various reasons, such as the changes being deemed unnecessary, the
	// issue being resolved in another way, or the author deciding to withdraw the
	// request.
	// Stability: development
	VCSChangeStateClosed = VCSChangeStateKey.String("closed")
	// Merged indicates that the change has been successfully integrated into the
	// target codebase.
	// Stability: development
	VCSChangeStateMerged = VCSChangeStateKey.String("merged")
)

// Enum values for vcs.line_change.type
var (
	// How many lines were added.
	// Stability: development
	VCSLineChangeTypeAdded = VCSLineChangeTypeKey.String("added")
	// How many lines were removed.
	// Stability: development
	VCSLineChangeTypeRemoved = VCSLineChangeTypeKey.String("removed")
)

// Enum values for vcs.provider.name
var (
	// [GitHub]
	// Stability: development
	//
	// [GitHub]: https://github.com
	VCSProviderNameGithub = VCSProviderNameKey.String("github")
	// [GitLab]
	// Stability: development
	//
	// [GitLab]: https://gitlab.com
	VCSProviderNameGitlab = VCSProviderNameKey.String("gitlab")
	// [Gitea]
	// Stability: development
	//
	// [Gitea]: https://gitea.io
	VCSProviderNameGitea = VCSProviderNameKey.String("gitea")
	// [Bitbucket]
	// Stability: development
	//
	// [Bitbucket]: https://bitbucket.org
	VCSProviderNameBitbucket = VCSProviderNameKey.String("bitbucket")
)

// Enum values for vcs.ref.base.type
var (
	// [branch]
	// Stability: development
	//
	// [branch]: https://git-scm.com/docs/gitglossary#Documentation/gitglossary.txt-aiddefbranchabranch
	VCSRefBaseTypeBranch = VCSRefBaseTypeKey.String("branch")
	// [tag]
	// Stability: development
	//
	// [tag]: https://git-scm.com/docs/gitglossary#Documentation/gitglossary.txt-aiddeftagatag
	VCSRefBaseTypeTag = VCSRefBaseTypeKey.String("tag")
)

// Enum values for vcs.ref.head.type
var (
	// [branch]
	// Stability: development
	//
	// [branch]: https://git-scm.com/docs/gitglossary#Documentation/gitglossary.txt-aiddefbranchabranch
	VCSRefHeadTypeBranch = VCSRefHeadTypeKey.String("branch")
	// [tag]
	// Stability: development
	//
	// [tag]: https://git-scm.com/docs/gitglossary#Documentation/gitglossary.txt-aiddeftagatag
	VCSRefHeadTypeTag = VCSRefHeadTypeKey.String("tag")
)

// Enum values for vcs.ref.type
var (
	// [branch]
	// Stability: development
	//
	// [branch]: https://git-scm.com/docs/gitglossary#Documentation/gitglossary.txt-aiddefbranchabranch
	VCSRefTypeBranch = VCSRefTypeKey.String("branch")
	// [tag]
	// Stability: development
	//
	// [tag]: https://git-scm.com/docs/gitglossary#Documentation/gitglossary.txt-aiddeftagatag
	VCSRefTypeTag = VCSRefTypeKey.String("tag")
)

// Enum values for vcs.revision_delta.direction
var (
	// How many revisions the change is behind the target ref.
	// Stability: development
	VCSRevisionDeltaDirectionBehind = VCSRevisionDeltaDirectionKey.String("behind")
	// How many revisions the change is ahead of the target ref.
	// Stability: development
	VCSRevisionDeltaDirectionAhead = VCSRevisionDeltaDirectionKey.String("ahead")
)

// Namespace: webengine
const (
	// WebEngineDescriptionKey is the attribute Key conforming to the
	// "webengine.description" semantic conventions. It represents the additional
	// description of the web engine (e.g. detailed version and edition
	// information).
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "WildFly Full 21.0.0.Final (WildFly Core 13.0.1.Final) -
	// 2.2.2.Final"
	WebEngineDescriptionKey = attribute.Key("webengine.description")

	// WebEngineNameKey is the attribute Key conforming to the "webengine.name"
	// semantic conventions. It represents the name of the web engine.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "WildFly"
	WebEngineNameKey = attribute.Key("webengine.name")

	// WebEngineVersionKey is the attribute Key conforming to the
	// "webengine.version" semantic conventions. It represents the version of the
	// web engine.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "21.0.0"
	WebEngineVersionKey = attribute.Key("webengine.version")
)

// WebEngineDescription returns an attribute KeyValue conforming to the
// "webengine.description" semantic conventions. It represents the additional
// description of the web engine (e.g. detailed version and edition information).
func WebEngineDescription(val string) attribute.KeyValue {
	return WebEngineDescriptionKey.String(val)
}

// WebEngineName returns an attribute KeyValue conforming to the "webengine.name"
// semantic conventions. It represents the name of the web engine.
func WebEngineName(val string) attribute.KeyValue {
	return WebEngineNameKey.String(val)
}

// WebEngineVersion returns an attribute KeyValue conforming to the
// "webengine.version" semantic conventions. It represents the version of the web
// engine.
func WebEngineVersion(val string) attribute.KeyValue {
	return WebEngineVersionKey.String(val)
}

// Namespace: zos
const (
	// ZOSSmfIDKey is the attribute Key conforming to the "zos.smf.id" semantic
	// conventions. It represents the System Management Facility (SMF) Identifier
	// uniquely identified a z/OS system within a SYSPLEX or mainframe environment
	// and is used for system and performance analysis.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "SYS1"
	ZOSSmfIDKey = attribute.Key("zos.smf.id")

	// ZOSSysplexNameKey is the attribute Key conforming to the "zos.sysplex.name"
	// semantic conventions. It represents the name of the SYSPLEX to which the z/OS
	// system belongs too.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: Development
	//
	// Examples: "SYSPLEX1"
	ZOSSysplexNameKey = attribute.Key("zos.sysplex.name")
)

// ZOSSmfID returns an attribute KeyValue conforming to the "zos.smf.id" semantic
// conventions. It represents the System Management Facility (SMF) Identifier
// uniquely identified a z/OS system within a SYSPLEX or mainframe environment
// and is used for system and performance analysis.
func ZOSSmfID(val string) attribute.KeyValue {
	return ZOSSmfIDKey.String(val)
}

// ZOSSysplexName returns an attribute KeyValue conforming to the
// "zos.sysplex.name" semantic conventions. It represents the name of the SYSPLEX
// to which the z/OS system belongs too.
func ZOSSysplexName(val string) attribute.KeyValue {
	return ZOSSysplexNameKey.String(val)
}