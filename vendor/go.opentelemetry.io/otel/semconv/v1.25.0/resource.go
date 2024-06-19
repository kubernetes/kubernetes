// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Code generated from semantic convention specification. DO NOT EDIT.

package semconv // import "go.opentelemetry.io/otel/semconv/v1.25.0"

import "go.opentelemetry.io/otel/attribute"

// Resources used by AWS Elastic Container Service (ECS).
const (
	// AWSECSTaskIDKey is the attribute Key conforming to the "aws.ecs.task.id"
	// semantic conventions. It represents the ID of a running ECS task. The ID
	// MUST be extracted from `task.arn`.
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (If and only if `task.arn` is
	// populated.)
	// Stability: experimental
	// Examples: '10838bed-421f-43ef-870a-f43feacbbb5b',
	// '23ebb8ac-c18f-46c6-8bbe-d55d0e37cfbd'
	AWSECSTaskIDKey = attribute.Key("aws.ecs.task.id")

	// AWSECSClusterARNKey is the attribute Key conforming to the
	// "aws.ecs.cluster.arn" semantic conventions. It represents the ARN of an
	// [ECS
	// cluster](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/clusters.html).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'arn:aws:ecs:us-west-2:123456789123:cluster/my-cluster'
	AWSECSClusterARNKey = attribute.Key("aws.ecs.cluster.arn")

	// AWSECSContainerARNKey is the attribute Key conforming to the
	// "aws.ecs.container.arn" semantic conventions. It represents the Amazon
	// Resource Name (ARN) of an [ECS container
	// instance](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_instances.html).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples:
	// 'arn:aws:ecs:us-west-1:123456789123:container/32624152-9086-4f0e-acae-1a75b14fe4d9'
	AWSECSContainerARNKey = attribute.Key("aws.ecs.container.arn")

	// AWSECSLaunchtypeKey is the attribute Key conforming to the
	// "aws.ecs.launchtype" semantic conventions. It represents the [launch
	// type](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/launch_types.html)
	// for an ECS task.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: experimental
	AWSECSLaunchtypeKey = attribute.Key("aws.ecs.launchtype")

	// AWSECSTaskARNKey is the attribute Key conforming to the
	// "aws.ecs.task.arn" semantic conventions. It represents the ARN of a
	// running [ECS
	// task](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-account-settings.html#ecs-resource-ids).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples:
	// 'arn:aws:ecs:us-west-1:123456789123:task/10838bed-421f-43ef-870a-f43feacbbb5b',
	// 'arn:aws:ecs:us-west-1:123456789123:task/my-cluster/task-id/23ebb8ac-c18f-46c6-8bbe-d55d0e37cfbd'
	AWSECSTaskARNKey = attribute.Key("aws.ecs.task.arn")

	// AWSECSTaskFamilyKey is the attribute Key conforming to the
	// "aws.ecs.task.family" semantic conventions. It represents the family
	// name of the [ECS task
	// definition](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definitions.html)
	// used to create the ECS task.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'opentelemetry-family'
	AWSECSTaskFamilyKey = attribute.Key("aws.ecs.task.family")

	// AWSECSTaskRevisionKey is the attribute Key conforming to the
	// "aws.ecs.task.revision" semantic conventions. It represents the revision
	// for the task definition used to create the ECS task.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '8', '26'
	AWSECSTaskRevisionKey = attribute.Key("aws.ecs.task.revision")
)

var (
	// ec2
	AWSECSLaunchtypeEC2 = AWSECSLaunchtypeKey.String("ec2")
	// fargate
	AWSECSLaunchtypeFargate = AWSECSLaunchtypeKey.String("fargate")
)

// AWSECSTaskID returns an attribute KeyValue conforming to the
// "aws.ecs.task.id" semantic conventions. It represents the ID of a running
// ECS task. The ID MUST be extracted from `task.arn`.
func AWSECSTaskID(val string) attribute.KeyValue {
	return AWSECSTaskIDKey.String(val)
}

// AWSECSClusterARN returns an attribute KeyValue conforming to the
// "aws.ecs.cluster.arn" semantic conventions. It represents the ARN of an [ECS
// cluster](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/clusters.html).
func AWSECSClusterARN(val string) attribute.KeyValue {
	return AWSECSClusterARNKey.String(val)
}

// AWSECSContainerARN returns an attribute KeyValue conforming to the
// "aws.ecs.container.arn" semantic conventions. It represents the Amazon
// Resource Name (ARN) of an [ECS container
// instance](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_instances.html).
func AWSECSContainerARN(val string) attribute.KeyValue {
	return AWSECSContainerARNKey.String(val)
}

// AWSECSTaskARN returns an attribute KeyValue conforming to the
// "aws.ecs.task.arn" semantic conventions. It represents the ARN of a running
// [ECS
// task](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-account-settings.html#ecs-resource-ids).
func AWSECSTaskARN(val string) attribute.KeyValue {
	return AWSECSTaskARNKey.String(val)
}

// AWSECSTaskFamily returns an attribute KeyValue conforming to the
// "aws.ecs.task.family" semantic conventions. It represents the family name of
// the [ECS task
// definition](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definitions.html)
// used to create the ECS task.
func AWSECSTaskFamily(val string) attribute.KeyValue {
	return AWSECSTaskFamilyKey.String(val)
}

// AWSECSTaskRevision returns an attribute KeyValue conforming to the
// "aws.ecs.task.revision" semantic conventions. It represents the revision for
// the task definition used to create the ECS task.
func AWSECSTaskRevision(val string) attribute.KeyValue {
	return AWSECSTaskRevisionKey.String(val)
}

// Resources used by AWS Elastic Kubernetes Service (EKS).
const (
	// AWSEKSClusterARNKey is the attribute Key conforming to the
	// "aws.eks.cluster.arn" semantic conventions. It represents the ARN of an
	// EKS cluster.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'arn:aws:ecs:us-west-2:123456789123:cluster/my-cluster'
	AWSEKSClusterARNKey = attribute.Key("aws.eks.cluster.arn")
)

// AWSEKSClusterARN returns an attribute KeyValue conforming to the
// "aws.eks.cluster.arn" semantic conventions. It represents the ARN of an EKS
// cluster.
func AWSEKSClusterARN(val string) attribute.KeyValue {
	return AWSEKSClusterARNKey.String(val)
}

// Resources specific to Amazon Web Services.
const (
	// AWSLogGroupARNsKey is the attribute Key conforming to the
	// "aws.log.group.arns" semantic conventions. It represents the Amazon
	// Resource Name(s) (ARN) of the AWS log group(s).
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples:
	// 'arn:aws:logs:us-west-1:123456789012:log-group:/aws/my/group:*'
	// Note: See the [log group ARN format
	// documentation](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/iam-access-control-overview-cwl.html#CWL_ARN_Format).
	AWSLogGroupARNsKey = attribute.Key("aws.log.group.arns")

	// AWSLogGroupNamesKey is the attribute Key conforming to the
	// "aws.log.group.names" semantic conventions. It represents the name(s) of
	// the AWS log group(s) an application is writing to.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '/aws/lambda/my-function', 'opentelemetry-service'
	// Note: Multiple log groups must be supported for cases like
	// multi-container applications, where a single application has sidecar
	// containers, and each write to their own log group.
	AWSLogGroupNamesKey = attribute.Key("aws.log.group.names")

	// AWSLogStreamARNsKey is the attribute Key conforming to the
	// "aws.log.stream.arns" semantic conventions. It represents the ARN(s) of
	// the AWS log stream(s).
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples:
	// 'arn:aws:logs:us-west-1:123456789012:log-group:/aws/my/group:log-stream:logs/main/10838bed-421f-43ef-870a-f43feacbbb5b'
	// Note: See the [log stream ARN format
	// documentation](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/iam-access-control-overview-cwl.html#CWL_ARN_Format).
	// One log group can contain several log streams, so these ARNs necessarily
	// identify both a log group and a log stream.
	AWSLogStreamARNsKey = attribute.Key("aws.log.stream.arns")

	// AWSLogStreamNamesKey is the attribute Key conforming to the
	// "aws.log.stream.names" semantic conventions. It represents the name(s)
	// of the AWS log stream(s) an application is writing to.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'logs/main/10838bed-421f-43ef-870a-f43feacbbb5b'
	AWSLogStreamNamesKey = attribute.Key("aws.log.stream.names")
)

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
// "aws.log.stream.names" semantic conventions. It represents the name(s) of
// the AWS log stream(s) an application is writing to.
func AWSLogStreamNames(val ...string) attribute.KeyValue {
	return AWSLogStreamNamesKey.StringSlice(val)
}

// Heroku dyno metadata
const (
	// HerokuAppIDKey is the attribute Key conforming to the "heroku.app.id"
	// semantic conventions. It represents the unique identifier for the
	// application
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '2daa2797-e42b-4624-9322-ec3f968df4da'
	HerokuAppIDKey = attribute.Key("heroku.app.id")

	// HerokuReleaseCommitKey is the attribute Key conforming to the
	// "heroku.release.commit" semantic conventions. It represents the commit
	// hash for the current release
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'e6134959463efd8966b20e75b913cafe3f5ec'
	HerokuReleaseCommitKey = attribute.Key("heroku.release.commit")

	// HerokuReleaseCreationTimestampKey is the attribute Key conforming to the
	// "heroku.release.creation_timestamp" semantic conventions. It represents
	// the time and date the release was created
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '2022-10-23T18:00:42Z'
	HerokuReleaseCreationTimestampKey = attribute.Key("heroku.release.creation_timestamp")
)

// HerokuAppID returns an attribute KeyValue conforming to the
// "heroku.app.id" semantic conventions. It represents the unique identifier
// for the application
func HerokuAppID(val string) attribute.KeyValue {
	return HerokuAppIDKey.String(val)
}

// HerokuReleaseCommit returns an attribute KeyValue conforming to the
// "heroku.release.commit" semantic conventions. It represents the commit hash
// for the current release
func HerokuReleaseCommit(val string) attribute.KeyValue {
	return HerokuReleaseCommitKey.String(val)
}

// HerokuReleaseCreationTimestamp returns an attribute KeyValue conforming
// to the "heroku.release.creation_timestamp" semantic conventions. It
// represents the time and date the release was created
func HerokuReleaseCreationTimestamp(val string) attribute.KeyValue {
	return HerokuReleaseCreationTimestampKey.String(val)
}

// Resource describing the packaged software running the application code. Web
// engines are typically executed using process.runtime.
const (
	// WebEngineNameKey is the attribute Key conforming to the "webengine.name"
	// semantic conventions. It represents the name of the web engine.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: experimental
	// Examples: 'WildFly'
	WebEngineNameKey = attribute.Key("webengine.name")

	// WebEngineDescriptionKey is the attribute Key conforming to the
	// "webengine.description" semantic conventions. It represents the
	// additional description of the web engine (e.g. detailed version and
	// edition information).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'WildFly Full 21.0.0.Final (WildFly Core 13.0.1.Final) -
	// 2.2.2.Final'
	WebEngineDescriptionKey = attribute.Key("webengine.description")

	// WebEngineVersionKey is the attribute Key conforming to the
	// "webengine.version" semantic conventions. It represents the version of
	// the web engine.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '21.0.0'
	WebEngineVersionKey = attribute.Key("webengine.version")
)

// WebEngineName returns an attribute KeyValue conforming to the
// "webengine.name" semantic conventions. It represents the name of the web
// engine.
func WebEngineName(val string) attribute.KeyValue {
	return WebEngineNameKey.String(val)
}

// WebEngineDescription returns an attribute KeyValue conforming to the
// "webengine.description" semantic conventions. It represents the additional
// description of the web engine (e.g. detailed version and edition
// information).
func WebEngineDescription(val string) attribute.KeyValue {
	return WebEngineDescriptionKey.String(val)
}

// WebEngineVersion returns an attribute KeyValue conforming to the
// "webengine.version" semantic conventions. It represents the version of the
// web engine.
func WebEngineVersion(val string) attribute.KeyValue {
	return WebEngineVersionKey.String(val)
}

// Attributes used by non-OTLP exporters to represent OpenTelemetry Scope's
// concepts.
const (
	// OTelScopeNameKey is the attribute Key conforming to the
	// "otel.scope.name" semantic conventions. It represents the name of the
	// instrumentation scope - (`InstrumentationScope.Name` in OTLP).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'io.opentelemetry.contrib.mongodb'
	OTelScopeNameKey = attribute.Key("otel.scope.name")

	// OTelScopeVersionKey is the attribute Key conforming to the
	// "otel.scope.version" semantic conventions. It represents the version of
	// the instrumentation scope - (`InstrumentationScope.Version` in OTLP).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '1.0.0'
	OTelScopeVersionKey = attribute.Key("otel.scope.version")
)

// OTelScopeName returns an attribute KeyValue conforming to the
// "otel.scope.name" semantic conventions. It represents the name of the
// instrumentation scope - (`InstrumentationScope.Name` in OTLP).
func OTelScopeName(val string) attribute.KeyValue {
	return OTelScopeNameKey.String(val)
}

// OTelScopeVersion returns an attribute KeyValue conforming to the
// "otel.scope.version" semantic conventions. It represents the version of the
// instrumentation scope - (`InstrumentationScope.Version` in OTLP).
func OTelScopeVersion(val string) attribute.KeyValue {
	return OTelScopeVersionKey.String(val)
}

// Span attributes used by non-OTLP exporters to represent OpenTelemetry
// Scope's concepts.
const (
	// OTelLibraryNameKey is the attribute Key conforming to the
	// "otel.library.name" semantic conventions. It represents the none
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: 'io.opentelemetry.contrib.mongodb'
	OTelLibraryNameKey = attribute.Key("otel.library.name")

	// OTelLibraryVersionKey is the attribute Key conforming to the
	// "otel.library.version" semantic conventions. It represents the none
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: experimental
	// Examples: '1.0.0'
	OTelLibraryVersionKey = attribute.Key("otel.library.version")
)

// OTelLibraryName returns an attribute KeyValue conforming to the
// "otel.library.name" semantic conventions. It represents the none
func OTelLibraryName(val string) attribute.KeyValue {
	return OTelLibraryNameKey.String(val)
}

// OTelLibraryVersion returns an attribute KeyValue conforming to the
// "otel.library.version" semantic conventions. It represents the none
func OTelLibraryVersion(val string) attribute.KeyValue {
	return OTelLibraryVersionKey.String(val)
}
