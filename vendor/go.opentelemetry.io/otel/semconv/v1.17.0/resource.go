// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Code generated from semantic convention specification. DO NOT EDIT.

package semconv // import "go.opentelemetry.io/otel/semconv/v1.17.0"

import "go.opentelemetry.io/otel/attribute"

// The web browser in which the application represented by the resource is
// running. The `browser.*` attributes MUST be used only for resources that
// represent applications running in a web browser (regardless of whether
// running on a mobile or desktop device).
const (
	// BrowserBrandsKey is the attribute Key conforming to the "browser.brands"
	// semantic conventions. It represents the array of brand name and version
	// separated by a space
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: ' Not A;Brand 99', 'Chromium 99', 'Chrome 99'
	// Note: This value is intended to be taken from the [UA client hints
	// API](https://wicg.github.io/ua-client-hints/#interface)
	// (`navigator.userAgentData.brands`).
	BrowserBrandsKey = attribute.Key("browser.brands")

	// BrowserPlatformKey is the attribute Key conforming to the
	// "browser.platform" semantic conventions. It represents the platform on
	// which the browser is running
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'Windows', 'macOS', 'Android'
	// Note: This value is intended to be taken from the [UA client hints
	// API](https://wicg.github.io/ua-client-hints/#interface)
	// (`navigator.userAgentData.platform`). If unavailable, the legacy
	// `navigator.platform` API SHOULD NOT be used instead and this attribute
	// SHOULD be left unset in order for the values to be consistent.
	// The list of possible values is defined in the [W3C User-Agent Client
	// Hints
	// specification](https://wicg.github.io/ua-client-hints/#sec-ch-ua-platform).
	// Note that some (but not all) of these values can overlap with values in
	// the [`os.type` and `os.name` attributes](./os.md). However, for
	// consistency, the values in the `browser.platform` attribute should
	// capture the exact value that the user agent provides.
	BrowserPlatformKey = attribute.Key("browser.platform")

	// BrowserMobileKey is the attribute Key conforming to the "browser.mobile"
	// semantic conventions. It represents a boolean that is true if the
	// browser is running on a mobile device
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: stable
	// Note: This value is intended to be taken from the [UA client hints
	// API](https://wicg.github.io/ua-client-hints/#interface)
	// (`navigator.userAgentData.mobile`). If unavailable, this attribute
	// SHOULD be left unset.
	BrowserMobileKey = attribute.Key("browser.mobile")

	// BrowserUserAgentKey is the attribute Key conforming to the
	// "browser.user_agent" semantic conventions. It represents the full
	// user-agent string provided by the browser
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)
	// AppleWebKit/537.36 (KHTML, '
	//  'like Gecko) Chrome/95.0.4638.54 Safari/537.36'
	// Note: The user-agent value SHOULD be provided only from browsers that do
	// not have a mechanism to retrieve brands and platform individually from
	// the User-Agent Client Hints API. To retrieve the value, the legacy
	// `navigator.userAgent` API can be used.
	BrowserUserAgentKey = attribute.Key("browser.user_agent")

	// BrowserLanguageKey is the attribute Key conforming to the
	// "browser.language" semantic conventions. It represents the preferred
	// language of the user using the browser
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'en', 'en-US', 'fr', 'fr-FR'
	// Note: This value is intended to be taken from the Navigator API
	// `navigator.language`.
	BrowserLanguageKey = attribute.Key("browser.language")
)

// BrowserBrands returns an attribute KeyValue conforming to the
// "browser.brands" semantic conventions. It represents the array of brand name
// and version separated by a space
func BrowserBrands(val ...string) attribute.KeyValue {
	return BrowserBrandsKey.StringSlice(val)
}

// BrowserPlatform returns an attribute KeyValue conforming to the
// "browser.platform" semantic conventions. It represents the platform on which
// the browser is running
func BrowserPlatform(val string) attribute.KeyValue {
	return BrowserPlatformKey.String(val)
}

// BrowserMobile returns an attribute KeyValue conforming to the
// "browser.mobile" semantic conventions. It represents a boolean that is true
// if the browser is running on a mobile device
func BrowserMobile(val bool) attribute.KeyValue {
	return BrowserMobileKey.Bool(val)
}

// BrowserUserAgent returns an attribute KeyValue conforming to the
// "browser.user_agent" semantic conventions. It represents the full user-agent
// string provided by the browser
func BrowserUserAgent(val string) attribute.KeyValue {
	return BrowserUserAgentKey.String(val)
}

// BrowserLanguage returns an attribute KeyValue conforming to the
// "browser.language" semantic conventions. It represents the preferred
// language of the user using the browser
func BrowserLanguage(val string) attribute.KeyValue {
	return BrowserLanguageKey.String(val)
}

// A cloud environment (e.g. GCP, Azure, AWS)
const (
	// CloudProviderKey is the attribute Key conforming to the "cloud.provider"
	// semantic conventions. It represents the name of the cloud provider.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	CloudProviderKey = attribute.Key("cloud.provider")

	// CloudAccountIDKey is the attribute Key conforming to the
	// "cloud.account.id" semantic conventions. It represents the cloud account
	// ID the resource is assigned to.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '111111111111', 'opentelemetry'
	CloudAccountIDKey = attribute.Key("cloud.account.id")

	// CloudRegionKey is the attribute Key conforming to the "cloud.region"
	// semantic conventions. It represents the geographical region the resource
	// is running.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'us-central1', 'us-east-1'
	// Note: Refer to your provider's docs to see the available regions, for
	// example [Alibaba Cloud
	// regions](https://www.alibabacloud.com/help/doc-detail/40654.htm), [AWS
	// regions](https://aws.amazon.com/about-aws/global-infrastructure/regions_az/),
	// [Azure
	// regions](https://azure.microsoft.com/en-us/global-infrastructure/geographies/),
	// [Google Cloud regions](https://cloud.google.com/about/locations), or
	// [Tencent Cloud
	// regions](https://intl.cloud.tencent.com/document/product/213/6091).
	CloudRegionKey = attribute.Key("cloud.region")

	// CloudAvailabilityZoneKey is the attribute Key conforming to the
	// "cloud.availability_zone" semantic conventions. It represents the cloud
	// regions often have multiple, isolated locations known as zones to
	// increase availability. Availability zone represents the zone where the
	// resource is running.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'us-east-1c'
	// Note: Availability zones are called "zones" on Alibaba Cloud and Google
	// Cloud.
	CloudAvailabilityZoneKey = attribute.Key("cloud.availability_zone")

	// CloudPlatformKey is the attribute Key conforming to the "cloud.platform"
	// semantic conventions. It represents the cloud platform in use.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	// Note: The prefix of the service SHOULD match the one specified in
	// `cloud.provider`.
	CloudPlatformKey = attribute.Key("cloud.platform")
)

var (
	// Alibaba Cloud
	CloudProviderAlibabaCloud = CloudProviderKey.String("alibaba_cloud")
	// Amazon Web Services
	CloudProviderAWS = CloudProviderKey.String("aws")
	// Microsoft Azure
	CloudProviderAzure = CloudProviderKey.String("azure")
	// Google Cloud Platform
	CloudProviderGCP = CloudProviderKey.String("gcp")
	// IBM Cloud
	CloudProviderIbmCloud = CloudProviderKey.String("ibm_cloud")
	// Tencent Cloud
	CloudProviderTencentCloud = CloudProviderKey.String("tencent_cloud")
)

var (
	// Alibaba Cloud Elastic Compute Service
	CloudPlatformAlibabaCloudECS = CloudPlatformKey.String("alibaba_cloud_ecs")
	// Alibaba Cloud Function Compute
	CloudPlatformAlibabaCloudFc = CloudPlatformKey.String("alibaba_cloud_fc")
	// Red Hat OpenShift on Alibaba Cloud
	CloudPlatformAlibabaCloudOpenshift = CloudPlatformKey.String("alibaba_cloud_openshift")
	// AWS Elastic Compute Cloud
	CloudPlatformAWSEC2 = CloudPlatformKey.String("aws_ec2")
	// AWS Elastic Container Service
	CloudPlatformAWSECS = CloudPlatformKey.String("aws_ecs")
	// AWS Elastic Kubernetes Service
	CloudPlatformAWSEKS = CloudPlatformKey.String("aws_eks")
	// AWS Lambda
	CloudPlatformAWSLambda = CloudPlatformKey.String("aws_lambda")
	// AWS Elastic Beanstalk
	CloudPlatformAWSElasticBeanstalk = CloudPlatformKey.String("aws_elastic_beanstalk")
	// AWS App Runner
	CloudPlatformAWSAppRunner = CloudPlatformKey.String("aws_app_runner")
	// Red Hat OpenShift on AWS (ROSA)
	CloudPlatformAWSOpenshift = CloudPlatformKey.String("aws_openshift")
	// Azure Virtual Machines
	CloudPlatformAzureVM = CloudPlatformKey.String("azure_vm")
	// Azure Container Instances
	CloudPlatformAzureContainerInstances = CloudPlatformKey.String("azure_container_instances")
	// Azure Kubernetes Service
	CloudPlatformAzureAKS = CloudPlatformKey.String("azure_aks")
	// Azure Functions
	CloudPlatformAzureFunctions = CloudPlatformKey.String("azure_functions")
	// Azure App Service
	CloudPlatformAzureAppService = CloudPlatformKey.String("azure_app_service")
	// Azure Red Hat OpenShift
	CloudPlatformAzureOpenshift = CloudPlatformKey.String("azure_openshift")
	// Google Cloud Compute Engine (GCE)
	CloudPlatformGCPComputeEngine = CloudPlatformKey.String("gcp_compute_engine")
	// Google Cloud Run
	CloudPlatformGCPCloudRun = CloudPlatformKey.String("gcp_cloud_run")
	// Google Cloud Kubernetes Engine (GKE)
	CloudPlatformGCPKubernetesEngine = CloudPlatformKey.String("gcp_kubernetes_engine")
	// Google Cloud Functions (GCF)
	CloudPlatformGCPCloudFunctions = CloudPlatformKey.String("gcp_cloud_functions")
	// Google Cloud App Engine (GAE)
	CloudPlatformGCPAppEngine = CloudPlatformKey.String("gcp_app_engine")
	// Red Hat OpenShift on Google Cloud
	CloudPlatformGoogleCloudOpenshift = CloudPlatformKey.String("google_cloud_openshift")
	// Red Hat OpenShift on IBM Cloud
	CloudPlatformIbmCloudOpenshift = CloudPlatformKey.String("ibm_cloud_openshift")
	// Tencent Cloud Cloud Virtual Machine (CVM)
	CloudPlatformTencentCloudCvm = CloudPlatformKey.String("tencent_cloud_cvm")
	// Tencent Cloud Elastic Kubernetes Service (EKS)
	CloudPlatformTencentCloudEKS = CloudPlatformKey.String("tencent_cloud_eks")
	// Tencent Cloud Serverless Cloud Function (SCF)
	CloudPlatformTencentCloudScf = CloudPlatformKey.String("tencent_cloud_scf")
)

// CloudAccountID returns an attribute KeyValue conforming to the
// "cloud.account.id" semantic conventions. It represents the cloud account ID
// the resource is assigned to.
func CloudAccountID(val string) attribute.KeyValue {
	return CloudAccountIDKey.String(val)
}

// CloudRegion returns an attribute KeyValue conforming to the
// "cloud.region" semantic conventions. It represents the geographical region
// the resource is running.
func CloudRegion(val string) attribute.KeyValue {
	return CloudRegionKey.String(val)
}

// CloudAvailabilityZone returns an attribute KeyValue conforming to the
// "cloud.availability_zone" semantic conventions. It represents the cloud
// regions often have multiple, isolated locations known as zones to increase
// availability. Availability zone represents the zone where the resource is
// running.
func CloudAvailabilityZone(val string) attribute.KeyValue {
	return CloudAvailabilityZoneKey.String(val)
}

// Resources used by AWS Elastic Container Service (ECS).
const (
	// AWSECSContainerARNKey is the attribute Key conforming to the
	// "aws.ecs.container.arn" semantic conventions. It represents the Amazon
	// Resource Name (ARN) of an [ECS container
	// instance](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_instances.html).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples:
	// 'arn:aws:ecs:us-west-1:123456789123:container/32624152-9086-4f0e-acae-1a75b14fe4d9'
	AWSECSContainerARNKey = attribute.Key("aws.ecs.container.arn")

	// AWSECSClusterARNKey is the attribute Key conforming to the
	// "aws.ecs.cluster.arn" semantic conventions. It represents the ARN of an
	// [ECS
	// cluster](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/clusters.html).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'arn:aws:ecs:us-west-2:123456789123:cluster/my-cluster'
	AWSECSClusterARNKey = attribute.Key("aws.ecs.cluster.arn")

	// AWSECSLaunchtypeKey is the attribute Key conforming to the
	// "aws.ecs.launchtype" semantic conventions. It represents the [launch
	// type](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/launch_types.html)
	// for an ECS task.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	AWSECSLaunchtypeKey = attribute.Key("aws.ecs.launchtype")

	// AWSECSTaskARNKey is the attribute Key conforming to the
	// "aws.ecs.task.arn" semantic conventions. It represents the ARN of an
	// [ECS task
	// definition](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definitions.html).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples:
	// 'arn:aws:ecs:us-west-1:123456789123:task/10838bed-421f-43ef-870a-f43feacbbb5b'
	AWSECSTaskARNKey = attribute.Key("aws.ecs.task.arn")

	// AWSECSTaskFamilyKey is the attribute Key conforming to the
	// "aws.ecs.task.family" semantic conventions. It represents the task
	// definition family this task definition is a member of.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'opentelemetry-family'
	AWSECSTaskFamilyKey = attribute.Key("aws.ecs.task.family")

	// AWSECSTaskRevisionKey is the attribute Key conforming to the
	// "aws.ecs.task.revision" semantic conventions. It represents the revision
	// for this task definition.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '8', '26'
	AWSECSTaskRevisionKey = attribute.Key("aws.ecs.task.revision")
)

var (
	// ec2
	AWSECSLaunchtypeEC2 = AWSECSLaunchtypeKey.String("ec2")
	// fargate
	AWSECSLaunchtypeFargate = AWSECSLaunchtypeKey.String("fargate")
)

// AWSECSContainerARN returns an attribute KeyValue conforming to the
// "aws.ecs.container.arn" semantic conventions. It represents the Amazon
// Resource Name (ARN) of an [ECS container
// instance](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_instances.html).
func AWSECSContainerARN(val string) attribute.KeyValue {
	return AWSECSContainerARNKey.String(val)
}

// AWSECSClusterARN returns an attribute KeyValue conforming to the
// "aws.ecs.cluster.arn" semantic conventions. It represents the ARN of an [ECS
// cluster](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/clusters.html).
func AWSECSClusterARN(val string) attribute.KeyValue {
	return AWSECSClusterARNKey.String(val)
}

// AWSECSTaskARN returns an attribute KeyValue conforming to the
// "aws.ecs.task.arn" semantic conventions. It represents the ARN of an [ECS
// task
// definition](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definitions.html).
func AWSECSTaskARN(val string) attribute.KeyValue {
	return AWSECSTaskARNKey.String(val)
}

// AWSECSTaskFamily returns an attribute KeyValue conforming to the
// "aws.ecs.task.family" semantic conventions. It represents the task
// definition family this task definition is a member of.
func AWSECSTaskFamily(val string) attribute.KeyValue {
	return AWSECSTaskFamilyKey.String(val)
}

// AWSECSTaskRevision returns an attribute KeyValue conforming to the
// "aws.ecs.task.revision" semantic conventions. It represents the revision for
// this task definition.
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
	// Stability: stable
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
	// AWSLogGroupNamesKey is the attribute Key conforming to the
	// "aws.log.group.names" semantic conventions. It represents the name(s) of
	// the AWS log group(s) an application is writing to.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '/aws/lambda/my-function', 'opentelemetry-service'
	// Note: Multiple log groups must be supported for cases like
	// multi-container applications, where a single application has sidecar
	// containers, and each write to their own log group.
	AWSLogGroupNamesKey = attribute.Key("aws.log.group.names")

	// AWSLogGroupARNsKey is the attribute Key conforming to the
	// "aws.log.group.arns" semantic conventions. It represents the Amazon
	// Resource Name(s) (ARN) of the AWS log group(s).
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: stable
	// Examples:
	// 'arn:aws:logs:us-west-1:123456789012:log-group:/aws/my/group:*'
	// Note: See the [log group ARN format
	// documentation](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/iam-access-control-overview-cwl.html#CWL_ARN_Format).
	AWSLogGroupARNsKey = attribute.Key("aws.log.group.arns")

	// AWSLogStreamNamesKey is the attribute Key conforming to the
	// "aws.log.stream.names" semantic conventions. It represents the name(s)
	// of the AWS log stream(s) an application is writing to.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'logs/main/10838bed-421f-43ef-870a-f43feacbbb5b'
	AWSLogStreamNamesKey = attribute.Key("aws.log.stream.names")

	// AWSLogStreamARNsKey is the attribute Key conforming to the
	// "aws.log.stream.arns" semantic conventions. It represents the ARN(s) of
	// the AWS log stream(s).
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: stable
	// Examples:
	// 'arn:aws:logs:us-west-1:123456789012:log-group:/aws/my/group:log-stream:logs/main/10838bed-421f-43ef-870a-f43feacbbb5b'
	// Note: See the [log stream ARN format
	// documentation](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/iam-access-control-overview-cwl.html#CWL_ARN_Format).
	// One log group can contain several log streams, so these ARNs necessarily
	// identify both a log group and a log stream.
	AWSLogStreamARNsKey = attribute.Key("aws.log.stream.arns")
)

// AWSLogGroupNames returns an attribute KeyValue conforming to the
// "aws.log.group.names" semantic conventions. It represents the name(s) of the
// AWS log group(s) an application is writing to.
func AWSLogGroupNames(val ...string) attribute.KeyValue {
	return AWSLogGroupNamesKey.StringSlice(val)
}

// AWSLogGroupARNs returns an attribute KeyValue conforming to the
// "aws.log.group.arns" semantic conventions. It represents the Amazon Resource
// Name(s) (ARN) of the AWS log group(s).
func AWSLogGroupARNs(val ...string) attribute.KeyValue {
	return AWSLogGroupARNsKey.StringSlice(val)
}

// AWSLogStreamNames returns an attribute KeyValue conforming to the
// "aws.log.stream.names" semantic conventions. It represents the name(s) of
// the AWS log stream(s) an application is writing to.
func AWSLogStreamNames(val ...string) attribute.KeyValue {
	return AWSLogStreamNamesKey.StringSlice(val)
}

// AWSLogStreamARNs returns an attribute KeyValue conforming to the
// "aws.log.stream.arns" semantic conventions. It represents the ARN(s) of the
// AWS log stream(s).
func AWSLogStreamARNs(val ...string) attribute.KeyValue {
	return AWSLogStreamARNsKey.StringSlice(val)
}

// A container instance.
const (
	// ContainerNameKey is the attribute Key conforming to the "container.name"
	// semantic conventions. It represents the container name used by container
	// runtime.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'opentelemetry-autoconf'
	ContainerNameKey = attribute.Key("container.name")

	// ContainerIDKey is the attribute Key conforming to the "container.id"
	// semantic conventions. It represents the container ID. Usually a UUID, as
	// for example used to [identify Docker
	// containers](https://docs.docker.com/engine/reference/run/#container-identification).
	// The UUID might be abbreviated.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'a3bf90e006b2'
	ContainerIDKey = attribute.Key("container.id")

	// ContainerRuntimeKey is the attribute Key conforming to the
	// "container.runtime" semantic conventions. It represents the container
	// runtime managing this container.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'docker', 'containerd', 'rkt'
	ContainerRuntimeKey = attribute.Key("container.runtime")

	// ContainerImageNameKey is the attribute Key conforming to the
	// "container.image.name" semantic conventions. It represents the name of
	// the image the container was built on.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'gcr.io/opentelemetry/operator'
	ContainerImageNameKey = attribute.Key("container.image.name")

	// ContainerImageTagKey is the attribute Key conforming to the
	// "container.image.tag" semantic conventions. It represents the container
	// image tag.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '0.1'
	ContainerImageTagKey = attribute.Key("container.image.tag")
)

// ContainerName returns an attribute KeyValue conforming to the
// "container.name" semantic conventions. It represents the container name used
// by container runtime.
func ContainerName(val string) attribute.KeyValue {
	return ContainerNameKey.String(val)
}

// ContainerID returns an attribute KeyValue conforming to the
// "container.id" semantic conventions. It represents the container ID. Usually
// a UUID, as for example used to [identify Docker
// containers](https://docs.docker.com/engine/reference/run/#container-identification).
// The UUID might be abbreviated.
func ContainerID(val string) attribute.KeyValue {
	return ContainerIDKey.String(val)
}

// ContainerRuntime returns an attribute KeyValue conforming to the
// "container.runtime" semantic conventions. It represents the container
// runtime managing this container.
func ContainerRuntime(val string) attribute.KeyValue {
	return ContainerRuntimeKey.String(val)
}

// ContainerImageName returns an attribute KeyValue conforming to the
// "container.image.name" semantic conventions. It represents the name of the
// image the container was built on.
func ContainerImageName(val string) attribute.KeyValue {
	return ContainerImageNameKey.String(val)
}

// ContainerImageTag returns an attribute KeyValue conforming to the
// "container.image.tag" semantic conventions. It represents the container
// image tag.
func ContainerImageTag(val string) attribute.KeyValue {
	return ContainerImageTagKey.String(val)
}

// The software deployment.
const (
	// DeploymentEnvironmentKey is the attribute Key conforming to the
	// "deployment.environment" semantic conventions. It represents the name of
	// the [deployment
	// environment](https://en.wikipedia.org/wiki/Deployment_environment) (aka
	// deployment tier).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'staging', 'production'
	DeploymentEnvironmentKey = attribute.Key("deployment.environment")
)

// DeploymentEnvironment returns an attribute KeyValue conforming to the
// "deployment.environment" semantic conventions. It represents the name of the
// [deployment
// environment](https://en.wikipedia.org/wiki/Deployment_environment) (aka
// deployment tier).
func DeploymentEnvironment(val string) attribute.KeyValue {
	return DeploymentEnvironmentKey.String(val)
}

// The device on which the process represented by this resource is running.
const (
	// DeviceIDKey is the attribute Key conforming to the "device.id" semantic
	// conventions. It represents a unique identifier representing the device
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '2ab2916d-a51f-4ac8-80ee-45ac31a28092'
	// Note: The device identifier MUST only be defined using the values
	// outlined below. This value is not an advertising identifier and MUST NOT
	// be used as such. On iOS (Swift or Objective-C), this value MUST be equal
	// to the [vendor
	// identifier](https://developer.apple.com/documentation/uikit/uidevice/1620059-identifierforvendor).
	// On Android (Java or Kotlin), this value MUST be equal to the Firebase
	// Installation ID or a globally unique UUID which is persisted across
	// sessions in your application. More information can be found
	// [here](https://developer.android.com/training/articles/user-data-ids) on
	// best practices and exact implementation details. Caution should be taken
	// when storing personal data or anything which can identify a user. GDPR
	// and data protection laws may apply, ensure you do your own due
	// diligence.
	DeviceIDKey = attribute.Key("device.id")

	// DeviceModelIdentifierKey is the attribute Key conforming to the
	// "device.model.identifier" semantic conventions. It represents the model
	// identifier for the device
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'iPhone3,4', 'SM-G920F'
	// Note: It's recommended this value represents a machine readable version
	// of the model identifier rather than the market or consumer-friendly name
	// of the device.
	DeviceModelIdentifierKey = attribute.Key("device.model.identifier")

	// DeviceModelNameKey is the attribute Key conforming to the
	// "device.model.name" semantic conventions. It represents the marketing
	// name for the device model
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'iPhone 6s Plus', 'Samsung Galaxy S6'
	// Note: It's recommended this value represents a human readable version of
	// the device model rather than a machine readable alternative.
	DeviceModelNameKey = attribute.Key("device.model.name")

	// DeviceManufacturerKey is the attribute Key conforming to the
	// "device.manufacturer" semantic conventions. It represents the name of
	// the device manufacturer
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'Apple', 'Samsung'
	// Note: The Android OS provides this field via
	// [Build](https://developer.android.com/reference/android/os/Build#MANUFACTURER).
	// iOS apps SHOULD hardcode the value `Apple`.
	DeviceManufacturerKey = attribute.Key("device.manufacturer")
)

// DeviceID returns an attribute KeyValue conforming to the "device.id"
// semantic conventions. It represents a unique identifier representing the
// device
func DeviceID(val string) attribute.KeyValue {
	return DeviceIDKey.String(val)
}

// DeviceModelIdentifier returns an attribute KeyValue conforming to the
// "device.model.identifier" semantic conventions. It represents the model
// identifier for the device
func DeviceModelIdentifier(val string) attribute.KeyValue {
	return DeviceModelIdentifierKey.String(val)
}

// DeviceModelName returns an attribute KeyValue conforming to the
// "device.model.name" semantic conventions. It represents the marketing name
// for the device model
func DeviceModelName(val string) attribute.KeyValue {
	return DeviceModelNameKey.String(val)
}

// DeviceManufacturer returns an attribute KeyValue conforming to the
// "device.manufacturer" semantic conventions. It represents the name of the
// device manufacturer
func DeviceManufacturer(val string) attribute.KeyValue {
	return DeviceManufacturerKey.String(val)
}

// A serverless instance.
const (
	// FaaSNameKey is the attribute Key conforming to the "faas.name" semantic
	// conventions. It represents the name of the single function that this
	// runtime instance executes.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'my-function', 'myazurefunctionapp/some-function-name'
	// Note: This is the name of the function as configured/deployed on the
	// FaaS
	// platform and is usually different from the name of the callback
	// function (which may be stored in the
	// [`code.namespace`/`code.function`](../../trace/semantic_conventions/span-general.md#source-code-attributes)
	// span attributes).
	//
	// For some cloud providers, the above definition is ambiguous. The
	// following
	// definition of function name MUST be used for this attribute
	// (and consequently the span name) for the listed cloud
	// providers/products:
	//
	// * **Azure:**  The full name `<FUNCAPP>/<FUNC>`, i.e., function app name
	//   followed by a forward slash followed by the function name (this form
	//   can also be seen in the resource JSON for the function).
	//   This means that a span attribute MUST be used, as an Azure function
	//   app can host multiple functions that would usually share
	//   a TracerProvider (see also the `faas.id` attribute).
	FaaSNameKey = attribute.Key("faas.name")

	// FaaSIDKey is the attribute Key conforming to the "faas.id" semantic
	// conventions. It represents the unique ID of the single function that
	// this runtime instance executes.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'arn:aws:lambda:us-west-2:123456789012:function:my-function'
	// Note: On some cloud providers, it may not be possible to determine the
	// full ID at startup,
	// so consider setting `faas.id` as a span attribute instead.
	//
	// The exact value to use for `faas.id` depends on the cloud provider:
	//
	// * **AWS Lambda:** The function
	// [ARN](https://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html).
	//   Take care not to use the "invoked ARN" directly but replace any
	//   [alias
	// suffix](https://docs.aws.amazon.com/lambda/latest/dg/configuration-aliases.html)
	//   with the resolved function version, as the same runtime instance may
	// be invokable with
	//   multiple different aliases.
	// * **GCP:** The [URI of the
	// resource](https://cloud.google.com/iam/docs/full-resource-names)
	// * **Azure:** The [Fully Qualified Resource
	// ID](https://docs.microsoft.com/en-us/rest/api/resources/resources/get-by-id)
	// of the invoked function,
	//   *not* the function app, having the form
	// `/subscriptions/<SUBSCIPTION_GUID>/resourceGroups/<RG>/providers/Microsoft.Web/sites/<FUNCAPP>/functions/<FUNC>`.
	//   This means that a span attribute MUST be used, as an Azure function
	// app can host multiple functions that would usually share
	//   a TracerProvider.
	FaaSIDKey = attribute.Key("faas.id")

	// FaaSVersionKey is the attribute Key conforming to the "faas.version"
	// semantic conventions. It represents the immutable version of the
	// function being executed.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '26', 'pinkfroid-00002'
	// Note: Depending on the cloud provider and platform, use:
	//
	// * **AWS Lambda:** The [function
	// version](https://docs.aws.amazon.com/lambda/latest/dg/configuration-versions.html)
	//   (an integer represented as a decimal string).
	// * **Google Cloud Run:** The
	// [revision](https://cloud.google.com/run/docs/managing/revisions)
	//   (i.e., the function name plus the revision suffix).
	// * **Google Cloud Functions:** The value of the
	//   [`K_REVISION` environment
	// variable](https://cloud.google.com/functions/docs/env-var#runtime_environment_variables_set_automatically).
	// * **Azure Functions:** Not applicable. Do not set this attribute.
	FaaSVersionKey = attribute.Key("faas.version")

	// FaaSInstanceKey is the attribute Key conforming to the "faas.instance"
	// semantic conventions. It represents the execution environment ID as a
	// string, that will be potentially reused for other invocations to the
	// same function/function version.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '2021/06/28/[$LATEST]2f399eb14537447da05ab2a2e39309de'
	// Note: * **AWS Lambda:** Use the (full) log stream name.
	FaaSInstanceKey = attribute.Key("faas.instance")

	// FaaSMaxMemoryKey is the attribute Key conforming to the
	// "faas.max_memory" semantic conventions. It represents the amount of
	// memory available to the serverless function in MiB.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 128
	// Note: It's recommended to set this attribute since e.g. too little
	// memory can easily stop a Java AWS Lambda function from working
	// correctly. On AWS Lambda, the environment variable
	// `AWS_LAMBDA_FUNCTION_MEMORY_SIZE` provides this information.
	FaaSMaxMemoryKey = attribute.Key("faas.max_memory")
)

// FaaSName returns an attribute KeyValue conforming to the "faas.name"
// semantic conventions. It represents the name of the single function that
// this runtime instance executes.
func FaaSName(val string) attribute.KeyValue {
	return FaaSNameKey.String(val)
}

// FaaSID returns an attribute KeyValue conforming to the "faas.id" semantic
// conventions. It represents the unique ID of the single function that this
// runtime instance executes.
func FaaSID(val string) attribute.KeyValue {
	return FaaSIDKey.String(val)
}

// FaaSVersion returns an attribute KeyValue conforming to the
// "faas.version" semantic conventions. It represents the immutable version of
// the function being executed.
func FaaSVersion(val string) attribute.KeyValue {
	return FaaSVersionKey.String(val)
}

// FaaSInstance returns an attribute KeyValue conforming to the
// "faas.instance" semantic conventions. It represents the execution
// environment ID as a string, that will be potentially reused for other
// invocations to the same function/function version.
func FaaSInstance(val string) attribute.KeyValue {
	return FaaSInstanceKey.String(val)
}

// FaaSMaxMemory returns an attribute KeyValue conforming to the
// "faas.max_memory" semantic conventions. It represents the amount of memory
// available to the serverless function in MiB.
func FaaSMaxMemory(val int) attribute.KeyValue {
	return FaaSMaxMemoryKey.Int(val)
}

// A host is defined as a general computing instance.
const (
	// HostIDKey is the attribute Key conforming to the "host.id" semantic
	// conventions. It represents the unique host ID. For Cloud, this must be
	// the instance_id assigned by the cloud provider. For non-containerized
	// Linux systems, the `machine-id` located in `/etc/machine-id` or
	// `/var/lib/dbus/machine-id` may be used.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'fdbf79e8af94cb7f9e8df36789187052'
	HostIDKey = attribute.Key("host.id")

	// HostNameKey is the attribute Key conforming to the "host.name" semantic
	// conventions. It represents the name of the host. On Unix systems, it may
	// contain what the hostname command returns, or the fully qualified
	// hostname, or another name specified by the user.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'opentelemetry-test'
	HostNameKey = attribute.Key("host.name")

	// HostTypeKey is the attribute Key conforming to the "host.type" semantic
	// conventions. It represents the type of host. For Cloud, this must be the
	// machine type.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'n1-standard-1'
	HostTypeKey = attribute.Key("host.type")

	// HostArchKey is the attribute Key conforming to the "host.arch" semantic
	// conventions. It represents the CPU architecture the host system is
	// running on.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	HostArchKey = attribute.Key("host.arch")

	// HostImageNameKey is the attribute Key conforming to the
	// "host.image.name" semantic conventions. It represents the name of the VM
	// image or OS install the host was instantiated from.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'infra-ami-eks-worker-node-7d4ec78312', 'CentOS-8-x86_64-1905'
	HostImageNameKey = attribute.Key("host.image.name")

	// HostImageIDKey is the attribute Key conforming to the "host.image.id"
	// semantic conventions. It represents the vM image ID. For Cloud, this
	// value is from the provider.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'ami-07b06b442921831e5'
	HostImageIDKey = attribute.Key("host.image.id")

	// HostImageVersionKey is the attribute Key conforming to the
	// "host.image.version" semantic conventions. It represents the version
	// string of the VM image as defined in [Version
	// Attributes](README.md#version-attributes).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '0.1'
	HostImageVersionKey = attribute.Key("host.image.version")
)

var (
	// AMD64
	HostArchAMD64 = HostArchKey.String("amd64")
	// ARM32
	HostArchARM32 = HostArchKey.String("arm32")
	// ARM64
	HostArchARM64 = HostArchKey.String("arm64")
	// Itanium
	HostArchIA64 = HostArchKey.String("ia64")
	// 32-bit PowerPC
	HostArchPPC32 = HostArchKey.String("ppc32")
	// 64-bit PowerPC
	HostArchPPC64 = HostArchKey.String("ppc64")
	// IBM z/Architecture
	HostArchS390x = HostArchKey.String("s390x")
	// 32-bit x86
	HostArchX86 = HostArchKey.String("x86")
)

// HostID returns an attribute KeyValue conforming to the "host.id" semantic
// conventions. It represents the unique host ID. For Cloud, this must be the
// instance_id assigned by the cloud provider. For non-containerized Linux
// systems, the `machine-id` located in `/etc/machine-id` or
// `/var/lib/dbus/machine-id` may be used.
func HostID(val string) attribute.KeyValue {
	return HostIDKey.String(val)
}

// HostName returns an attribute KeyValue conforming to the "host.name"
// semantic conventions. It represents the name of the host. On Unix systems,
// it may contain what the hostname command returns, or the fully qualified
// hostname, or another name specified by the user.
func HostName(val string) attribute.KeyValue {
	return HostNameKey.String(val)
}

// HostType returns an attribute KeyValue conforming to the "host.type"
// semantic conventions. It represents the type of host. For Cloud, this must
// be the machine type.
func HostType(val string) attribute.KeyValue {
	return HostTypeKey.String(val)
}

// HostImageName returns an attribute KeyValue conforming to the
// "host.image.name" semantic conventions. It represents the name of the VM
// image or OS install the host was instantiated from.
func HostImageName(val string) attribute.KeyValue {
	return HostImageNameKey.String(val)
}

// HostImageID returns an attribute KeyValue conforming to the
// "host.image.id" semantic conventions. It represents the vM image ID. For
// Cloud, this value is from the provider.
func HostImageID(val string) attribute.KeyValue {
	return HostImageIDKey.String(val)
}

// HostImageVersion returns an attribute KeyValue conforming to the
// "host.image.version" semantic conventions. It represents the version string
// of the VM image as defined in [Version
// Attributes](README.md#version-attributes).
func HostImageVersion(val string) attribute.KeyValue {
	return HostImageVersionKey.String(val)
}

// A Kubernetes Cluster.
const (
	// K8SClusterNameKey is the attribute Key conforming to the
	// "k8s.cluster.name" semantic conventions. It represents the name of the
	// cluster.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'opentelemetry-cluster'
	K8SClusterNameKey = attribute.Key("k8s.cluster.name")
)

// K8SClusterName returns an attribute KeyValue conforming to the
// "k8s.cluster.name" semantic conventions. It represents the name of the
// cluster.
func K8SClusterName(val string) attribute.KeyValue {
	return K8SClusterNameKey.String(val)
}

// A Kubernetes Node object.
const (
	// K8SNodeNameKey is the attribute Key conforming to the "k8s.node.name"
	// semantic conventions. It represents the name of the Node.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'node-1'
	K8SNodeNameKey = attribute.Key("k8s.node.name")

	// K8SNodeUIDKey is the attribute Key conforming to the "k8s.node.uid"
	// semantic conventions. It represents the UID of the Node.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '1eb3a0c6-0477-4080-a9cb-0cb7db65c6a2'
	K8SNodeUIDKey = attribute.Key("k8s.node.uid")
)

// K8SNodeName returns an attribute KeyValue conforming to the
// "k8s.node.name" semantic conventions. It represents the name of the Node.
func K8SNodeName(val string) attribute.KeyValue {
	return K8SNodeNameKey.String(val)
}

// K8SNodeUID returns an attribute KeyValue conforming to the "k8s.node.uid"
// semantic conventions. It represents the UID of the Node.
func K8SNodeUID(val string) attribute.KeyValue {
	return K8SNodeUIDKey.String(val)
}

// A Kubernetes Namespace.
const (
	// K8SNamespaceNameKey is the attribute Key conforming to the
	// "k8s.namespace.name" semantic conventions. It represents the name of the
	// namespace that the pod is running in.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'default'
	K8SNamespaceNameKey = attribute.Key("k8s.namespace.name")
)

// K8SNamespaceName returns an attribute KeyValue conforming to the
// "k8s.namespace.name" semantic conventions. It represents the name of the
// namespace that the pod is running in.
func K8SNamespaceName(val string) attribute.KeyValue {
	return K8SNamespaceNameKey.String(val)
}

// A Kubernetes Pod object.
const (
	// K8SPodUIDKey is the attribute Key conforming to the "k8s.pod.uid"
	// semantic conventions. It represents the UID of the Pod.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '275ecb36-5aa8-4c2a-9c47-d8bb681b9aff'
	K8SPodUIDKey = attribute.Key("k8s.pod.uid")

	// K8SPodNameKey is the attribute Key conforming to the "k8s.pod.name"
	// semantic conventions. It represents the name of the Pod.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'opentelemetry-pod-autoconf'
	K8SPodNameKey = attribute.Key("k8s.pod.name")
)

// K8SPodUID returns an attribute KeyValue conforming to the "k8s.pod.uid"
// semantic conventions. It represents the UID of the Pod.
func K8SPodUID(val string) attribute.KeyValue {
	return K8SPodUIDKey.String(val)
}

// K8SPodName returns an attribute KeyValue conforming to the "k8s.pod.name"
// semantic conventions. It represents the name of the Pod.
func K8SPodName(val string) attribute.KeyValue {
	return K8SPodNameKey.String(val)
}

// A container in a
// [PodTemplate](https://kubernetes.io/docs/concepts/workloads/pods/#pod-templates).
const (
	// K8SContainerNameKey is the attribute Key conforming to the
	// "k8s.container.name" semantic conventions. It represents the name of the
	// Container from Pod specification, must be unique within a Pod. Container
	// runtime usually uses different globally unique name (`container.name`).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'redis'
	K8SContainerNameKey = attribute.Key("k8s.container.name")

	// K8SContainerRestartCountKey is the attribute Key conforming to the
	// "k8s.container.restart_count" semantic conventions. It represents the
	// number of times the container was restarted. This attribute can be used
	// to identify a particular container (running or stopped) within a
	// container spec.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 0, 2
	K8SContainerRestartCountKey = attribute.Key("k8s.container.restart_count")
)

// K8SContainerName returns an attribute KeyValue conforming to the
// "k8s.container.name" semantic conventions. It represents the name of the
// Container from Pod specification, must be unique within a Pod. Container
// runtime usually uses different globally unique name (`container.name`).
func K8SContainerName(val string) attribute.KeyValue {
	return K8SContainerNameKey.String(val)
}

// K8SContainerRestartCount returns an attribute KeyValue conforming to the
// "k8s.container.restart_count" semantic conventions. It represents the number
// of times the container was restarted. This attribute can be used to identify
// a particular container (running or stopped) within a container spec.
func K8SContainerRestartCount(val int) attribute.KeyValue {
	return K8SContainerRestartCountKey.Int(val)
}

// A Kubernetes ReplicaSet object.
const (
	// K8SReplicaSetUIDKey is the attribute Key conforming to the
	// "k8s.replicaset.uid" semantic conventions. It represents the UID of the
	// ReplicaSet.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '275ecb36-5aa8-4c2a-9c47-d8bb681b9aff'
	K8SReplicaSetUIDKey = attribute.Key("k8s.replicaset.uid")

	// K8SReplicaSetNameKey is the attribute Key conforming to the
	// "k8s.replicaset.name" semantic conventions. It represents the name of
	// the ReplicaSet.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'opentelemetry'
	K8SReplicaSetNameKey = attribute.Key("k8s.replicaset.name")
)

// K8SReplicaSetUID returns an attribute KeyValue conforming to the
// "k8s.replicaset.uid" semantic conventions. It represents the UID of the
// ReplicaSet.
func K8SReplicaSetUID(val string) attribute.KeyValue {
	return K8SReplicaSetUIDKey.String(val)
}

// K8SReplicaSetName returns an attribute KeyValue conforming to the
// "k8s.replicaset.name" semantic conventions. It represents the name of the
// ReplicaSet.
func K8SReplicaSetName(val string) attribute.KeyValue {
	return K8SReplicaSetNameKey.String(val)
}

// A Kubernetes Deployment object.
const (
	// K8SDeploymentUIDKey is the attribute Key conforming to the
	// "k8s.deployment.uid" semantic conventions. It represents the UID of the
	// Deployment.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '275ecb36-5aa8-4c2a-9c47-d8bb681b9aff'
	K8SDeploymentUIDKey = attribute.Key("k8s.deployment.uid")

	// K8SDeploymentNameKey is the attribute Key conforming to the
	// "k8s.deployment.name" semantic conventions. It represents the name of
	// the Deployment.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'opentelemetry'
	K8SDeploymentNameKey = attribute.Key("k8s.deployment.name")
)

// K8SDeploymentUID returns an attribute KeyValue conforming to the
// "k8s.deployment.uid" semantic conventions. It represents the UID of the
// Deployment.
func K8SDeploymentUID(val string) attribute.KeyValue {
	return K8SDeploymentUIDKey.String(val)
}

// K8SDeploymentName returns an attribute KeyValue conforming to the
// "k8s.deployment.name" semantic conventions. It represents the name of the
// Deployment.
func K8SDeploymentName(val string) attribute.KeyValue {
	return K8SDeploymentNameKey.String(val)
}

// A Kubernetes StatefulSet object.
const (
	// K8SStatefulSetUIDKey is the attribute Key conforming to the
	// "k8s.statefulset.uid" semantic conventions. It represents the UID of the
	// StatefulSet.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '275ecb36-5aa8-4c2a-9c47-d8bb681b9aff'
	K8SStatefulSetUIDKey = attribute.Key("k8s.statefulset.uid")

	// K8SStatefulSetNameKey is the attribute Key conforming to the
	// "k8s.statefulset.name" semantic conventions. It represents the name of
	// the StatefulSet.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'opentelemetry'
	K8SStatefulSetNameKey = attribute.Key("k8s.statefulset.name")
)

// K8SStatefulSetUID returns an attribute KeyValue conforming to the
// "k8s.statefulset.uid" semantic conventions. It represents the UID of the
// StatefulSet.
func K8SStatefulSetUID(val string) attribute.KeyValue {
	return K8SStatefulSetUIDKey.String(val)
}

// K8SStatefulSetName returns an attribute KeyValue conforming to the
// "k8s.statefulset.name" semantic conventions. It represents the name of the
// StatefulSet.
func K8SStatefulSetName(val string) attribute.KeyValue {
	return K8SStatefulSetNameKey.String(val)
}

// A Kubernetes DaemonSet object.
const (
	// K8SDaemonSetUIDKey is the attribute Key conforming to the
	// "k8s.daemonset.uid" semantic conventions. It represents the UID of the
	// DaemonSet.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '275ecb36-5aa8-4c2a-9c47-d8bb681b9aff'
	K8SDaemonSetUIDKey = attribute.Key("k8s.daemonset.uid")

	// K8SDaemonSetNameKey is the attribute Key conforming to the
	// "k8s.daemonset.name" semantic conventions. It represents the name of the
	// DaemonSet.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'opentelemetry'
	K8SDaemonSetNameKey = attribute.Key("k8s.daemonset.name")
)

// K8SDaemonSetUID returns an attribute KeyValue conforming to the
// "k8s.daemonset.uid" semantic conventions. It represents the UID of the
// DaemonSet.
func K8SDaemonSetUID(val string) attribute.KeyValue {
	return K8SDaemonSetUIDKey.String(val)
}

// K8SDaemonSetName returns an attribute KeyValue conforming to the
// "k8s.daemonset.name" semantic conventions. It represents the name of the
// DaemonSet.
func K8SDaemonSetName(val string) attribute.KeyValue {
	return K8SDaemonSetNameKey.String(val)
}

// A Kubernetes Job object.
const (
	// K8SJobUIDKey is the attribute Key conforming to the "k8s.job.uid"
	// semantic conventions. It represents the UID of the Job.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '275ecb36-5aa8-4c2a-9c47-d8bb681b9aff'
	K8SJobUIDKey = attribute.Key("k8s.job.uid")

	// K8SJobNameKey is the attribute Key conforming to the "k8s.job.name"
	// semantic conventions. It represents the name of the Job.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'opentelemetry'
	K8SJobNameKey = attribute.Key("k8s.job.name")
)

// K8SJobUID returns an attribute KeyValue conforming to the "k8s.job.uid"
// semantic conventions. It represents the UID of the Job.
func K8SJobUID(val string) attribute.KeyValue {
	return K8SJobUIDKey.String(val)
}

// K8SJobName returns an attribute KeyValue conforming to the "k8s.job.name"
// semantic conventions. It represents the name of the Job.
func K8SJobName(val string) attribute.KeyValue {
	return K8SJobNameKey.String(val)
}

// A Kubernetes CronJob object.
const (
	// K8SCronJobUIDKey is the attribute Key conforming to the
	// "k8s.cronjob.uid" semantic conventions. It represents the UID of the
	// CronJob.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '275ecb36-5aa8-4c2a-9c47-d8bb681b9aff'
	K8SCronJobUIDKey = attribute.Key("k8s.cronjob.uid")

	// K8SCronJobNameKey is the attribute Key conforming to the
	// "k8s.cronjob.name" semantic conventions. It represents the name of the
	// CronJob.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'opentelemetry'
	K8SCronJobNameKey = attribute.Key("k8s.cronjob.name")
)

// K8SCronJobUID returns an attribute KeyValue conforming to the
// "k8s.cronjob.uid" semantic conventions. It represents the UID of the
// CronJob.
func K8SCronJobUID(val string) attribute.KeyValue {
	return K8SCronJobUIDKey.String(val)
}

// K8SCronJobName returns an attribute KeyValue conforming to the
// "k8s.cronjob.name" semantic conventions. It represents the name of the
// CronJob.
func K8SCronJobName(val string) attribute.KeyValue {
	return K8SCronJobNameKey.String(val)
}

// The operating system (OS) on which the process represented by this resource
// is running.
const (
	// OSTypeKey is the attribute Key conforming to the "os.type" semantic
	// conventions. It represents the operating system type.
	//
	// Type: Enum
	// RequirementLevel: Required
	// Stability: stable
	OSTypeKey = attribute.Key("os.type")

	// OSDescriptionKey is the attribute Key conforming to the "os.description"
	// semantic conventions. It represents the human readable (not intended to
	// be parsed) OS version information, like e.g. reported by `ver` or
	// `lsb_release -a` commands.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'Microsoft Windows [Version 10.0.18363.778]', 'Ubuntu 18.04.1
	// LTS'
	OSDescriptionKey = attribute.Key("os.description")

	// OSNameKey is the attribute Key conforming to the "os.name" semantic
	// conventions. It represents the human readable operating system name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'iOS', 'Android', 'Ubuntu'
	OSNameKey = attribute.Key("os.name")

	// OSVersionKey is the attribute Key conforming to the "os.version"
	// semantic conventions. It represents the version string of the operating
	// system as defined in [Version
	// Attributes](../../resource/semantic_conventions/README.md#version-attributes).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '14.2.1', '18.04.1'
	OSVersionKey = attribute.Key("os.version")
)

var (
	// Microsoft Windows
	OSTypeWindows = OSTypeKey.String("windows")
	// Linux
	OSTypeLinux = OSTypeKey.String("linux")
	// Apple Darwin
	OSTypeDarwin = OSTypeKey.String("darwin")
	// FreeBSD
	OSTypeFreeBSD = OSTypeKey.String("freebsd")
	// NetBSD
	OSTypeNetBSD = OSTypeKey.String("netbsd")
	// OpenBSD
	OSTypeOpenBSD = OSTypeKey.String("openbsd")
	// DragonFly BSD
	OSTypeDragonflyBSD = OSTypeKey.String("dragonflybsd")
	// HP-UX (Hewlett Packard Unix)
	OSTypeHPUX = OSTypeKey.String("hpux")
	// AIX (Advanced Interactive eXecutive)
	OSTypeAIX = OSTypeKey.String("aix")
	// SunOS, Oracle Solaris
	OSTypeSolaris = OSTypeKey.String("solaris")
	// IBM z/OS
	OSTypeZOS = OSTypeKey.String("z_os")
)

// OSDescription returns an attribute KeyValue conforming to the
// "os.description" semantic conventions. It represents the human readable (not
// intended to be parsed) OS version information, like e.g. reported by `ver`
// or `lsb_release -a` commands.
func OSDescription(val string) attribute.KeyValue {
	return OSDescriptionKey.String(val)
}

// OSName returns an attribute KeyValue conforming to the "os.name" semantic
// conventions. It represents the human readable operating system name.
func OSName(val string) attribute.KeyValue {
	return OSNameKey.String(val)
}

// OSVersion returns an attribute KeyValue conforming to the "os.version"
// semantic conventions. It represents the version string of the operating
// system as defined in [Version
// Attributes](../../resource/semantic_conventions/README.md#version-attributes).
func OSVersion(val string) attribute.KeyValue {
	return OSVersionKey.String(val)
}

// An operating system process.
const (
	// ProcessPIDKey is the attribute Key conforming to the "process.pid"
	// semantic conventions. It represents the process identifier (PID).
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 1234
	ProcessPIDKey = attribute.Key("process.pid")

	// ProcessParentPIDKey is the attribute Key conforming to the
	// "process.parent_pid" semantic conventions. It represents the parent
	// Process identifier (PID).
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 111
	ProcessParentPIDKey = attribute.Key("process.parent_pid")

	// ProcessExecutableNameKey is the attribute Key conforming to the
	// "process.executable.name" semantic conventions. It represents the name
	// of the process executable. On Linux based systems, can be set to the
	// `Name` in `proc/[pid]/status`. On Windows, can be set to the base name
	// of `GetProcessImageFileNameW`.
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (See alternative attributes
	// below.)
	// Stability: stable
	// Examples: 'otelcol'
	ProcessExecutableNameKey = attribute.Key("process.executable.name")

	// ProcessExecutablePathKey is the attribute Key conforming to the
	// "process.executable.path" semantic conventions. It represents the full
	// path to the process executable. On Linux based systems, can be set to
	// the target of `proc/[pid]/exe`. On Windows, can be set to the result of
	// `GetProcessImageFileNameW`.
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (See alternative attributes
	// below.)
	// Stability: stable
	// Examples: '/usr/bin/cmd/otelcol'
	ProcessExecutablePathKey = attribute.Key("process.executable.path")

	// ProcessCommandKey is the attribute Key conforming to the
	// "process.command" semantic conventions. It represents the command used
	// to launch the process (i.e. the command name). On Linux based systems,
	// can be set to the zeroth string in `proc/[pid]/cmdline`. On Windows, can
	// be set to the first parameter extracted from `GetCommandLineW`.
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (See alternative attributes
	// below.)
	// Stability: stable
	// Examples: 'cmd/otelcol'
	ProcessCommandKey = attribute.Key("process.command")

	// ProcessCommandLineKey is the attribute Key conforming to the
	// "process.command_line" semantic conventions. It represents the full
	// command used to launch the process as a single string representing the
	// full command. On Windows, can be set to the result of `GetCommandLineW`.
	// Do not set this if you have to assemble it just for monitoring; use
	// `process.command_args` instead.
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (See alternative attributes
	// below.)
	// Stability: stable
	// Examples: 'C:\\cmd\\otecol --config="my directory\\config.yaml"'
	ProcessCommandLineKey = attribute.Key("process.command_line")

	// ProcessCommandArgsKey is the attribute Key conforming to the
	// "process.command_args" semantic conventions. It represents the all the
	// command arguments (including the command/executable itself) as received
	// by the process. On Linux-based systems (and some other Unixoid systems
	// supporting procfs), can be set according to the list of null-delimited
	// strings extracted from `proc/[pid]/cmdline`. For libc-based executables,
	// this would be the full argv vector passed to `main`.
	//
	// Type: string[]
	// RequirementLevel: ConditionallyRequired (See alternative attributes
	// below.)
	// Stability: stable
	// Examples: 'cmd/otecol', '--config=config.yaml'
	ProcessCommandArgsKey = attribute.Key("process.command_args")

	// ProcessOwnerKey is the attribute Key conforming to the "process.owner"
	// semantic conventions. It represents the username of the user that owns
	// the process.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'root'
	ProcessOwnerKey = attribute.Key("process.owner")
)

// ProcessPID returns an attribute KeyValue conforming to the "process.pid"
// semantic conventions. It represents the process identifier (PID).
func ProcessPID(val int) attribute.KeyValue {
	return ProcessPIDKey.Int(val)
}

// ProcessParentPID returns an attribute KeyValue conforming to the
// "process.parent_pid" semantic conventions. It represents the parent Process
// identifier (PID).
func ProcessParentPID(val int) attribute.KeyValue {
	return ProcessParentPIDKey.Int(val)
}

// ProcessExecutableName returns an attribute KeyValue conforming to the
// "process.executable.name" semantic conventions. It represents the name of
// the process executable. On Linux based systems, can be set to the `Name` in
// `proc/[pid]/status`. On Windows, can be set to the base name of
// `GetProcessImageFileNameW`.
func ProcessExecutableName(val string) attribute.KeyValue {
	return ProcessExecutableNameKey.String(val)
}

// ProcessExecutablePath returns an attribute KeyValue conforming to the
// "process.executable.path" semantic conventions. It represents the full path
// to the process executable. On Linux based systems, can be set to the target
// of `proc/[pid]/exe`. On Windows, can be set to the result of
// `GetProcessImageFileNameW`.
func ProcessExecutablePath(val string) attribute.KeyValue {
	return ProcessExecutablePathKey.String(val)
}

// ProcessCommand returns an attribute KeyValue conforming to the
// "process.command" semantic conventions. It represents the command used to
// launch the process (i.e. the command name). On Linux based systems, can be
// set to the zeroth string in `proc/[pid]/cmdline`. On Windows, can be set to
// the first parameter extracted from `GetCommandLineW`.
func ProcessCommand(val string) attribute.KeyValue {
	return ProcessCommandKey.String(val)
}

// ProcessCommandLine returns an attribute KeyValue conforming to the
// "process.command_line" semantic conventions. It represents the full command
// used to launch the process as a single string representing the full command.
// On Windows, can be set to the result of `GetCommandLineW`. Do not set this
// if you have to assemble it just for monitoring; use `process.command_args`
// instead.
func ProcessCommandLine(val string) attribute.KeyValue {
	return ProcessCommandLineKey.String(val)
}

// ProcessCommandArgs returns an attribute KeyValue conforming to the
// "process.command_args" semantic conventions. It represents the all the
// command arguments (including the command/executable itself) as received by
// the process. On Linux-based systems (and some other Unixoid systems
// supporting procfs), can be set according to the list of null-delimited
// strings extracted from `proc/[pid]/cmdline`. For libc-based executables,
// this would be the full argv vector passed to `main`.
func ProcessCommandArgs(val ...string) attribute.KeyValue {
	return ProcessCommandArgsKey.StringSlice(val)
}

// ProcessOwner returns an attribute KeyValue conforming to the
// "process.owner" semantic conventions. It represents the username of the user
// that owns the process.
func ProcessOwner(val string) attribute.KeyValue {
	return ProcessOwnerKey.String(val)
}

// The single (language) runtime instance which is monitored.
const (
	// ProcessRuntimeNameKey is the attribute Key conforming to the
	// "process.runtime.name" semantic conventions. It represents the name of
	// the runtime of this process. For compiled native binaries, this SHOULD
	// be the name of the compiler.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'OpenJDK Runtime Environment'
	ProcessRuntimeNameKey = attribute.Key("process.runtime.name")

	// ProcessRuntimeVersionKey is the attribute Key conforming to the
	// "process.runtime.version" semantic conventions. It represents the
	// version of the runtime of this process, as returned by the runtime
	// without modification.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '14.0.2'
	ProcessRuntimeVersionKey = attribute.Key("process.runtime.version")

	// ProcessRuntimeDescriptionKey is the attribute Key conforming to the
	// "process.runtime.description" semantic conventions. It represents an
	// additional description about the runtime of the process, for example a
	// specific vendor customization of the runtime environment.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'Eclipse OpenJ9 Eclipse OpenJ9 VM openj9-0.21.0'
	ProcessRuntimeDescriptionKey = attribute.Key("process.runtime.description")
)

// ProcessRuntimeName returns an attribute KeyValue conforming to the
// "process.runtime.name" semantic conventions. It represents the name of the
// runtime of this process. For compiled native binaries, this SHOULD be the
// name of the compiler.
func ProcessRuntimeName(val string) attribute.KeyValue {
	return ProcessRuntimeNameKey.String(val)
}

// ProcessRuntimeVersion returns an attribute KeyValue conforming to the
// "process.runtime.version" semantic conventions. It represents the version of
// the runtime of this process, as returned by the runtime without
// modification.
func ProcessRuntimeVersion(val string) attribute.KeyValue {
	return ProcessRuntimeVersionKey.String(val)
}

// ProcessRuntimeDescription returns an attribute KeyValue conforming to the
// "process.runtime.description" semantic conventions. It represents an
// additional description about the runtime of the process, for example a
// specific vendor customization of the runtime environment.
func ProcessRuntimeDescription(val string) attribute.KeyValue {
	return ProcessRuntimeDescriptionKey.String(val)
}

// A service instance.
const (
	// ServiceNameKey is the attribute Key conforming to the "service.name"
	// semantic conventions. It represents the logical name of the service.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'shoppingcart'
	// Note: MUST be the same for all instances of horizontally scaled
	// services. If the value was not specified, SDKs MUST fallback to
	// `unknown_service:` concatenated with
	// [`process.executable.name`](process.md#process), e.g.
	// `unknown_service:bash`. If `process.executable.name` is not available,
	// the value MUST be set to `unknown_service`.
	ServiceNameKey = attribute.Key("service.name")

	// ServiceNamespaceKey is the attribute Key conforming to the
	// "service.namespace" semantic conventions. It represents a namespace for
	// `service.name`.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'Shop'
	// Note: A string value having a meaning that helps to distinguish a group
	// of services, for example the team name that owns a group of services.
	// `service.name` is expected to be unique within the same namespace. If
	// `service.namespace` is not specified in the Resource then `service.name`
	// is expected to be unique for all services that have no explicit
	// namespace defined (so the empty/unspecified namespace is simply one more
	// valid namespace). Zero-length namespace string is assumed equal to
	// unspecified namespace.
	ServiceNamespaceKey = attribute.Key("service.namespace")

	// ServiceInstanceIDKey is the attribute Key conforming to the
	// "service.instance.id" semantic conventions. It represents the string ID
	// of the service instance.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '627cc493-f310-47de-96bd-71410b7dec09'
	// Note: MUST be unique for each instance of the same
	// `service.namespace,service.name` pair (in other words
	// `service.namespace,service.name,service.instance.id` triplet MUST be
	// globally unique). The ID helps to distinguish instances of the same
	// service that exist at the same time (e.g. instances of a horizontally
	// scaled service). It is preferable for the ID to be persistent and stay
	// the same for the lifetime of the service instance, however it is
	// acceptable that the ID is ephemeral and changes during important
	// lifetime events for the service (e.g. service restarts). If the service
	// has no inherent unique ID that can be used as the value of this
	// attribute it is recommended to generate a random Version 1 or Version 4
	// RFC 4122 UUID (services aiming for reproducible UUIDs may also use
	// Version 5, see RFC 4122 for more recommendations).
	ServiceInstanceIDKey = attribute.Key("service.instance.id")

	// ServiceVersionKey is the attribute Key conforming to the
	// "service.version" semantic conventions. It represents the version string
	// of the service API or implementation.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '2.0.0'
	ServiceVersionKey = attribute.Key("service.version")
)

// ServiceName returns an attribute KeyValue conforming to the
// "service.name" semantic conventions. It represents the logical name of the
// service.
func ServiceName(val string) attribute.KeyValue {
	return ServiceNameKey.String(val)
}

// ServiceNamespace returns an attribute KeyValue conforming to the
// "service.namespace" semantic conventions. It represents a namespace for
// `service.name`.
func ServiceNamespace(val string) attribute.KeyValue {
	return ServiceNamespaceKey.String(val)
}

// ServiceInstanceID returns an attribute KeyValue conforming to the
// "service.instance.id" semantic conventions. It represents the string ID of
// the service instance.
func ServiceInstanceID(val string) attribute.KeyValue {
	return ServiceInstanceIDKey.String(val)
}

// ServiceVersion returns an attribute KeyValue conforming to the
// "service.version" semantic conventions. It represents the version string of
// the service API or implementation.
func ServiceVersion(val string) attribute.KeyValue {
	return ServiceVersionKey.String(val)
}

// The telemetry SDK used to capture data recorded by the instrumentation
// libraries.
const (
	// TelemetrySDKNameKey is the attribute Key conforming to the
	// "telemetry.sdk.name" semantic conventions. It represents the name of the
	// telemetry SDK as defined above.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'opentelemetry'
	TelemetrySDKNameKey = attribute.Key("telemetry.sdk.name")

	// TelemetrySDKLanguageKey is the attribute Key conforming to the
	// "telemetry.sdk.language" semantic conventions. It represents the
	// language of the telemetry SDK.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	TelemetrySDKLanguageKey = attribute.Key("telemetry.sdk.language")

	// TelemetrySDKVersionKey is the attribute Key conforming to the
	// "telemetry.sdk.version" semantic conventions. It represents the version
	// string of the telemetry SDK.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '1.2.3'
	TelemetrySDKVersionKey = attribute.Key("telemetry.sdk.version")

	// TelemetryAutoVersionKey is the attribute Key conforming to the
	// "telemetry.auto.version" semantic conventions. It represents the version
	// string of the auto instrumentation agent, if used.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '1.2.3'
	TelemetryAutoVersionKey = attribute.Key("telemetry.auto.version")
)

var (
	// cpp
	TelemetrySDKLanguageCPP = TelemetrySDKLanguageKey.String("cpp")
	// dotnet
	TelemetrySDKLanguageDotnet = TelemetrySDKLanguageKey.String("dotnet")
	// erlang
	TelemetrySDKLanguageErlang = TelemetrySDKLanguageKey.String("erlang")
	// go
	TelemetrySDKLanguageGo = TelemetrySDKLanguageKey.String("go")
	// java
	TelemetrySDKLanguageJava = TelemetrySDKLanguageKey.String("java")
	// nodejs
	TelemetrySDKLanguageNodejs = TelemetrySDKLanguageKey.String("nodejs")
	// php
	TelemetrySDKLanguagePHP = TelemetrySDKLanguageKey.String("php")
	// python
	TelemetrySDKLanguagePython = TelemetrySDKLanguageKey.String("python")
	// ruby
	TelemetrySDKLanguageRuby = TelemetrySDKLanguageKey.String("ruby")
	// webjs
	TelemetrySDKLanguageWebjs = TelemetrySDKLanguageKey.String("webjs")
	// swift
	TelemetrySDKLanguageSwift = TelemetrySDKLanguageKey.String("swift")
)

// TelemetrySDKName returns an attribute KeyValue conforming to the
// "telemetry.sdk.name" semantic conventions. It represents the name of the
// telemetry SDK as defined above.
func TelemetrySDKName(val string) attribute.KeyValue {
	return TelemetrySDKNameKey.String(val)
}

// TelemetrySDKVersion returns an attribute KeyValue conforming to the
// "telemetry.sdk.version" semantic conventions. It represents the version
// string of the telemetry SDK.
func TelemetrySDKVersion(val string) attribute.KeyValue {
	return TelemetrySDKVersionKey.String(val)
}

// TelemetryAutoVersion returns an attribute KeyValue conforming to the
// "telemetry.auto.version" semantic conventions. It represents the version
// string of the auto instrumentation agent, if used.
func TelemetryAutoVersion(val string) attribute.KeyValue {
	return TelemetryAutoVersionKey.String(val)
}

// Resource describing the packaged software running the application code. Web
// engines are typically executed using process.runtime.
const (
	// WebEngineNameKey is the attribute Key conforming to the "webengine.name"
	// semantic conventions. It represents the name of the web engine.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'WildFly'
	WebEngineNameKey = attribute.Key("webengine.name")

	// WebEngineVersionKey is the attribute Key conforming to the
	// "webengine.version" semantic conventions. It represents the version of
	// the web engine.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '21.0.0'
	WebEngineVersionKey = attribute.Key("webengine.version")

	// WebEngineDescriptionKey is the attribute Key conforming to the
	// "webengine.description" semantic conventions. It represents the
	// additional description of the web engine (e.g. detailed version and
	// edition information).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'WildFly Full 21.0.0.Final (WildFly Core 13.0.1.Final) -
	// 2.2.2.Final'
	WebEngineDescriptionKey = attribute.Key("webengine.description")
)

// WebEngineName returns an attribute KeyValue conforming to the
// "webengine.name" semantic conventions. It represents the name of the web
// engine.
func WebEngineName(val string) attribute.KeyValue {
	return WebEngineNameKey.String(val)
}

// WebEngineVersion returns an attribute KeyValue conforming to the
// "webengine.version" semantic conventions. It represents the version of the
// web engine.
func WebEngineVersion(val string) attribute.KeyValue {
	return WebEngineVersionKey.String(val)
}

// WebEngineDescription returns an attribute KeyValue conforming to the
// "webengine.description" semantic conventions. It represents the additional
// description of the web engine (e.g. detailed version and edition
// information).
func WebEngineDescription(val string) attribute.KeyValue {
	return WebEngineDescriptionKey.String(val)
}

// Attributes used by non-OTLP exporters to represent OpenTelemetry Scope's
// concepts.
const (
	// OtelScopeNameKey is the attribute Key conforming to the
	// "otel.scope.name" semantic conventions. It represents the name of the
	// instrumentation scope - (`InstrumentationScope.Name` in OTLP).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'io.opentelemetry.contrib.mongodb'
	OtelScopeNameKey = attribute.Key("otel.scope.name")

	// OtelScopeVersionKey is the attribute Key conforming to the
	// "otel.scope.version" semantic conventions. It represents the version of
	// the instrumentation scope - (`InstrumentationScope.Version` in OTLP).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '1.0.0'
	OtelScopeVersionKey = attribute.Key("otel.scope.version")
)

// OtelScopeName returns an attribute KeyValue conforming to the
// "otel.scope.name" semantic conventions. It represents the name of the
// instrumentation scope - (`InstrumentationScope.Name` in OTLP).
func OtelScopeName(val string) attribute.KeyValue {
	return OtelScopeNameKey.String(val)
}

// OtelScopeVersion returns an attribute KeyValue conforming to the
// "otel.scope.version" semantic conventions. It represents the version of the
// instrumentation scope - (`InstrumentationScope.Version` in OTLP).
func OtelScopeVersion(val string) attribute.KeyValue {
	return OtelScopeVersionKey.String(val)
}

// Span attributes used by non-OTLP exporters to represent OpenTelemetry
// Scope's concepts.
const (
	// OtelLibraryNameKey is the attribute Key conforming to the
	// "otel.library.name" semantic conventions. It represents the deprecated,
	// use the `otel.scope.name` attribute.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 'io.opentelemetry.contrib.mongodb'
	OtelLibraryNameKey = attribute.Key("otel.library.name")

	// OtelLibraryVersionKey is the attribute Key conforming to the
	// "otel.library.version" semantic conventions. It represents the
	// deprecated, use the `otel.scope.version` attribute.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: '1.0.0'
	OtelLibraryVersionKey = attribute.Key("otel.library.version")
)

// OtelLibraryName returns an attribute KeyValue conforming to the
// "otel.library.name" semantic conventions. It represents the deprecated, use
// the `otel.scope.name` attribute.
func OtelLibraryName(val string) attribute.KeyValue {
	return OtelLibraryNameKey.String(val)
}

// OtelLibraryVersion returns an attribute KeyValue conforming to the
// "otel.library.version" semantic conventions. It represents the deprecated,
// use the `otel.scope.version` attribute.
func OtelLibraryVersion(val string) attribute.KeyValue {
	return OtelLibraryVersionKey.String(val)
}
