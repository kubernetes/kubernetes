// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Code generated from semantic convention specification. DO NOT EDIT.

package semconv // import "go.opentelemetry.io/otel/semconv/v1.4.0"

import "go.opentelemetry.io/otel/attribute"

// A cloud environment (e.g. GCP, Azure, AWS)
const (
	// Name of the cloud provider.
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	// Examples: 'gcp'
	CloudProviderKey = attribute.Key("cloud.provider")
	// The cloud account ID the resource is assigned to.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '111111111111', 'opentelemetry'
	CloudAccountIDKey = attribute.Key("cloud.account.id")
	// The geographical region the resource is running. Refer to your provider's docs
	// to see the available regions, for example [AWS
	// regions](https://aws.amazon.com/about-aws/global-infrastructure/regions_az/),
	// [Azure regions](https://azure.microsoft.com/en-us/global-
	// infrastructure/geographies/), or [Google Cloud
	// regions](https://cloud.google.com/about/locations).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'us-central1', 'us-east-1'
	CloudRegionKey = attribute.Key("cloud.region")
	// Cloud regions often have multiple, isolated locations known as zones to
	// increase availability. Availability zone represents the zone where the resource
	// is running.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'us-east-1c'
	// Note: Availability zones are called "zones" on Google Cloud.
	CloudAvailabilityZoneKey = attribute.Key("cloud.availability_zone")
	// The cloud platform in use.
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	// Examples: 'aws_ec2', 'azure_vm', 'gcp_compute_engine'
	// Note: The prefix of the service SHOULD match the one specified in
	// `cloud.provider`.
	CloudPlatformKey = attribute.Key("cloud.platform")
)

var (
	// Amazon Web Services
	CloudProviderAWS = CloudProviderKey.String("aws")
	// Microsoft Azure
	CloudProviderAzure = CloudProviderKey.String("azure")
	// Google Cloud Platform
	CloudProviderGCP = CloudProviderKey.String("gcp")
)

var (
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
)

// Resources used by AWS Elastic Container Service (ECS).
const (
	// The Amazon Resource Name (ARN) of an [ECS container instance](https://docs.aws.
	// amazon.com/AmazonECS/latest/developerguide/ECS_instances.html).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'arn:aws:ecs:us-
	// west-1:123456789123:container/32624152-9086-4f0e-acae-1a75b14fe4d9'
	AWSECSContainerARNKey = attribute.Key("aws.ecs.container.arn")
	// The ARN of an [ECS cluster](https://docs.aws.amazon.com/AmazonECS/latest/develo
	// perguide/clusters.html).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'arn:aws:ecs:us-west-2:123456789123:cluster/my-cluster'
	AWSECSClusterARNKey = attribute.Key("aws.ecs.cluster.arn")
	// The [launch type](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/l
	// aunch_types.html) for an ECS task.
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	// Examples: 'ec2', 'fargate'
	AWSECSLaunchtypeKey = attribute.Key("aws.ecs.launchtype")
	// The ARN of an [ECS task definition](https://docs.aws.amazon.com/AmazonECS/lates
	// t/developerguide/task_definitions.html).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'arn:aws:ecs:us-
	// west-1:123456789123:task/10838bed-421f-43ef-870a-f43feacbbb5b'
	AWSECSTaskARNKey = attribute.Key("aws.ecs.task.arn")
	// The task definition family this task definition is a member of.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'opentelemetry-family'
	AWSECSTaskFamilyKey = attribute.Key("aws.ecs.task.family")
	// The revision for this task definition.
	//
	// Type: string
	// Required: No
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

// Resources used by AWS Elastic Kubernetes Service (EKS).
const (
	// The ARN of an EKS cluster.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'arn:aws:ecs:us-west-2:123456789123:cluster/my-cluster'
	AWSEKSClusterARNKey = attribute.Key("aws.eks.cluster.arn")
)

// Resources specific to Amazon Web Services.
const (
	// The name(s) of the AWS log group(s) an application is writing to.
	//
	// Type: string[]
	// Required: No
	// Stability: stable
	// Examples: '/aws/lambda/my-function', 'opentelemetry-service'
	// Note: Multiple log groups must be supported for cases like multi-container
	// applications, where a single application has sidecar containers, and each write
	// to their own log group.
	AWSLogGroupNamesKey = attribute.Key("aws.log.group.names")
	// The Amazon Resource Name(s) (ARN) of the AWS log group(s).
	//
	// Type: string[]
	// Required: No
	// Stability: stable
	// Examples: 'arn:aws:logs:us-west-1:123456789012:log-group:/aws/my/group:*'
	// Note: See the [log group ARN format
	// documentation](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/iam-
	// access-control-overview-cwl.html#CWL_ARN_Format).
	AWSLogGroupARNsKey = attribute.Key("aws.log.group.arns")
	// The name(s) of the AWS log stream(s) an application is writing to.
	//
	// Type: string[]
	// Required: No
	// Stability: stable
	// Examples: 'logs/main/10838bed-421f-43ef-870a-f43feacbbb5b'
	AWSLogStreamNamesKey = attribute.Key("aws.log.stream.names")
	// The ARN(s) of the AWS log stream(s).
	//
	// Type: string[]
	// Required: No
	// Stability: stable
	// Examples: 'arn:aws:logs:us-west-1:123456789012:log-group:/aws/my/group:log-
	// stream:logs/main/10838bed-421f-43ef-870a-f43feacbbb5b'
	// Note: See the [log stream ARN format
	// documentation](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/iam-
	// access-control-overview-cwl.html#CWL_ARN_Format). One log group can contain
	// several log streams, so these ARNs necessarily identify both a log group and a
	// log stream.
	AWSLogStreamARNsKey = attribute.Key("aws.log.stream.arns")
)

// A container instance.
const (
	// Container name.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'opentelemetry-autoconf'
	ContainerNameKey = attribute.Key("container.name")
	// Container ID. Usually a UUID, as for example used to [identify Docker
	// containers](https://docs.docker.com/engine/reference/run/#container-
	// identification). The UUID might be abbreviated.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'a3bf90e006b2'
	ContainerIDKey = attribute.Key("container.id")
	// The container runtime managing this container.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'docker', 'containerd', 'rkt'
	ContainerRuntimeKey = attribute.Key("container.runtime")
	// Name of the image the container was built on.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'gcr.io/opentelemetry/operator'
	ContainerImageNameKey = attribute.Key("container.image.name")
	// Container image tag.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '0.1'
	ContainerImageTagKey = attribute.Key("container.image.tag")
)

// The software deployment.
const (
	// Name of the [deployment
	// environment](https://en.wikipedia.org/wiki/Deployment_environment) (aka
	// deployment tier).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'staging', 'production'
	DeploymentEnvironmentKey = attribute.Key("deployment.environment")
)

// The device on which the process represented by this resource is running.
const (
	// A unique identifier representing the device
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '2ab2916d-a51f-4ac8-80ee-45ac31a28092'
	// Note: The device identifier MUST only be defined using the values outlined
	// below. This value is not an advertising identifier and MUST NOT be used as
	// such. On iOS (Swift or Objective-C), this value MUST be equal to the [vendor id
	// entifier](https://developer.apple.com/documentation/uikit/uidevice/1620059-iden
	// tifierforvendor). On Android (Java or Kotlin), this value MUST be equal to the
	// Firebase Installation ID or a globally unique UUID which is persisted across
	// sessions in your application. More information can be found
	// [here](https://developer.android.com/training/articles/user-data-ids) on best
	// practices and exact implementation details. Caution should be taken when
	// storing personal data or anything which can identify a user. GDPR and data
	// protection laws may apply, ensure you do your own due diligence.
	DeviceIDKey = attribute.Key("device.id")
	// The model identifier for the device
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'iPhone3,4', 'SM-G920F'
	// Note: It's recommended this value represents a machine readable version of the
	// model identifier rather than the market or consumer-friendly name of the
	// device.
	DeviceModelIdentifierKey = attribute.Key("device.model.identifier")
	// The marketing name for the device model
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'iPhone 6s Plus', 'Samsung Galaxy S6'
	// Note: It's recommended this value represents a human readable version of the
	// device model rather than a machine readable alternative.
	DeviceModelNameKey = attribute.Key("device.model.name")
)

// A serverless instance.
const (
	// The name of the function being executed.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'my-function'
	FaaSNameKey = attribute.Key("faas.name")
	// The unique ID of the function being executed.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'arn:aws:lambda:us-west-2:123456789012:function:my-function'
	// Note: For example, in AWS Lambda this field corresponds to the
	// [ARN](https://docs.aws.amazon.com/general/latest/gr/aws-arns-and-
	// namespaces.html) value, in GCP to the URI of the resource, and in Azure to the
	// [FunctionDirectory](https://github.com/Azure/azure-functions-
	// host/wiki/Retrieving-information-about-the-currently-running-function) field.
	FaaSIDKey = attribute.Key("faas.id")
	// The version string of the function being executed as defined in [Version
	// Attributes](../../resource/semantic_conventions/README.md#version-attributes).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '2.0.0'
	FaaSVersionKey = attribute.Key("faas.version")
	// The execution environment ID as a string.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'my-function:instance-0001'
	FaaSInstanceKey = attribute.Key("faas.instance")
	// The amount of memory available to the serverless function in MiB.
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 128
	// Note: It's recommended to set this attribute since e.g. too little memory can
	// easily stop a Java AWS Lambda function from working correctly. On AWS Lambda,
	// the environment variable `AWS_LAMBDA_FUNCTION_MEMORY_SIZE` provides this
	// information.
	FaaSMaxMemoryKey = attribute.Key("faas.max_memory")
)

// A host is defined as a general computing instance.
const (
	// Unique host ID. For Cloud, this must be the instance_id assigned by the cloud
	// provider.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'opentelemetry-test'
	HostIDKey = attribute.Key("host.id")
	// Name of the host. On Unix systems, it may contain what the hostname command
	// returns, or the fully qualified hostname, or another name specified by the
	// user.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'opentelemetry-test'
	HostNameKey = attribute.Key("host.name")
	// Type of host. For Cloud, this must be the machine type.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'n1-standard-1'
	HostTypeKey = attribute.Key("host.type")
	// The CPU architecture the host system is running on.
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	HostArchKey = attribute.Key("host.arch")
	// Name of the VM image or OS install the host was instantiated from.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'infra-ami-eks-worker-node-7d4ec78312', 'CentOS-8-x86_64-1905'
	HostImageNameKey = attribute.Key("host.image.name")
	// VM image ID. For Cloud, this value is from the provider.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'ami-07b06b442921831e5'
	HostImageIDKey = attribute.Key("host.image.id")
	// The version string of the VM image as defined in [Version
	// Attributes](README.md#version-attributes).
	//
	// Type: string
	// Required: No
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
	// 32-bit x86
	HostArchX86 = HostArchKey.String("x86")
)

// A Kubernetes Cluster.
const (
	// The name of the cluster.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'opentelemetry-cluster'
	K8SClusterNameKey = attribute.Key("k8s.cluster.name")
)

// A Kubernetes Node object.
const (
	// The name of the Node.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'node-1'
	K8SNodeNameKey = attribute.Key("k8s.node.name")
	// The UID of the Node.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '1eb3a0c6-0477-4080-a9cb-0cb7db65c6a2'
	K8SNodeUIDKey = attribute.Key("k8s.node.uid")
)

// A Kubernetes Namespace.
const (
	// The name of the namespace that the pod is running in.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'default'
	K8SNamespaceNameKey = attribute.Key("k8s.namespace.name")
)

// A Kubernetes Pod object.
const (
	// The UID of the Pod.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '275ecb36-5aa8-4c2a-9c47-d8bb681b9aff'
	K8SPodUIDKey = attribute.Key("k8s.pod.uid")
	// The name of the Pod.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'opentelemetry-pod-autoconf'
	K8SPodNameKey = attribute.Key("k8s.pod.name")
)

// A container in a [PodTemplate](https://kubernetes.io/docs/concepts/workloads/pods/#pod-templates).
const (
	// The name of the Container in a Pod template.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'redis'
	K8SContainerNameKey = attribute.Key("k8s.container.name")
)

// A Kubernetes ReplicaSet object.
const (
	// The UID of the ReplicaSet.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '275ecb36-5aa8-4c2a-9c47-d8bb681b9aff'
	K8SReplicasetUIDKey = attribute.Key("k8s.replicaset.uid")
	// The name of the ReplicaSet.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'opentelemetry'
	K8SReplicasetNameKey = attribute.Key("k8s.replicaset.name")
)

// A Kubernetes Deployment object.
const (
	// The UID of the Deployment.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '275ecb36-5aa8-4c2a-9c47-d8bb681b9aff'
	K8SDeploymentUIDKey = attribute.Key("k8s.deployment.uid")
	// The name of the Deployment.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'opentelemetry'
	K8SDeploymentNameKey = attribute.Key("k8s.deployment.name")
)

// A Kubernetes StatefulSet object.
const (
	// The UID of the StatefulSet.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '275ecb36-5aa8-4c2a-9c47-d8bb681b9aff'
	K8SStatefulsetUIDKey = attribute.Key("k8s.statefulset.uid")
	// The name of the StatefulSet.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'opentelemetry'
	K8SStatefulsetNameKey = attribute.Key("k8s.statefulset.name")
)

// A Kubernetes DaemonSet object.
const (
	// The UID of the DaemonSet.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '275ecb36-5aa8-4c2a-9c47-d8bb681b9aff'
	K8SDaemonsetUIDKey = attribute.Key("k8s.daemonset.uid")
	// The name of the DaemonSet.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'opentelemetry'
	K8SDaemonsetNameKey = attribute.Key("k8s.daemonset.name")
)

// A Kubernetes Job object.
const (
	// The UID of the Job.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '275ecb36-5aa8-4c2a-9c47-d8bb681b9aff'
	K8SJobUIDKey = attribute.Key("k8s.job.uid")
	// The name of the Job.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'opentelemetry'
	K8SJobNameKey = attribute.Key("k8s.job.name")
)

// A Kubernetes CronJob object.
const (
	// The UID of the CronJob.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '275ecb36-5aa8-4c2a-9c47-d8bb681b9aff'
	K8SCronJobUIDKey = attribute.Key("k8s.cronjob.uid")
	// The name of the CronJob.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'opentelemetry'
	K8SCronJobNameKey = attribute.Key("k8s.cronjob.name")
)

// The operating system (OS) on which the process represented by this resource is running.
const (
	// The operating system type.
	//
	// Type: Enum
	// Required: Always
	// Stability: stable
	OSTypeKey = attribute.Key("os.type")
	// Human readable (not intended to be parsed) OS version information, like e.g.
	// reported by `ver` or `lsb_release -a` commands.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'Microsoft Windows [Version 10.0.18363.778]', 'Ubuntu 18.04.1 LTS'
	OSDescriptionKey = attribute.Key("os.description")
	// Human readable operating system name.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'iOS', 'Android', 'Ubuntu'
	OSNameKey = attribute.Key("os.name")
	// The version string of the operating system as defined in [Version
	// Attributes](../../resource/semantic_conventions/README.md#version-attributes).
	//
	// Type: string
	// Required: No
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
	// Oracle Solaris
	OSTypeSolaris = OSTypeKey.String("solaris")
	// IBM z/OS
	OSTypeZOS = OSTypeKey.String("z_os")
)

// An operating system process.
const (
	// Process identifier (PID).
	//
	// Type: int
	// Required: No
	// Stability: stable
	// Examples: 1234
	ProcessPIDKey = attribute.Key("process.pid")
	// The name of the process executable. On Linux based systems, can be set to the
	// `Name` in `proc/[pid]/status`. On Windows, can be set to the base name of
	// `GetProcessImageFileNameW`.
	//
	// Type: string
	// Required: See below
	// Stability: stable
	// Examples: 'otelcol'
	ProcessExecutableNameKey = attribute.Key("process.executable.name")
	// The full path to the process executable. On Linux based systems, can be set to
	// the target of `proc/[pid]/exe`. On Windows, can be set to the result of
	// `GetProcessImageFileNameW`.
	//
	// Type: string
	// Required: See below
	// Stability: stable
	// Examples: '/usr/bin/cmd/otelcol'
	ProcessExecutablePathKey = attribute.Key("process.executable.path")
	// The command used to launch the process (i.e. the command name). On Linux based
	// systems, can be set to the zeroth string in `proc/[pid]/cmdline`. On Windows,
	// can be set to the first parameter extracted from `GetCommandLineW`.
	//
	// Type: string
	// Required: See below
	// Stability: stable
	// Examples: 'cmd/otelcol'
	ProcessCommandKey = attribute.Key("process.command")
	// The full command used to launch the process as a single string representing the
	// full command. On Windows, can be set to the result of `GetCommandLineW`. Do not
	// set this if you have to assemble it just for monitoring; use
	// `process.command_args` instead.
	//
	// Type: string
	// Required: See below
	// Stability: stable
	// Examples: 'C:\\cmd\\otecol --config="my directory\\config.yaml"'
	ProcessCommandLineKey = attribute.Key("process.command_line")
	// All the command arguments (including the command/executable itself) as received
	// by the process. On Linux-based systems (and some other Unixoid systems
	// supporting procfs), can be set according to the list of null-delimited strings
	// extracted from `proc/[pid]/cmdline`. For libc-based executables, this would be
	// the full argv vector passed to `main`.
	//
	// Type: string[]
	// Required: See below
	// Stability: stable
	// Examples: 'cmd/otecol', '--config=config.yaml'
	ProcessCommandArgsKey = attribute.Key("process.command_args")
	// The username of the user that owns the process.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'root'
	ProcessOwnerKey = attribute.Key("process.owner")
)

// The single (language) runtime instance which is monitored.
const (
	// The name of the runtime of this process. For compiled native binaries, this
	// SHOULD be the name of the compiler.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'OpenJDK Runtime Environment'
	ProcessRuntimeNameKey = attribute.Key("process.runtime.name")
	// The version of the runtime of this process, as returned by the runtime without
	// modification.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '14.0.2'
	ProcessRuntimeVersionKey = attribute.Key("process.runtime.version")
	// An additional description about the runtime of the process, for example a
	// specific vendor customization of the runtime environment.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'Eclipse OpenJ9 Eclipse OpenJ9 VM openj9-0.21.0'
	ProcessRuntimeDescriptionKey = attribute.Key("process.runtime.description")
)

// A service instance.
const (
	// Logical name of the service.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'shoppingcart'
	// Note: MUST be the same for all instances of horizontally scaled services. If
	// the value was not specified, SDKs MUST fallback to `unknown_service:`
	// concatenated with [`process.executable.name`](process.md#process), e.g.
	// `unknown_service:bash`. If `process.executable.name` is not available, the
	// value MUST be set to `unknown_service`.
	ServiceNameKey = attribute.Key("service.name")
	// A namespace for `service.name`.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'Shop'
	// Note: A string value having a meaning that helps to distinguish a group of
	// services, for example the team name that owns a group of services.
	// `service.name` is expected to be unique within the same namespace. If
	// `service.namespace` is not specified in the Resource then `service.name` is
	// expected to be unique for all services that have no explicit namespace defined
	// (so the empty/unspecified namespace is simply one more valid namespace). Zero-
	// length namespace string is assumed equal to unspecified namespace.
	ServiceNamespaceKey = attribute.Key("service.namespace")
	// The string ID of the service instance.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '627cc493-f310-47de-96bd-71410b7dec09'
	// Note: MUST be unique for each instance of the same
	// `service.namespace,service.name` pair (in other words
	// `service.namespace,service.name,service.instance.id` triplet MUST be globally
	// unique). The ID helps to distinguish instances of the same service that exist
	// at the same time (e.g. instances of a horizontally scaled service). It is
	// preferable for the ID to be persistent and stay the same for the lifetime of
	// the service instance, however it is acceptable that the ID is ephemeral and
	// changes during important lifetime events for the service (e.g. service
	// restarts). If the service has no inherent unique ID that can be used as the
	// value of this attribute it is recommended to generate a random Version 1 or
	// Version 4 RFC 4122 UUID (services aiming for reproducible UUIDs may also use
	// Version 5, see RFC 4122 for more recommendations).
	ServiceInstanceIDKey = attribute.Key("service.instance.id")
	// The version string of the service API or implementation.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '2.0.0'
	ServiceVersionKey = attribute.Key("service.version")
)

// The telemetry SDK used to capture data recorded by the instrumentation libraries.
const (
	// The name of the telemetry SDK as defined above.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'opentelemetry'
	TelemetrySDKNameKey = attribute.Key("telemetry.sdk.name")
	// The language of the telemetry SDK.
	//
	// Type: Enum
	// Required: No
	// Stability: stable
	TelemetrySDKLanguageKey = attribute.Key("telemetry.sdk.language")
	// The version string of the telemetry SDK.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '1.2.3'
	TelemetrySDKVersionKey = attribute.Key("telemetry.sdk.version")
	// The version string of the auto instrumentation agent, if used.
	//
	// Type: string
	// Required: No
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
)

// Resource describing the packaged software running the application code. Web engines are typically executed using process.runtime.
const (
	// The name of the web engine.
	//
	// Type: string
	// Required: Always
	// Stability: stable
	// Examples: 'WildFly'
	WebEngineNameKey = attribute.Key("webengine.name")
	// The version of the web engine.
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: '21.0.0'
	WebEngineVersionKey = attribute.Key("webengine.version")
	// Additional description of the web engine (e.g. detailed version and edition
	// information).
	//
	// Type: string
	// Required: No
	// Stability: stable
	// Examples: 'WildFly Full 21.0.0.Final (WildFly Core 13.0.1.Final) - 2.2.2.Final'
	WebEngineDescriptionKey = attribute.Key("webengine.description")
)
