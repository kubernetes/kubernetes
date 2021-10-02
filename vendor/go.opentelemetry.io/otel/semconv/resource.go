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

package semconv // import "go.opentelemetry.io/otel/semconv"

import "go.opentelemetry.io/otel/attribute"

// Semantic conventions for service resource attribute keys.
const (
	// Name of the service.
	ServiceNameKey = attribute.Key("service.name")

	// A namespace for `service.name`. This needs to have meaning that helps
	// to distinguish a group of services. For example, the team name that
	// owns a group of services. `service.name` is expected to be unique
	// within the same namespace.
	ServiceNamespaceKey = attribute.Key("service.namespace")

	// A unique identifier of the service instance. In conjunction with the
	// `service.name` and `service.namespace` this must be unique.
	ServiceInstanceIDKey = attribute.Key("service.instance.id")

	// The version of the service API.
	ServiceVersionKey = attribute.Key("service.version")
)

// Semantic conventions for telemetry SDK resource attribute keys.
const (
	// The name of the telemetry SDK.
	//
	// The default OpenTelemetry SDK provided by the OpenTelemetry project
	// MUST set telemetry.sdk.name to the value `opentelemetry`.
	//
	// If another SDK is used, this attribute MUST be set to the import path
	// of that SDK's package.
	//
	// The value `opentelemetry` is reserved and MUST NOT be used by
	// non-OpenTelemetry SDKs.
	TelemetrySDKNameKey = attribute.Key("telemetry.sdk.name")

	// The language of the telemetry SDK.
	TelemetrySDKLanguageKey = attribute.Key("telemetry.sdk.language")

	// The version string of the telemetry SDK.
	TelemetrySDKVersionKey = attribute.Key("telemetry.sdk.version")
)

// Semantic conventions for telemetry SDK resource attributes.
var (
	TelemetrySDKLanguageGo = TelemetrySDKLanguageKey.String("go")
)

// Semantic conventions for container resource attribute keys.
const (
	// A uniquely identifying name for the Container.
	ContainerNameKey = attribute.Key("container.name")

	// Container ID, usually a UUID, as for example used to
	// identify Docker containers. The UUID might be abbreviated.
	ContainerIDKey = attribute.Key("container.id")

	// Name of the image the container was built on.
	ContainerImageNameKey = attribute.Key("container.image.name")

	// Container image tag.
	ContainerImageTagKey = attribute.Key("container.image.tag")
)

// Semantic conventions for Function-as-a-Service resource attribute keys.
const (
	// A uniquely identifying name for the FaaS.
	FaaSNameKey = attribute.Key("faas.name")

	// The unique name of the function being executed.
	FaaSIDKey = attribute.Key("faas.id")

	// The version of the function being executed.
	FaaSVersionKey = attribute.Key("faas.version")

	// The execution environment identifier.
	FaaSInstanceKey = attribute.Key("faas.instance")
)

// Semantic conventions for operating system process resource attribute keys.
const (
	// Process identifier (PID).
	ProcessPIDKey = attribute.Key("process.pid")
	// The name of the process executable. On Linux based systems, can be
	// set to the `Name` in `proc/[pid]/status`. On Windows, can be set to
	// the base name of `GetProcessImageFileNameW`.
	ProcessExecutableNameKey = attribute.Key("process.executable.name")
	// The full path to the process executable. On Linux based systems, can
	// be set to the target of `proc/[pid]/exe`. On Windows, can be set to
	// the result of `GetProcessImageFileNameW`.
	ProcessExecutablePathKey = attribute.Key("process.executable.path")
	// The command used to launch the process (i.e. the command name). On
	// Linux based systems, can be set to the zeroth string in
	// `proc/[pid]/cmdline`. On Windows, can be set to the first parameter
	// extracted from `GetCommandLineW`.
	ProcessCommandKey = attribute.Key("process.command")
	// The full command used to launch the process. The value can be either
	// a list of strings representing the ordered list of arguments, or a
	// single string representing the full command. On Linux based systems,
	// can be set to the list of null-delimited strings extracted from
	// `proc/[pid]/cmdline`. On Windows, can be set to the result of
	// `GetCommandLineW`.
	ProcessCommandLineKey = attribute.Key("process.command_line")
	// All the command arguments (including the command/executable itself)
	// as received by the process. On Linux-based systems (and some other
	// Unixoid systems supporting procfs), can be set according to the list
	// of null-delimited strings extracted from `proc/[pid]/cmdline`. For
	// libc-based executables, this would be the full argv vector passed to
	// `main`.
	ProcessCommandArgsKey = attribute.Key("process.command_args")
	// The username of the user that owns the process.
	ProcessOwnerKey = attribute.Key("process.owner")
	// The name of the runtime of this process. For compiled native
	// binaries, this SHOULD be the name of the compiler.
	ProcessRuntimeNameKey = attribute.Key("process.runtime.name")
	// The version of the runtime of this process, as returned by the
	// runtime without modification.
	ProcessRuntimeVersionKey = attribute.Key("process.runtime.version")
	// An additional description about the runtime of the process, for
	// example a specific vendor customization of the runtime environment.
	ProcessRuntimeDescriptionKey = attribute.Key("process.runtime.description")
)

// Semantic conventions for Kubernetes resource attribute keys.
const (
	// A uniquely identifying name for the Kubernetes cluster. Kubernetes
	// does not have cluster names as an internal concept so this may be
	// set to any meaningful value within the environment. For example,
	// GKE clusters have a name which can be used for this attribute.
	K8SClusterNameKey = attribute.Key("k8s.cluster.name")

	// The name of the Node.
	K8SNodeNameKey = attribute.Key("k8s.node.name")

	// The UID of the Node.
	K8SNodeUIDKey = attribute.Key("k8s.node.uid")

	// The name of the namespace that the pod is running in.
	K8SNamespaceNameKey = attribute.Key("k8s.namespace.name")

	// The uid of the Pod.
	K8SPodUIDKey = attribute.Key("k8s.pod.uid")

	// The name of the pod.
	K8SPodNameKey = attribute.Key("k8s.pod.name")

	// The name of the Container in a Pod template.
	K8SContainerNameKey = attribute.Key("k8s.container.name")

	// The uid of the ReplicaSet.
	K8SReplicaSetUIDKey = attribute.Key("k8s.replicaset.uid")

	// The name of the ReplicaSet.
	K8SReplicaSetNameKey = attribute.Key("k8s.replicaset.name")

	// The uid of the Deployment.
	K8SDeploymentUIDKey = attribute.Key("k8s.deployment.uid")

	// The name of the deployment.
	K8SDeploymentNameKey = attribute.Key("k8s.deployment.name")

	// The uid of the StatefulSet.
	K8SStatefulSetUIDKey = attribute.Key("k8s.statefulset.uid")

	// The name of the StatefulSet.
	K8SStatefulSetNameKey = attribute.Key("k8s.statefulset.name")

	// The uid of the DaemonSet.
	K8SDaemonSetUIDKey = attribute.Key("k8s.daemonset.uid")

	// The name of the DaemonSet.
	K8SDaemonSetNameKey = attribute.Key("k8s.daemonset.name")

	// The uid of the Job.
	K8SJobUIDKey = attribute.Key("k8s.job.uid")

	// The name of the Job.
	K8SJobNameKey = attribute.Key("k8s.job.name")

	// The uid of the CronJob.
	K8SCronJobUIDKey = attribute.Key("k8s.cronjob.uid")

	// The name of the CronJob.
	K8SCronJobNameKey = attribute.Key("k8s.cronjob.name")
)

// Semantic conventions for OS resource attribute keys.
const (
	// The operating system type.
	OSTypeKey = attribute.Key("os.type")
	// Human readable (not intended to be parsed) OS version information.
	OSDescriptionKey = attribute.Key("os.description")
)

// Semantic conventions for host resource attribute keys.
const (
	// A uniquely identifying name for the host: 'hostname', FQDN, or user specified name
	HostNameKey = attribute.Key("host.name")

	// Unique host ID. For cloud environments this will be the instance ID.
	HostIDKey = attribute.Key("host.id")

	// Type of host. For cloud environments this will be the machine type.
	HostTypeKey = attribute.Key("host.type")

	// Name of the OS or VM image the host is running.
	HostImageNameKey = attribute.Key("host.image.name")

	// Identifier of the image the host is running.
	HostImageIDKey = attribute.Key("host.image.id")

	// Version of the image the host is running.
	HostImageVersionKey = attribute.Key("host.image.version")
)

// Semantic conventions for cloud environment resource attribute keys.
const (
	// Name of the cloud provider.
	CloudProviderKey = attribute.Key("cloud.provider")

	// The account ID from the cloud provider used for authorization.
	CloudAccountIDKey = attribute.Key("cloud.account.id")

	// Geographical region where this resource is.
	CloudRegionKey = attribute.Key("cloud.region")

	// Zone of the region where this resource is.
	CloudZoneKey = attribute.Key("cloud.zone")
)

// Semantic conventions for common cloud provider resource attributes.
var (
	CloudProviderAWS   = CloudProviderKey.String("aws")
	CloudProviderAzure = CloudProviderKey.String("azure")
	CloudProviderGCP   = CloudProviderKey.String("gcp")
)

// Semantic conventions for deployment attributes.
const (
	// Name of the deployment environment (aka deployment tier); e.g. (staging, production).
	DeploymentEnvironmentKey = attribute.Key("deployment.environment")
)
