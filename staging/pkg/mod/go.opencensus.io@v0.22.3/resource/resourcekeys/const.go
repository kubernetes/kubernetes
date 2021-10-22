// Copyright 2019, OpenCensus Authors
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

// Package resourcekeys contains well known type and label keys for resources.
package resourcekeys // import "go.opencensus.io/resource/resourcekeys"

// Constants for Kubernetes resources.
const (
	K8SType = "k8s"

	// A uniquely identifying name for the Kubernetes cluster. Kubernetes
	// does not have cluster names as an internal concept so this may be
	// set to any meaningful value within the environment. For example,
	// GKE clusters have a name which can be used for this label.
	K8SKeyClusterName    = "k8s.cluster.name"
	K8SKeyNamespaceName  = "k8s.namespace.name"
	K8SKeyPodName        = "k8s.pod.name"
	K8SKeyDeploymentName = "k8s.deployment.name"
)

// Constants for Container resources.
const (
	ContainerType = "container"

	// A uniquely identifying name for the Container.
	ContainerKeyName      = "container.name"
	ContainerKeyImageName = "container.image.name"
	ContainerKeyImageTag  = "container.image.tag"
)

// Constants for Cloud resources.
const (
	CloudType = "cloud"

	CloudKeyProvider  = "cloud.provider"
	CloudKeyAccountID = "cloud.account.id"
	CloudKeyRegion    = "cloud.region"
	CloudKeyZone      = "cloud.zone"

	// Cloud Providers
	CloudProviderAWS   = "aws"
	CloudProviderGCP   = "gcp"
	CloudProviderAZURE = "azure"
)

// Constants for Host resources.
const (
	HostType = "host"

	// A uniquely identifying name for the host.
	HostKeyName = "host.name"

	// A hostname as returned by the 'hostname' command on host machine.
	HostKeyHostName = "host.hostname"
	HostKeyID       = "host.id"
	HostKeyType     = "host.type"
)
