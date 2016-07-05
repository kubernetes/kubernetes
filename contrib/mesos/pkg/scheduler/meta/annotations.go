/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package meta

// kubernetes api object annotations
const (
	// Namespace is the label and annotation namespace for mesos keys
	Namespace = "k8s.mesosphere.io"

	// the BindingHostKey pod annotation marks a pod as being assigned to a Mesos
	// slave. It is already or will be launched on the slave as a task.
	BindingHostKey = Namespace + "/bindingHost"

	TaskIdKey                = Namespace + "/taskId"
	SlaveIdKey               = Namespace + "/slaveId"
	OfferIdKey               = Namespace + "/offerId"
	ExecutorIdKey            = Namespace + "/executorId"
	ExecutorResourcesKey     = Namespace + "/executorResources"
	PortMappingKey           = Namespace + "/portMapping"
	PortMappingKeyPrefix     = Namespace + "/port_"
	PortMappingKeyFormat     = PortMappingKeyPrefix + "%s_%d"
	PortNameMappingKeyPrefix = Namespace + "/portName_"
	PortNameMappingKeyFormat = PortNameMappingKeyPrefix + "%s_%s"
	ContainerPortKeyFormat   = Namespace + "/containerPort_%s_%s_%d"
	StaticPodFilenameKey     = Namespace + "/staticPodFilename"
	RolesKey                 = Namespace + "/roles"
)
