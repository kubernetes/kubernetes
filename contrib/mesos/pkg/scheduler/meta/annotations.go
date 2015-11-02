/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	// the BindingHostKey pod annotation marks a pod as being assigned to a Mesos
	// slave. It is already or will be launched on the slave as a task.
	BindingHostKey = "k8s.mesosphere.io/bindingHost"

	TaskIdKey                = "k8s.mesosphere.io/taskId"
	SlaveIdKey               = "k8s.mesosphere.io/slaveId"
	OfferIdKey               = "k8s.mesosphere.io/offerId"
	PortMappingKeyPrefix     = "k8s.mesosphere.io/port_"
	PortMappingKeyFormat     = PortMappingKeyPrefix + "%s_%d"
	PortNameMappingKeyPrefix = "k8s.mesosphere.io/portName_"
	PortNameMappingKeyFormat = PortNameMappingKeyPrefix + "%s_%s"
	ContainerPortKeyFormat   = "k8s.mesosphere.io/containerPort_%s_%s_%d"
)
