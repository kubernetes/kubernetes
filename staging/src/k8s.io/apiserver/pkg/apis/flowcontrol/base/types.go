/*
Copyright 2025 The Kubernetes Authors.

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

package base

import (
	flowcontrolv1 "k8s.io/api/flowcontrol/v1"
)

// V1ConfigCollection is the collection of default V1 configuration objects to use.
// Deeply immutable.
type V1ConfigCollection struct {
	// Mandatory holds the objects that define an apiserver's initial behavior.  The
	// registered defaulting procedures make no changes to these
	// particular objects (this is verified in the unit tests of the
	// internalbootstrap package; it can not be verified in this package
	// because that would require importing k8s.io/kubernetes).
	Mandatory V1ConfigSlices
	// Suggested holds the objects that define the current suggested additional configuration.
	Suggested V1ConfigSlices
	// PriorityLevelConfigurationExempt also appears in `Mandatory.PriorityLevelConfigurations`.
	PriorityLevelConfigurationExempt *flowcontrolv1.PriorityLevelConfiguration
	// PriorityLevelConfigurationCatchAll also appears in `Mandatory.PriorityLevelConfigurations`.
	PriorityLevelConfigurationCatchAll *flowcontrolv1.PriorityLevelConfiguration
	// FlowSchemaExempt also appears in `Mandatory.FlowsSchemas`.
	FlowSchemaExempt *flowcontrolv1.FlowSchema
	// FlowSchemaCatchAll also appears in `Mandatory.FlowSchemas`.
	FlowSchemaCatchAll *flowcontrolv1.FlowSchema
}

// V1ConfigSlices is a collection of v1 APF configuration objects.
// Deeply immutable.
type V1ConfigSlices struct {
	PriorityLevelConfigurations []*flowcontrolv1.PriorityLevelConfiguration
	FlowSchemas                 []*flowcontrolv1.FlowSchema
}
