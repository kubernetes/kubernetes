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

package resourcequota

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

// ResourceSet is an alias for ease of use that returns true
type ResourceSet map[api.ResourceName]struct{}

// UsageOptions is used as input for how to measure usage of a set of resources
type UsageOptions struct {
	// Namespace to measure
	Namespace string
	// Resources to measure
	Resources ResourceSet
	// A field selector that must match resources
	FieldSelector fields.Selector
	// A label selector that must match resources
	LabelSelector labels.Selector
}

// Usage represents the observed entities tracked by quota
// TODO: add []ObjectReference that tracks identifiers of tracked resources
type Usage struct {
	// Used maps resource to quantity used
	Used api.ResourceList
}

// UsageFunc measures usage of a quota tracked resource
type UsageFunc func(options UsageOptions) (Usage, error)

// UsageFuncRegistry manages a set of functions that can calculate usage
type UsageFuncRegistry interface {
	// Map of internal group kind to a function that knows how to measure usage
	UsageFuncs() map[unversioned.GroupKind]UsageFunc
}

// MonitoringControllerOptions is an options struct passed when instantiating
// a controller that monitors resources tracked by quota that need to be
// replenished faster than the default full resync interval.
type MonitoringControllerOptions struct {
	// The resource that should be monitored
	GroupKind unversioned.GroupKind
	// how often the monitoring controller does a full resync
	ResyncPeriod controller.ResyncPeriodFunc
	// ResourceEventHandlerFuncs that should be injected into the monitoring
	// controller that enables interfacing with the quota controller
	ResourceEventHandlerFuncs framework.ResourceEventHandlerFuncs
}

// MonitoringControllerFactory knows how to build monitoring controllers
type MonitoringControllerFactory interface {
	// NewController returns a controller configured with specified options
	NewController(options *MonitoringControllerOptions) (*framework.Controller, error)
}
