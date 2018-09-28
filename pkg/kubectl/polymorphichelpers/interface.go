/*
Copyright 2018 The Kubernetes Authors.

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

package polymorphichelpers

import (
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/kubectl"
)

// LogsForObjectFunc is a function type that can tell you how to get logs for a runtime.object
type LogsForObjectFunc func(restClientGetter genericclioptions.RESTClientGetter, object, options runtime.Object, timeout time.Duration, allContainers bool) ([]*rest.Request, error)

// LogsForObjectFn gives a way to easily override the function for unit testing if needed.
var LogsForObjectFn LogsForObjectFunc = logsForObject

// AttachableLogsForObjectFunc is a function type that can tell you how to get the pod for which to attach a given object
type AttachableLogsForObjectFunc func(restClientGetter genericclioptions.RESTClientGetter, object runtime.Object, timeout time.Duration) (*v1.Pod, error)

// AttachablePodForObjectFn gives a way to easily override the function for unit testing if needed.
var AttachablePodForObjectFn AttachableLogsForObjectFunc = attachablePodForObject

// HistoryViewerFunc is a function type that can tell you how to view change history
type HistoryViewerFunc func(restClientGetter genericclioptions.RESTClientGetter, mapping *meta.RESTMapping) (kubectl.HistoryViewer, error)

// HistoryViewerFn gives a way to easily override the function for unit testing if needed
var HistoryViewerFn HistoryViewerFunc = historyViewer

// StatusViewerFunc is a function type that can tell you how to print rollout status
type StatusViewerFunc func(restClientGetter genericclioptions.RESTClientGetter, mapping *meta.RESTMapping) (kubectl.StatusViewer, error)

// StatusViewerFn gives a way to easily override the function for unit testing if needed
var StatusViewerFn StatusViewerFunc = statusViewer

// UpdatePodSpecForObjectFunc will call the provided function on the pod spec this object supports,
// return false if no pod spec is supported, or return an error.
type UpdatePodSpecForObjectFunc func(obj runtime.Object, fn func(*v1.PodSpec) error) (bool, error)

// UpdatePodSpecForObjectFn gives a way to easily override the function for unit testing if needed
var UpdatePodSpecForObjectFn UpdatePodSpecForObjectFunc = updatePodSpecForObject

// MapBasedSelectorForObjectFunc will call the provided function on mapping the baesd selector for object,
// return "" if object is not supported, or return an error.
type MapBasedSelectorForObjectFunc func(object runtime.Object) (string, error)

// MapBasedSelectorForObjectFn gives a way to easily override the function for unit testing if needed
var MapBasedSelectorForObjectFn MapBasedSelectorForObjectFunc = mapBasedSelectorForObject

// ProtocolsForObjectFunc will call the provided function on the protocols for the object,
// return nil-map if no protocols for the object, or return an error.
type ProtocolsForObjectFunc func(object runtime.Object) (map[string]string, error)

// ProtocolsForObjectFn gives a way to easily override the function for unit testing if needed
var ProtocolsForObjectFn ProtocolsForObjectFunc = protocolsForObject

// PortsForObjectFunc returns the ports associated with the provided object
type PortsForObjectFunc func(object runtime.Object) ([]string, error)

// PortsForObjectFn gives a way to easily override the function for unit testing if needed
var PortsForObjectFn PortsForObjectFunc = portsForObject

// CanBeAutoscaledFunc checks whether the kind of resources could be autoscaled
type CanBeAutoscaledFunc func(kind schema.GroupKind) error

// CanBeAutoscaledFn gives a way to easily override the function for unit testing if needed
var CanBeAutoscaledFn CanBeAutoscaledFunc = canBeAutoscaled

// CanBeExposedFunc is a function type that can tell you whether a given GroupKind is capable of being exposed
type CanBeExposedFunc func(kind schema.GroupKind) error

// CanBeExposedFn gives a way to easily override the function for unit testing if needed
var CanBeExposedFn CanBeExposedFunc = canBeExposed

// ObjectPauserFunc is a function type that marks the object in a given info as paused.
type ObjectPauserFunc func(runtime.Object) ([]byte, error)

// ObjectPauserFn gives a way to easily override the function for unit testing if needed.
// Returns the patched object in bytes and any error that occurred during the encoding or
// in case the object is already paused.
var ObjectPauserFn ObjectPauserFunc = defaultObjectPauser

// ObjectResumerFunc is a function type that marks the object in a given info as resumed.
type ObjectResumerFunc func(runtime.Object) ([]byte, error)

// ObjectResumerFn gives a way to easily override the function for unit testing if needed.
// Returns the patched object in bytes and any error that occurred during the encoding or
// in case the object is already resumed.
var ObjectResumerFn ObjectResumerFunc = defaultObjectResumer

// RollbackerFunc gives a way to change the rollback version of the specified RESTMapping type
type RollbackerFunc func(restClientGetter genericclioptions.RESTClientGetter, mapping *meta.RESTMapping) (kubectl.Rollbacker, error)

// RollbackerFn gives a way to easily override the function for unit testing if needed
var RollbackerFn RollbackerFunc = rollbacker
