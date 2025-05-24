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

package pod

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// EventType defines the type of pod state change event
type EventType string

const (
	PodAdded      EventType = "PodAdded"
	PodUpdated    EventType = "PodUpdated"
	PodRemoved    EventType = "PodRemoved"
	PodTerminated EventType = "PodTerminated"
)

// PodStateEvent represents a state change event for a pod
type PodStateEvent struct {
	Type EventType
	Pod  *v1.Pod
	// UID is the unique identifier for the pod
	UID types.UID
}

// PodStateChannel is a channel used to communicate pod state changes
type PodStateChannel chan PodStateEvent
