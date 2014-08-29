/*
Copyright 2014 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// Registry is an interface implemented by things that know how to store Pod objects.
type Registry interface {
	// ListPods obtains a list of pods that match selector.
	ListPods(selector labels.Selector) (*api.PodList, error)
	// Watch for new/changed/deleted pods
	WatchPods(resourceVersion uint64, filter func(*api.Pod) bool) (watch.Interface, error)
	// Get a specific pod
	GetPod(podID string) (*api.Pod, error)
	// Create a pod based on a specification.
	CreatePod(pod api.Pod) error
	// Update an existing pod
	UpdatePod(pod api.Pod) error
	// Delete an existing pod
	DeletePod(podID string) error
}
