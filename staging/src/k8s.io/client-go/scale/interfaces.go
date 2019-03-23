/*
Copyright 2017 The Kubernetes Authors.

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

package scale

import (
	autoscalingapi "k8s.io/api/autoscaling/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// ScalesGetter can produce a ScaleInterface
// for a particular namespace.
type ScalesGetter interface {
	Scales(namespace string) ScaleInterface
}

// ScaleInterface can fetch and update scales for
// resources in a particular namespace which implement
// the scale subresource.
type ScaleInterface interface {
	// Get fetches the scale of the given scalable resource.
	Get(resource schema.GroupResource, name string) (*autoscalingapi.Scale, error)

	// Update updates the scale of the given scalable resource.
	Update(resource schema.GroupResource, scale *autoscalingapi.Scale) (*autoscalingapi.Scale, error)
}
