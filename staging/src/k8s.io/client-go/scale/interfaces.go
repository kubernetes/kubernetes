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
	"k8s.io/apimachinery/pkg/runtime/schema"
	scaleapi "k8s.io/client-go/pkg/apis/autoscaling/v1"
)

// ScalesGetter has a method to get a ScaleInterface to
// fetch scales for a given namespace.
type ScalesGetter interface {
	Scales(namespace string) ScaleInterface
}

// ScaleInteface supports fetch scales for resources which
// implement the scale subresource.
type ScaleInterface interface {
	// Get fetches the scale for the given scalable resource.
	Get(kind schema.GroupKind, name string) (*scaleapi.Scale, error)

	// Update scales the given scalable resource.
	Update(kind schema.GroupKind, scale *scaleapi.Scale) (*scaleapi.Scale, error)
}
