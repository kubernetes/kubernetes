/*
Copyright 2024 The Kubernetes Authors.

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

package patch

import (
	"context"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
)

// Patcher provides a patch function to perform a mutation to an object in the admission chain.
type Patcher interface {
	// Patch returns a copy of the object in the request, modified to change specified by the patch.
	// The original object in the request MUST NOT be modified in-place.
	Patch(ctx context.Context, request Request, runtimeCELCostBudget int64) (runtime.Object, error)
}

// Request defines the arguments required by a patcher.
type Request struct {
	MatchedResource     schema.GroupVersionResource
	VersionedAttributes *admission.VersionedAttributes
	ObjectInterfaces    admission.ObjectInterfaces
	OptionalVariables   cel.OptionalVariableBindings
	Namespace           *v1.Namespace
	TypeConverter       managedfields.TypeConverter
}
