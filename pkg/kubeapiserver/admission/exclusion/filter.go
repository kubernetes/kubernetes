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

package exclusion

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/resourcefilter"
)

// NewFilter creates a resource filter with the built-in exclusion list.
func NewFilter() resourcefilter.Interface {
	return &filter{excluded: sets.New[schema.GroupResource](excluded...)}
}

type filter struct {
	excluded sets.Set[schema.GroupResource]
}

func (f *filter) ShouldHandle(a admission.Attributes) bool {
	gvr := a.GetResource()
	// ignore the version for the decision-making
	// because putting different versions into different category
	// is almost always a mistake.
	gr := gvr.GroupResource()
	return !f.excluded.Has(gr)
}
