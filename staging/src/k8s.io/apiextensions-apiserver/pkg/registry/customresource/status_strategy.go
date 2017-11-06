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

package customresource

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

type statusStrategy struct {
	customResourceStrategy
}

func NewStatusStrategy(strategy customResourceStrategy) statusStrategy {
	return statusStrategy{strategy}
}

func (a statusStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newCustomResourceObject := obj.(*unstructured.Unstructured)
	oldCustomResourceObject := old.(*unstructured.Unstructured)

	newCustomResource := newCustomResourceObject.UnstructuredContent()
	oldCustomResource := oldCustomResourceObject.UnstructuredContent()

	// update is not allowed to set spec and metadata
	_, ok1 := newCustomResource["spec"]
	_, ok2 := oldCustomResource["spec"]
	switch {
	case ok2:
		newCustomResource["spec"] = oldCustomResource["spec"]
	case ok1:
		delete(newCustomResource, "spec")
	}

	newCustomResourceObject.SetAnnotations(oldCustomResourceObject.GetAnnotations())
	newCustomResourceObject.SetFinalizers(oldCustomResourceObject.GetFinalizers())
	newCustomResourceObject.SetGeneration(oldCustomResourceObject.GetGeneration())
	newCustomResourceObject.SetLabels(oldCustomResourceObject.GetLabels())
	newCustomResourceObject.SetOwnerReferences(oldCustomResourceObject.GetOwnerReferences())
	newCustomResourceObject.SetSelfLink(oldCustomResourceObject.GetSelfLink())
}

// ValidateUpdate is the default update validation for an end user updating status.
func (a statusStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	return a.customResourceStrategy.validator.ValidateStatusUpdate(ctx, obj, old, a.scale)
}
