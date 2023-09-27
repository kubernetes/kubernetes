/*
Copyright 2023 The KCP Authors.

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

package conversion

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers"
)

type kcpWildcardPartialMetadataConverter struct {
}

func NewKCPWildcardPartialMetadataConverter() *kcpWildcardPartialMetadataConverter {
	return &kcpWildcardPartialMetadataConverter{}
}

var _ CRConverter = &kcpWildcardPartialMetadataConverter{}

// Convert is a NOP converter that additionally stores the original APIVersion of each item in the annotation
// kcp.io/original-api-version. This is necessary for kcp with wildcard partial metadata list/watch requests.
// For example, if the request is for /clusters/*/apis/kcp.io/v1/widgets, and it's a partial metadata request, the
// server returns ALL widgets, regardless of their API version. But because this is a partial metadata request, the
// API version of the returned object is always meta.k8s.io/$version (could be v1 or v1beta1). Any client needing to
// modify or delete the returned object must know its exact API version. Therefore, we set this annotation with the
// actual original API version of the object. Clients can use it when constructing dynamic clients to guarantee they
// // are using the correct API version.
func (c *kcpWildcardPartialMetadataConverter) Convert(list *unstructured.UnstructuredList, targetGV schema.GroupVersion) (*unstructured.UnstructuredList, error) {
	for i := range list.Items {
		item := &list.Items[i]

		// First preserve the actual API version
		annotations := item.GetAnnotations()
		if annotations == nil {
			annotations = make(map[string]string)
		}
		annotations[handlers.KCPOriginalAPIVersionAnnotation] = item.GetAPIVersion()
		item.SetAnnotations(annotations)

		// Now that we've preserved it, we can change it to the targetGV.
		item.SetGroupVersionKind(targetGV.WithKind(item.GroupVersionKind().Kind))
	}
	return list, nil
}
