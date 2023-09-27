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

package handlers

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/endpoints/request"
)

const KCPOriginalAPIVersionAnnotation = "kcp.io/original-api-version"

// setKCPOriginalAPIVersionAnnotation sets the annotation kcp.io/original-api-version on partial indicating the actual
// API version of the original object. This is necessary for kcp with wildcard partial metadata list/watch requests.
// For example, if the request is for /clusters/*/apis/kcp.io/v1/widgets, and it's a partial metadata request, the
// server returns ALL widgets, regardless of their API version. But because this is a partial metadata request, the
// API version of the returned object is always meta.k8s.io/$version (could be v1 or v1beta1). Any client needing to
// modify or delete the returned object must know its exact API version. Therefore, we set this annotation with the
// actual original API version of the object. Clients can use it when constructing dynamic clients to guarantee they
// are using the correct API version.
func setKCPOriginalAPIVersionAnnotation(ctx context.Context, original any, partial *metav1.PartialObjectMetadata) {
	if cluster := request.ClusterFrom(ctx); !cluster.Wildcard {
		return
	}

	annotations := partial.GetAnnotations()

	if annotations[KCPOriginalAPIVersionAnnotation] != "" {
		// Do not overwrite the annotation if it is present. It is set by the kcpWildcardPartialMetadataConverter
		// during the conversion process so we don't lose the original API version. Changing it here would lead to
		// an incorrect value.
		return
	}

	if annotations == nil {
		annotations = make(map[string]string)
	}

	t, err := meta.TypeAccessor(original)
	if err != nil {
		panic(fmt.Errorf("unable to get a TypeAccessor for %T: %w", original, err))
	}

	annotations[KCPOriginalAPIVersionAnnotation] = t.GetAPIVersion()
	partial.SetAnnotations(annotations)
}
