/*
Copyright 2016 The Kubernetes Authors.

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

package util

import (
	"reflect"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	extensions_v1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	deputils "k8s.io/kubernetes/pkg/controller/deployment/util"
)

// Checks if cluster-independent, user provided data in two given Deployment are equal.
// This function assumes that revisions are not kept in sync across the clusters.
func DeploymentEquivalent(a, b *extensions_v1.Deployment) bool {
	if a.Name != b.Name {
		return false
	}
	if a.Namespace != b.Namespace {
		return false
	}
	if !reflect.DeepEqual(a.Labels, b.Labels) && (len(a.Labels) != 0 || len(b.Labels) != 0) {
		return false
	}
	hasKeysAndVals := func(x, y map[string]string) bool {
		if x == nil {
			x = map[string]string{}
		}
		if y == nil {
			y = map[string]string{}
		}
		for k, v := range x {
			if k == deputils.RevisionAnnotation {
				continue
			}
			v2, found := y[k]
			if !found || v != v2 {
				return false
			}
		}
		return true
	}
	return hasKeysAndVals(a.Annotations, b.Annotations) &&
		hasKeysAndVals(b.Annotations, a.Annotations) &&
		reflect.DeepEqual(a.Spec, b.Spec)
}

// Copies object meta for Deployment, skipping revision information.
func DeepCopyDeploymentObjectMeta(meta metav1.ObjectMeta) metav1.ObjectMeta {
	meta = DeepCopyRelevantObjectMeta(meta)
	delete(meta.Annotations, deputils.RevisionAnnotation)
	return meta
}

// Copies object meta for Deployment, skipping revision information.
func DeepCopyDeployment(a *extensions_v1.Deployment) *extensions_v1.Deployment {
	return &extensions_v1.Deployment{
		ObjectMeta: DeepCopyDeploymentObjectMeta(a.ObjectMeta),
		Spec:       *(DeepCopyApiTypeOrPanic(&a.Spec).(*extensions_v1.DeploymentSpec)),
	}
}
