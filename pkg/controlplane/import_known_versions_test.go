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

package controlplane

import (
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	apinamingtest "k8s.io/apimachinery/pkg/api/apitesting/naming"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

func TestGroupVersions(t *testing.T) {
	// legacyUnsuffixedGroups contains the groups released prior to deciding that kubernetes API groups should be dns-suffixed
	// new groups should be suffixed with ".k8s.io" (https://github.com/kubernetes/kubernetes/pull/31887#issuecomment-244462396)
	legacyUnsuffixedGroups := sets.NewString(
		"",
		"apps",
		"autoscaling",
		"batch",
		"extensions",
		"policy",
	)

	// No new groups should be added to the legacyUnsuffixedGroups exclusion list
	if len(legacyUnsuffixedGroups) != 6 {
		t.Errorf("No additional unnamespaced groups should be created")
	}

	if err := apinamingtest.VerifyGroupNames(legacyscheme.Scheme, legacyUnsuffixedGroups); err != nil {
		t.Errorf("%v", err)
	}
}

// These types are registered in external versions, and therefore include json tags,
// but are also registered in internal versions (or referenced from internal types),
// so we explicitly allow tags for them
var typesAllowedTags = map[reflect.Type]bool{
	reflect.TypeOf(intstr.IntOrString{}):              true,
	reflect.TypeOf(metav1.Time{}):                     true,
	reflect.TypeOf(metav1.MicroTime{}):                true,
	reflect.TypeOf(metav1.Duration{}):                 true,
	reflect.TypeOf(metav1.TypeMeta{}):                 true,
	reflect.TypeOf(metav1.ListMeta{}):                 true,
	reflect.TypeOf(metav1.ObjectMeta{}):               true,
	reflect.TypeOf(metav1.OwnerReference{}):           true,
	reflect.TypeOf(metav1.LabelSelector{}):            true,
	reflect.TypeOf(metav1.LabelSelectorRequirement{}): true,
	reflect.TypeOf(metav1.GetOptions{}):               true,
	reflect.TypeOf(metav1.ListOptions{}):              true,
	reflect.TypeOf(metav1.DeleteOptions{}):            true,
	reflect.TypeOf(metav1.GroupVersionKind{}):         true,
	reflect.TypeOf(metav1.GroupVersionResource{}):     true,
	reflect.TypeOf(metav1.Status{}):                   true,
	reflect.TypeOf(metav1.Condition{}):                true,
}

// These fields are limited exceptions to the standard JSON naming structure.
// Additions should only be made if a non-standard field name was released and cannot be changed for compatibility reasons.
var allowedNonstandardJSONNames = map[reflect.Type]string{
	reflect.TypeOf(v1.DaemonEndpoint{}): "Port",
}

func TestTypeTags(t *testing.T) {
	if err := apinamingtest.VerifyTagNaming(legacyscheme.Scheme, typesAllowedTags, allowedNonstandardJSONNames); err != nil {
		t.Errorf("%v", err)
	}
}
