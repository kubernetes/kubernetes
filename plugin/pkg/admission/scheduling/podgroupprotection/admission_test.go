/*
Copyright The Kubernetes Authors.

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

package podgroupprotection

import (
	"context"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	schedulingapi "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/dump"
)

func TestAdmit(t *testing.T) {
	pg := &schedulingapi.PodGroup{
		TypeMeta: metav1.TypeMeta{Kind: "PodGroup"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-podgroup",
			Namespace: "default",
		},
	}

	pgWithFinalizer := pg.DeepCopy()
	pgWithFinalizer.Finalizers = []string{schedulingapi.PodGroupProtectionFinalizer}

	tests := []struct {
		name           string
		enabled        bool
		resource       schema.GroupVersionResource
		object         runtime.Object
		expectedObject runtime.Object
		namespace      string
	}{
		{
			name:           "podgroup create with plugin enabled, add finalizer",
			enabled:        true,
			resource:       schedulingapi.SchemeGroupVersion.WithResource("podgroups"),
			object:         pg,
			expectedObject: pgWithFinalizer,
			namespace:      pg.Namespace,
		},
		{
			name:           "podgroup finalizer already exists, no new finalizer",
			enabled:        true,
			resource:       schedulingapi.SchemeGroupVersion.WithResource("podgroups"),
			object:         pgWithFinalizer,
			expectedObject: pgWithFinalizer,
			namespace:      pgWithFinalizer.Namespace,
		},
		{
			name:           "podgroup create with plugin disabled, no finalizer added",
			enabled:        false,
			resource:       schedulingapi.SchemeGroupVersion.WithResource("podgroups"),
			object:         pg,
			expectedObject: pg,
			namespace:      pg.Namespace,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, test.enabled)

			ctrl := newPlugin()
			ctrl.InspectFeatureGates(utilfeature.DefaultFeatureGate)

			obj := test.object.DeepCopyObject()
			attrs := admission.NewAttributesRecord(
				obj,
				obj.DeepCopyObject(),
				schema.GroupVersionKind{},
				test.namespace,
				"foo",
				test.resource,
				"",
				admission.Create,
				&metav1.CreateOptions{},
				false,
				nil,
			)

			if err := ctrl.Admit(context.TODO(), attrs, nil); err != nil {
				t.Errorf("got unexpected error: %v", err)
			}
			if !reflect.DeepEqual(test.expectedObject, obj) {
				t.Errorf("Expected object:\n%s\ngot:\n%s", dump.Pretty(test.expectedObject), dump.Pretty(obj))
			}
		})
	}
}
