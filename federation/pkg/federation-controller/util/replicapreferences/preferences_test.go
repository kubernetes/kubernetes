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

package replicapreferences

import (
	"testing"

	extensionsv1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/runtime"
)

const (
	TestPreferencesAnnotationKey = "federation.kubernetes.io/test-preferences"
)

func TestGetAllocationPreferences(t *testing.T) {
	testCases := []struct {
		testname      string
		prefs         string
		obj           runtime.Object
		errorExpected bool
	}{
		{
			testname: "good preferences",
			prefs: `{"rebalance": true,
				  "clusters": {
				    "k8s-1": {"minReplicas": 10, "maxReplicas": 20, "weight": 2},
				    "*": {"weight": 1}
				}}`,
			obj: &extensionsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-obj",
					Namespace: metav1.NamespaceDefault,
					SelfLink:  "/api/v1/namespaces/default/obj/test-obj",
				},
			},
			errorExpected: false,
		},
		{
			testname: "failed preferences",
			prefs:    `{`, // bad json
			obj: &extensionsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-obj",
					Namespace: metav1.NamespaceDefault,
					SelfLink:  "/api/v1/namespaces/default/obj/test-obj",
				},
			},
			errorExpected: true,
		},
	}

	// prepare the objects
	for _, tc := range testCases {
		accessor, _ := meta.Accessor(tc.obj)
		anno := accessor.GetAnnotations()
		if anno == nil {
			anno = make(map[string]string)
			accessor.SetAnnotations(anno)
		}
		anno[TestPreferencesAnnotationKey] = tc.prefs
	}

	// test get preferences
	for _, tc := range testCases {
		pref, err := GetAllocationPreferences(tc.obj, TestPreferencesAnnotationKey)
		if tc.errorExpected {
			assert.NotNil(t, err)
		} else {
			assert.NotNil(t, pref)
			assert.Nil(t, err)
		}
	}
}
