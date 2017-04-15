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

package util

import (
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"

	"github.com/stretchr/testify/assert"
)

const (
	TestPreferencesAnnotationKey = "federation.kubernetes.io/test-preferences"
)

func TestParsePreferences(t *testing.T) {
	successPrefs := []string{
		`{"rebalance": true,
		  "clusters": {
		    "k8s-1": {"minReplicas": 10, "maxReplicas": 20, "weight": 2},
		    "*": {"weight": 1}
		}}`,
	}
	failedPrefes := []string{
		`{`, // bad json
	}

	obj := &extensionsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-obj",
			Namespace: metav1.NamespaceDefault,
			SelfLink:  "/api/v1/namespaces/default/obj/test-obj",
		},
	}

	accessor, _ := meta.Accessor(obj)
	anno := accessor.GetAnnotations()
	if anno == nil {
		anno = make(map[string]string)
		accessor.SetAnnotations(anno)
	}
	for _, prefString := range successPrefs {
		anno[TestPreferencesAnnotationKey] = prefString
		pref, err := UnmarshalPreferences(obj, TestPreferencesAnnotationKey)
		assert.NotNil(t, pref)
		assert.Nil(t, err)
	}
	for _, prefString := range failedPrefes {
		anno[TestPreferencesAnnotationKey] = prefString
		pref, err := UnmarshalPreferences(obj, TestPreferencesAnnotationKey)
		assert.Nil(t, pref)
		assert.NotNil(t, err)
	}

	wrongObj := &v1.Secret{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Secret",
			APIVersion: testapi.Default.GroupVersion().String(),
		},
	}
	expectedErr := fmt.Errorf("Unknnown object type while parsing annotations %v", wrongObj)
	for _, prefString := range successPrefs {
		anno[TestPreferencesAnnotationKey] = prefString
		pref, err := UnmarshalPreferences(wrongObj, TestPreferencesAnnotationKey)
		assert.Nil(t, pref)
		assert.Equal(t, err, expectedErr)
	}
}
