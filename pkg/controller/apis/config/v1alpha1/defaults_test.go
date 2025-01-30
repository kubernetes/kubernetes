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

package v1alpha1

import (
	"encoding/json"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	kubectrlmgrconfigv1alpha1 "k8s.io/kube-controller-manager/config/v1alpha1"
)

func TestKubeControllerDefaultsRoundTrip(t *testing.T) {
	ks1 := &kubectrlmgrconfigv1alpha1.KubeControllerManagerConfiguration{}
	SetDefaults_KubeControllerManagerConfiguration(ks1)
	cm, err := convertObjToConfigMap("KubeControllerManagerConfiguration", ks1)
	if err != nil {
		t.Errorf("unexpected ConvertObjToConfigMap error %v", err)
	}

	ks2 := &kubectrlmgrconfigv1alpha1.KubeControllerManagerConfiguration{}
	if err = json.Unmarshal([]byte(cm.Data["KubeControllerManagerConfiguration"]), ks2); err != nil {
		t.Errorf("unexpected error unserializing controller manager config %v", err)
	}

	if !reflect.DeepEqual(ks2, ks1) {
		t.Errorf("Expected:\n%#v\n\nGot:\n%#v", ks1, ks2)
	}
}

// convertObjToConfigMap converts an object to a ConfigMap.
// This is specifically meant for ComponentConfigs.
func convertObjToConfigMap(name string, obj runtime.Object) (*v1.ConfigMap, error) {
	eJSONBytes, err := json.Marshal(obj)
	if err != nil {
		return nil, err
	}
	cm := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Data: map[string]string{
			name: string(eJSONBytes[:]),
		},
	}
	return cm, nil
}
