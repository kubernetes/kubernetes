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

package v1alpha1

import (
	"encoding/json"
	"reflect"
	"testing"

	componentconfig "k8s.io/kubernetes/pkg/apis/componentconfig"
)

func TestSchedulerDefaults(t *testing.T) {
	ks1 := &KubeSchedulerConfiguration{}
	SetDefaults_KubeSchedulerConfiguration(ks1)
	cm, err := componentconfig.ConvertObjToConfigMap("KubeSchedulerConfiguration", ks1)
	if err != nil {
		t.Errorf("unexpected ConvertObjToConfigMap error %v", err)
	}

	ks2 := &KubeSchedulerConfiguration{}
	if err = json.Unmarshal([]byte(cm.Data["KubeSchedulerConfiguration"]), ks2); err != nil {
		t.Errorf("unexpected error unserializing scheduler config %v", err)
	}

	if !reflect.DeepEqual(ks2, ks1) {
		t.Errorf("Expected:\n%#v\n\nGot:\n%#v", ks1, ks2)
	}
}
