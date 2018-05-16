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

package testdata

import (
	"io/ioutil"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

const test196 = "testdata/kubeadm196.yaml"

func TestUpgrade(t *testing.T) {
	testYAML, err := ioutil.ReadFile(test196)
	if err != nil {
		t.Fatalf("couldn't read test data: %v", err)
	}

	decoded, err := kubeadmutil.LoadYAML(testYAML)
	if err != nil {
		t.Fatalf("couldn't unmarshal test yaml: %v", err)
	}

	scheme := runtime.NewScheme()
	AddToScheme(scheme)
	codecs := serializer.NewCodecFactory(scheme)

	obj := &MasterConfiguration{}
	if err := Migrate(decoded, obj, codecs); err != nil {
		t.Fatalf("couldn't decode migrated object: %v", err)
	}
}
