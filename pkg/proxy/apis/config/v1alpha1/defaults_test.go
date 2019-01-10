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
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	configtesting "k8s.io/component/pkg/testing"
)

func TestDefaulting(t *testing.T) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	if err := AddToScheme(scheme); err != nil {
		t.Fatalf("programmer error: AddToScheme must not error: %v", err)
	}
	tc := configtesting.GetDefaultingTestCases(scheme)
	t.Logf("%v", tc)
	configtesting.RunTestsOnYAMLData(t, tc, scheme, &codecs)
	t.Error("foo")
}
