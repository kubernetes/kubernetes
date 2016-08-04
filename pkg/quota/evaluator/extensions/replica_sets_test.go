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

package extensions

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	extensionsapi "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/quota"
)

func TestReplicaSetEvaluatorMatchesResources(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	evaluator := NewReplicaSetEvaluator(kubeClient)
	expected := quota.ToSet([]api.ResourceName{
		extensionsapi.ResourceReplicaSets,
	})
	actual := quota.ToSet(evaluator.MatchesResources())
	if !expected.Equal(actual) {
		t.Errorf("expected: %v, actual: %v", expected, actual)
	}
}

func TestReplicaSetEvaluatorUsage(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	evaluator := NewReplicaSetEvaluator(kubeClient)
	testCases := map[string]struct {
		rs    *extensionsapi.ReplicaSet
		usage api.ResourceList
	}{
		"single replica set": {
			rs: &extensionsapi.ReplicaSet{
				Spec: extensionsapi.ReplicaSetSpec{},
			},
			usage: api.ResourceList{
				extensionsapi.ResourceReplicaSets: resource.MustParse("1"),
			},
		},
	}
	for testName, testCase := range testCases {
		actual := evaluator.Usage(testCase.rs)
		if !quota.Equals(testCase.usage, actual) {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.usage, actual)
		}
	}
}
