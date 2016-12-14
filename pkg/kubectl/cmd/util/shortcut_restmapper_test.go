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
	"errors"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api/testapi"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/runtime/schema"
)

func TestReplaceAliases(t *testing.T) {
	srvResDefault := []*metav1.APIResourceList{
		{
			GroupVersion: "catalog/v1",
			APIResources: []metav1.APIResource{
				{
					Name:      "servicecatalog",
					ShortName: "sc",
				},
			},
		},
		{
			GroupVersion: "storage.k8s.io/v1beta1",
			APIResources: []metav1.APIResource{
				{
					Name:      "storageclass",
					ShortName: "sc",
				},
			},
		},
	}
	tests := []struct {
		name      string
		arg       string
		expected  string
		srvRes    []*metav1.APIResourceList
		srvResErr bool
	}{
		{
			name:     "no-replacement",
			arg:      "service",
			expected: "service",
		},
		{
			name:     "all-replacement",
			arg:      "all",
			expected: "pods,replicationcontrollers,services,statefulsets,horizontalpodautoscalers,jobs,deployments,replicasets",
		},
		{
			name:     "alias-in-comma-separated-arg",
			arg:      "all,secrets",
			expected: "pods,replicationcontrollers,services,statefulsets,horizontalpodautoscalers,jobs,deployments,replicasets,secrets",
		},
		{
			name:      "sc-replacement-from-srv-resources",
			arg:       "sc",
			expected:  "servicecatalog",
			srvRes:    srvResDefault,
			srvResErr: false,
		},
		{
			name:      "po-replacement-from-hardcoded-resources",
			arg:       "po",
			expected:  "pods",
			srvRes:    srvResDefault,
			srvResErr: false,
		},
		{
			name:      "sc-untouched-srv-error",
			arg:       "sc",
			expected:  "sc",
			srvRes:    srvResDefault,
			srvResErr: true,
		},
	}

	c := NewFakeDiscoveryClient()
	mapper := NewShortcutExpander(testapi.Default.RESTMapper(), c)

	for _, test := range tests {
		if test.srvRes != nil {
			c.ServerResourcesHandler = func() ([]*metav1.APIResourceList, error) {
				if test.srvResErr {
					return nil, errors.New("Some nasty error")
				}
				return test.srvRes, nil
			}
		}
		resources := []string{}
		for _, arg := range strings.Split(test.arg, ",") {
			curr, _ := mapper.AliasesForResource(arg)
			resources = append(resources, curr...)
		}
		if strings.Join(resources, ",") != test.expected {
			t.Errorf("%s: unexpected argument: expected %s, got %s", test.name, test.expected, resources)
		}
	}
}

func TestKindFor(t *testing.T) {
	dc := NewFakeDiscoveryClient()
	target := NewShortcutExpander(testapi.Default.RESTMapper(), dc)
	expectedGVK := schema.GroupVersionKind{Group: "extensions", Version: "v1beta1", Kind: "HorizontalPodAutoscaler"}

	dc.ServerResourcesForGroupVersionHandler = func(groupVersion string) (*metav1.APIResourceList, error) {
		return &metav1.APIResourceList{
			GroupVersion: schema.GroupVersion{Group: "extensions", Version: "v1beta1"}.String(),
			APIResources: []metav1.APIResource{
				{
					Name:      "horizontalpodautoscalers",
					ShortName: "abc",
				},
			},
		}, nil
	}

	ret, err := target.KindFor(schema.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "abc"})
	if err != nil {
		t.Errorf("unexpected error returned %s", err.Error())
	}
	if ret != expectedGVK {
		t.Errorf("unexpected data returned %#v, expected %#v", ret, expectedGVK)
	}
}
