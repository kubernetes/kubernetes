/*
Copyright 2022 The KCP Authors.

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

package cacher

import (
	"testing"

	"github.com/kcp-dev/logicalcluster/v3"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

func TestCreateClusterAwareContext(t *testing.T) {
	scenarios := []struct {
		name            string
		existingObject  runtime.Object
		expectedCluster genericapirequest.Cluster
	}{
		{
			name:            "scoped, built-in type",
			existingObject:  makePod("pod1", "root:org:abc", "default"),
			expectedCluster: genericapirequest.Cluster{Name: logicalcluster.New("root:org:abc")},
		},
		{
			name: "cluster wide, built-in type",
			existingObject: func() *v1.Pod {
				p := makePod("pod1", "root:org:abc", "default")
				p.Namespace = ""
				return p
			}(),
			expectedCluster: genericapirequest.Cluster{Name: logicalcluster.New("root:org:abc")},
		},
		{
			name:            "scoped, identity, built-in type",
			existingObject:  makePod("pod1", "root:org:abc", "default"),
			expectedCluster: genericapirequest.Cluster{Name: logicalcluster.New("root:org:abc")},
		},

		{
			name:            "scoped, unstructured type",
			existingObject:  makeUnstructured("root:org:abc", "default"),
			expectedCluster: genericapirequest.Cluster{Name: logicalcluster.New("root:org:abc")},
		},
		{
			name:            "cluster wide, unstructured type",
			existingObject:  makeUnstructured("root:org:abc", ""),
			expectedCluster: genericapirequest.Cluster{Name: logicalcluster.New("root:org:abc")},
		},
		{
			name:            "scoped, identity, unstructured type",
			existingObject:  makeUnstructured("root:org:abc", "default"),
			expectedCluster: genericapirequest.Cluster{Name: logicalcluster.New("root:org:abc")},
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			actualCtx := createClusterAwareContext(scenario.existingObject)
			actualCluster, err := genericapirequest.ValidClusterFrom(actualCtx)
			if err != nil {
				t.Fatal(err)
			}
			if *actualCluster != scenario.expectedCluster {
				t.Errorf("expected %v, got %v", scenario.expectedCluster, actualCluster)
			}
		})
	}
}

func makePod(name, clusterName, ns string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns,
			Name:      name,
			Annotations: map[string]string{
				logicalcluster.AnnotationKey: clusterName,
			},
		},
	}
}

func makeUnstructured(clusterName, ns string) *unstructured.Unstructured {
	obj := &unstructured.Unstructured{}
	obj.SetAnnotations(map[string]string{
		logicalcluster.AnnotationKey: clusterName,
	})
	obj.SetNamespace(ns)
	return obj
}
