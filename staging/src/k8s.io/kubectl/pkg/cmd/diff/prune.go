/*
Copyright 2021 The Kubernetes Authors.

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

package diff

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubectl/pkg/util/prune"
)

type pruner struct {
	mapper        meta.RESTMapper
	dynamicClient dynamic.Interface

	visitedUids       sets.String
	visitedNamespaces sets.String
	labelSelector     string
	resources         []prune.Resource
}

func newPruner(dc dynamic.Interface, m meta.RESTMapper, r []prune.Resource) *pruner {
	return &pruner{
		visitedUids:       sets.NewString(),
		visitedNamespaces: sets.NewString(),
		dynamicClient:     dc,
		mapper:            m,
		resources:         r,
	}
}

func (p *pruner) pruneAll() ([]runtime.Object, error) {
	var allPruned []runtime.Object
	namespacedRESTMappings, nonNamespacedRESTMappings, err := prune.GetRESTMappings(p.mapper, p.resources)
	if err != nil {
		return allPruned, fmt.Errorf("error retrieving RESTMappings to prune: %v", err)
	}

	for n := range p.visitedNamespaces {
		for _, m := range namespacedRESTMappings {
			if pobjs, err := p.prune(n, m); err != nil {
				return pobjs, fmt.Errorf("error pruning namespaced object %v: %v", m.GroupVersionKind, err)
			} else {
				allPruned = append(allPruned, pobjs...)
			}
		}
	}
	for _, m := range nonNamespacedRESTMappings {
		if pobjs, err := p.prune(metav1.NamespaceNone, m); err != nil {
			return allPruned, fmt.Errorf("error pruning nonNamespaced object %v: %v", m.GroupVersionKind, err)
		} else {
			allPruned = append(allPruned, pobjs...)
		}
	}

	return allPruned, nil
}

func (p *pruner) prune(namespace string, mapping *meta.RESTMapping) ([]runtime.Object, error) {
	objList, err := p.dynamicClient.Resource(mapping.Resource).
		Namespace(namespace).
		List(context.TODO(), metav1.ListOptions{
			LabelSelector: p.labelSelector,
		})
	if err != nil {
		return nil, err
	}

	objs, err := meta.ExtractList(objList)
	if err != nil {
		return nil, err
	}

	var pobjs []runtime.Object
	for _, obj := range objs {
		metadata, err := meta.Accessor(obj)
		if err != nil {
			return pobjs, err
		}
		annots := metadata.GetAnnotations()
		if _, ok := annots[corev1.LastAppliedConfigAnnotation]; !ok {
			continue
		}
		uid := metadata.GetUID()
		if p.visitedUids.Has(string(uid)) {
			continue
		}

		pobjs = append(pobjs, obj)
	}
	return pobjs, nil
}

// MarkVisited marks visited namespaces and uids
func (p *pruner) MarkVisited(info *resource.Info) {
	if info.Namespaced() {
		p.visitedNamespaces.Insert(info.Namespace)
	}

	metadata, err := meta.Accessor(info.Object)
	if err != nil {
		return
	}
	p.visitedUids.Insert(string(metadata.GetUID()))
}
