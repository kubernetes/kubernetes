/*
Copyright The Kubernetes Authors.

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

package history

import (
	"fmt"
	"testing"

	apps "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/controller"
)

func BenchmarkListControllerRevisions(b *testing.B) {
	parent := &apps.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-parent",
			Namespace: "default",
			UID:       types.UID("test-parent-uid"),
		},
	}
	parentKind := schema.GroupVersionKind{
		Group:   "apps",
		Version: "v1",
		Kind:    "StatefulSet",
	}

	scenarios := []struct {
		name             string
		totalRevisions   int
		ownedPercentage  float64 // percentage of totalRevisions that are owned
		orphanPercentage float64 // percentage of totalRevisions that are orphans

	}{
		{
			name:             "0.1pct_Owned_0pct_Orphan",
			totalRevisions:   50000,
			ownedPercentage:  0.1,
			orphanPercentage: 0,
		},
		{
			name:             "0.1pct_Owned_0.1pct_Orphan",
			totalRevisions:   50000,
			ownedPercentage:  0.1,
			orphanPercentage: 0.1,
		},
		{
			name:             "1pct_Owned_0pct_Orphan",
			totalRevisions:   50000,
			ownedPercentage:  1,
			orphanPercentage: 0,
		},
		{
			name:             "1pct_Owned_1pct_Orphan",
			totalRevisions:   50000,
			ownedPercentage:  1,
			orphanPercentage: 1,
		},
		{
			name:             "10pct_Owned_0pct_Orphan",
			totalRevisions:   50000,
			ownedPercentage:  10,
			orphanPercentage: 0,
		},
		{
			name:             "10pct_Owned_10pct_Orphan",
			totalRevisions:   50000,
			ownedPercentage:  10,
			orphanPercentage: 10,
		},
		{
			name:             "100pct_Owned_0pct_Orphan",
			totalRevisions:   50000,
			ownedPercentage:  100,
			orphanPercentage: 0,
		},
	}

	for _, s := range scenarios {
		b.Run(s.name, func(b *testing.B) {
			ownedRevisions := int(float64(s.totalRevisions) * s.ownedPercentage / 100)
			orphans := int(float64(s.totalRevisions) * s.orphanPercentage / 100)
			otherOwned := s.totalRevisions - ownedRevisions - orphans

			client := fake.NewClientset()
			informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
			revisionInformer := informerFactory.Apps().V1().ControllerRevisions()

			if err := AddControllerRevisionControllerIndexer(revisionInformer.Informer()); err != nil {
				b.Fatalf("failed to add indexer: %v", err)
			}

			revisions := make([]runtime.Object, 0, s.totalRevisions)
			for i := 0; i < ownedRevisions; i++ {
				revisions = append(revisions, createRevision(s.name, i, parent, parentKind))
			}

			// Create orphaned revisions (nil controller ref)
			for i := 0; i < orphans; i++ {
				rev := createRevision(s.name, ownedRevisions+i, nil, schema.GroupVersionKind{})
				// Make sure it has no owner ref or at least no controller ref
				rev.OwnerReferences = nil
				revisions = append(revisions, rev)
			}

			// Create revisions owned by other controllers
			otherParent := &apps.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "other-parent",
					Namespace: "default",
					UID:       types.UID("other-parent-uid"),
				},
			}
			for i := 0; i < otherOwned; i++ {
				revisions = append(revisions, createRevision(s.name, ownedRevisions+orphans+i, otherParent, parentKind))
			}

			indexer := revisionInformer.Informer().GetIndexer()
			for _, obj := range revisions {
				if err := indexer.Add(obj); err != nil {
					b.Fatalf("failed to add object to indexer: %v", err)
				}
			}

			history := NewHistory(client, revisionInformer.Lister(), indexer)
			selector, _ := labels.Parse("name=" + s.name)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := history.ListControllerRevisions(parent, parentKind, selector)
				if err != nil {
					b.Fatalf("ListControllerRevisions failed: %v", err)
				}
			}
		})
	}
}

func createRevision(scenarioName string, index int, owner metav1.Object, ownerKind schema.GroupVersionKind) *apps.ControllerRevision {
	rev := &apps.ControllerRevision{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-%d", scenarioName, index),
			Namespace: "default",
			Labels: map[string]string{
				"name": scenarioName,
			},
			UID: types.UID(fmt.Sprintf("%s-%d", scenarioName, index)), // Ensure unique UID
		},
		Data:     runtime.RawExtension{Raw: []byte("{}")},
		Revision: int64(index),
	}

	if owner != nil {
		rev.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(owner, ownerKind)}
	}

	return rev
}
