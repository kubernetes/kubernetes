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

package v1

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/cache"
)

// BenchmarkGetPodPodDisruptionBudgets looks up the one PDB matching a pod among
// n PDBs in its namespace. Each PDB selects the same labels, written as
// matchLabels, as matchExpressions, or split between the two.
func BenchmarkGetPodPodDisruptionBudgets(b *testing.B) {
	selectorShapes := []struct {
		name string
		// exprKeys are the labels written as matchExpressions, the rest are matchLabels.
		exprKeys sets.Set[string]
	}{
		{"matchLabels", sets.New[string]()},
		{"matchExpressions", sets.KeySet(benchMatchLabels(0))},
		{"mixed", sets.New("environment", "partition", "service")},
	}
	for _, shape := range selectorShapes {
		for _, n := range []int{100, 1000, 2000} {
			b.Run(fmt.Sprintf("selector=%s/pdbs=%d", shape.name, n), func(b *testing.B) {
				indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
				for i := 0; i < n; i++ {
					pdb := &policy.PodDisruptionBudget{
						ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: fmt.Sprintf("pdb-%d", i)},
						Spec:       policy.PodDisruptionBudgetSpec{Selector: benchSelector(benchMatchLabels(i), shape.exprKeys)},
					}
					if err := indexer.Add(pdb); err != nil {
						b.Fatal(err)
					}
				}
				lister := NewPodDisruptionBudgetLister(indexer)

				podLabels := benchMatchLabels(n / 2)
				podLabels["pod-template-hash"] = "5d8f9c7b6d"
				pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pod", Labels: podLabels}}

				b.ReportAllocs()
				for b.Loop() {
					pdbs, err := lister.GetPodPodDisruptionBudgets(pod)
					if err != nil {
						b.Fatal(err)
					}
					if len(pdbs) != 1 {
						b.Fatalf("got %d matching PDBs, want 1", len(pdbs))
					}
				}
			})
		}
	}
}

// benchSelector returns a selector for labels, writing the keys in exprKeys as
// single-value In matchExpressions and the rest as matchLabels.
func benchSelector(labels map[string]string, exprKeys sets.Set[string]) *metav1.LabelSelector {
	selector := &metav1.LabelSelector{}
	for _, k := range sets.List(sets.KeySet(labels)) {
		if exprKeys.Has(k) {
			selector.MatchExpressions = append(selector.MatchExpressions, metav1.LabelSelectorRequirement{
				Key:      k,
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{labels[k]},
			})
			continue
		}
		if selector.MatchLabels == nil {
			selector.MatchLabels = map[string]string{}
		}
		selector.MatchLabels[k] = labels[k]
	}
	return selector
}

// benchMatchLabels returns the labels PDB i selects: 6 labels, one shared by
// every PDB and one unique per PDB
func benchMatchLabels(i int) map[string]string {
	return map[string]string{
		"app.kubernetes.io/managed-by": "op",                    // shared, sorts first
		"deployment-name-hash":         fmt.Sprintf("%032d", i), // unique per PDB
		"deployment-number":            fmt.Sprintf("%02d", i%100),
		"environment":                  fmt.Sprintf("env-%09d", i%389),
		"partition":                    fmt.Sprintf("partition-%021d", i%58),
		"service":                      fmt.Sprintf("service-%013d", i%1380),
	}
}
