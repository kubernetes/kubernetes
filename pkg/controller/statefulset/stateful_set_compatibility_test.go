/*
Copyright 2025 The Kubernetes Authors.

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

package statefulset

import (
	"os"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"sigs.k8s.io/json"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

func TestStatefulSetCompatibility(t *testing.T) {
	set133 := &appsv1.StatefulSet{}
	set134 := &appsv1.StatefulSet{}
	rev133 := &appsv1.ControllerRevision{}
	rev134 := &appsv1.ControllerRevision{}
	load(t, "compatibility_set_1.33.0.json", set133)
	load(t, "compatibility_set_1.34.0.json", set134)
	load(t, "compatibility_revision_1.33.0.json", rev133)
	load(t, "compatibility_revision_1.34.0.json", rev134)

	testcases := []struct {
		name      string
		set       *appsv1.StatefulSet
		revisions []*appsv1.ControllerRevision
	}{
		{
			name:      "1.33 set, 1.33 rev",
			set:       set133.DeepCopy(),
			revisions: []*appsv1.ControllerRevision{rev133.DeepCopy()},
		},
		{
			name:      "1.34 set, 1.34 rev",
			set:       set134.DeepCopy(),
			revisions: []*appsv1.ControllerRevision{rev134.DeepCopy()},
		},
		{
			name:      "1.34 set, 1.33+1.34 rev",
			set:       set134.DeepCopy(),
			revisions: []*appsv1.ControllerRevision{rev133.DeepCopy(), rev134.DeepCopy()},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			latestRev := tc.revisions[len(tc.revisions)-1]
			client := fake.NewClientset(tc.set)
			_, _, ssc := setupController(client)
			currentRev, updateRev, _, err := ssc.(*defaultStatefulSetControl).getStatefulSetRevisions(tc.set, tc.revisions)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(currentRev, latestRev) {
				t.Fatalf("expected no change from latestRev, got %s", cmp.Diff(latestRev, currentRev))
			}
			if !reflect.DeepEqual(updateRev, latestRev) {
				t.Fatalf("expected no change from latestRev, got %s", cmp.Diff(latestRev, updateRev))
			}
		})
	}
}

func BenchmarkStatefulSetCompatibility(b *testing.B) {
	set133 := &appsv1.StatefulSet{}
	set134 := &appsv1.StatefulSet{}
	rev133 := &appsv1.ControllerRevision{}
	rev134 := &appsv1.ControllerRevision{}
	load(b, "compatibility_set_1.33.0.json", set133)
	load(b, "compatibility_set_1.34.0.json", set134)
	load(b, "compatibility_revision_1.33.0.json", rev133)
	load(b, "compatibility_revision_1.34.0.json", rev134)

	testcases := []struct {
		name      string
		set       *appsv1.StatefulSet
		revisions []*appsv1.ControllerRevision
	}{
		{
			name:      "1.33 set, 1.33 rev",
			set:       set133.DeepCopy(),
			revisions: []*appsv1.ControllerRevision{rev133.DeepCopy()},
		},
		{
			name:      "1.34 set, 1.34 rev",
			set:       set134.DeepCopy(),
			revisions: []*appsv1.ControllerRevision{rev134.DeepCopy()},
		},
		{
			name:      "1.34 set, 1.33+1.34 rev",
			set:       set134.DeepCopy(),
			revisions: []*appsv1.ControllerRevision{rev133.DeepCopy(), rev134.DeepCopy()},
		},
	}

	for _, tc := range testcases {
		b.Run(tc.name, func(b *testing.B) {
			latestRev := tc.revisions[len(tc.revisions)-1]
			client := fake.NewClientset(tc.set)
			_, _, ssc := setupController(client)
			for i := 0; i < b.N; i++ {
				currentRev, updateRev, _, err := ssc.(*defaultStatefulSetControl).getStatefulSetRevisions(tc.set, tc.revisions)
				if err != nil {
					b.Fatal(err)
				}
				if !reflect.DeepEqual(currentRev, latestRev) {
					b.Fatalf("expected no change from latestRev, got %s", cmp.Diff(latestRev, currentRev))
				}
				if !reflect.DeepEqual(updateRev, latestRev) {
					b.Fatalf("expected no change from latestRev, got %s", cmp.Diff(latestRev, updateRev))
				}
			}
		})
	}
}

func load(t testing.TB, filename string, object runtime.Object) {
	data, err := os.ReadFile("testdata/" + filename)
	if err != nil {
		t.Fatal(err)
	}
	if strictErrs, err := json.UnmarshalStrict(data, object); err != nil {
		t.Fatal(err)
	} else if len(strictErrs) > 0 {
		t.Fatal(strictErrs)
	}
	// apply defaulting just as if it was read from etcd
	legacyscheme.Scheme.Default(object)
}
