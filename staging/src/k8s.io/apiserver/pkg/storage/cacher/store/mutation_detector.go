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

package store

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"weak"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
)

type mutationDetector struct {
	// The writable B-tree, latest snapshot, historical snapshots, and index
	// buckets share *Element pointers. GET and unindexed current-state LISTs
	// read the latest snapshot. Indexed LISTs read a listSnapshot of pointers
	// from an index bucket. Exact-RV and continued LISTs can read historical
	// snapshots. A single baseline covers each element across these views.
	//
	// Weak keys let GC reclaim elements after the store, snapshots, index
	// buckets, and in-flight readers release them. An in-flight LIST can
	// still hold a snapshot after compaction.
	//
	// The map allows store writes to register baselines during a detector scan.
	baselines sync.Map // weak.Pointer[Element] -> runtime.Object
}

func (d *mutationDetector) track(elem *Element) {
	d.baselines.LoadOrStore(weak.Make(elem), elem.Object.DeepCopyObject())
}

func (d *mutationDetector) checkMutations(ctx context.Context) {
	// Range may skip baselines added during the scan. The next complete scan
	// checks them if their elements are still live.
	d.baselines.Range(func(key, value any) bool {
		if ctx.Err() != nil {
			return false
		}
		elem := key.(weak.Pointer[Element]).Value()
		if elem == nil {
			// A delete or replacement can remove the element from current state.
			// Compaction can drop historical snapshots that still hold it. Once
			// the last strong reference is gone and GC clears the weak pointer,
			// drop the baseline so its deep copy can be collected.
			d.baselines.Delete(key)
			return true
		}
		baseline, ok := value.(runtime.Object)
		if !ok {
			panic(fmt.Sprintf("watch cache mutation detector: invalid baseline for key %q: got %T, want runtime.Object", elem.Key, value))
		}
		if reflect.DeepEqual(elem.Object, baseline) {
			return true
		}
		if obj, err := meta.Accessor(baseline); err == nil {
			name := obj.GetName()
			if namespace := obj.GetNamespace(); namespace != "" {
				name = namespace + "/" + name
			}
			panic(fmt.Sprintf("watch cache mutation detected for %T %q, admitted at RV %q", baseline, name, obj.GetResourceVersion()))
		}
		panic(fmt.Sprintf("watch cache mutation detected for %T with cache key %q", baseline, elem.Key))
	})
}
