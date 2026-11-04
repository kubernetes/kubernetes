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

package images

import (
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

// UseTracker records images that the kubelet is about to run containers
// from, so that image garbage collection does not remove them.
//
// Image GC decides which images are in use from a single snapshot of the
// runtime's containers, then removes the unused ones one by one. A container
// that is created from an already present image after that snapshot is
// invisible to the GC run, so the image would be removed from under it.
// Callers mark an image with Acquire once they found it and release it once
// the container exists in the runtime, and GC re-checks the marks right
// before each removal. Removals tells callers whether the image may have been
// removed between finding it and marking it.
type UseTracker struct {
	lock sync.Mutex
	// removalDone is signalled every time GC finishes removing an image.
	removalDone *sync.Cond

	// refs counts the callers that are using an image but have not released it yet.
	refs map[string]int
	// lastReleased is when an image was last released. GC runs that took their
	// snapshot of containers before this time may not have seen the container.
	lastReleased map[string]time.Time
	// removing holds the images that GC is removing right now.
	removing sets.Set[string]
	// removals is bumped whenever GC starts or finishes removing any image,
	// which lets callers notice that an image they found may be gone.
	removals uint64
}

// NewUseTracker returns an empty UseTracker.
func NewUseTracker() *UseTracker {
	t := &UseTracker{
		refs:         make(map[string]int),
		lastReleased: make(map[string]time.Time),
		removing:     sets.New[string](),
	}
	t.removalDone = sync.NewCond(&t.lock)
	return t
}

// useTrackerKey matches the keys of the image GC records.
func useTrackerKey(imageRef, runtimeHandler string) string {
	if utilfeature.DefaultFeatureGate.Enabled(features.RuntimeClassInImageCriAPI) {
		return getImageTuple(imageRef, runtimeHandler)
	}
	return imageRef
}

// Removals returns a counter that changes whenever image GC starts or
// finishes removing an image. A caller reads it before looking up an image
// and compares it after Acquire: if it is unchanged, no removal overlapped
// the lookup, so the image is still present.
func (t *UseTracker) Removals() uint64 {
	t.lock.Lock()
	defer t.lock.Unlock()
	return t.removals
}

// Acquire marks the image as in use until the returned function is called.
// If GC is removing the image right now, Acquire waits for the removal to
// finish, so the caller must check afterwards that the image still exists.
func (t *UseTracker) Acquire(imageRef, runtimeHandler string) (release func()) {
	key := useTrackerKey(imageRef, runtimeHandler)

	t.lock.Lock()
	for t.removing.Has(key) {
		t.removalDone.Wait()
	}
	t.refs[key]++
	t.lock.Unlock()

	var once sync.Once
	return func() {
		once.Do(func() {
			t.lock.Lock()
			defer t.lock.Unlock()
			t.refs[key]--
			if t.refs[key] <= 0 {
				delete(t.refs, key)
			}
			t.lastReleased[key] = time.Now()
		})
	}
}

// startRemoval reports whether GC may remove the image, given that it took
// its snapshot of running containers at snapshotTime. If it returns true, GC
// must call finishRemoval once the removal is done.
func (t *UseTracker) startRemoval(key string, snapshotTime time.Time) bool {
	t.lock.Lock()
	defer t.lock.Unlock()
	if t.refs[key] > 0 {
		return false
	}
	if released, ok := t.lastReleased[key]; ok && !released.Before(snapshotTime) {
		return false
	}
	t.removing.Insert(key)
	t.removals++
	return true
}

func (t *UseTracker) finishRemoval(key string) {
	t.lock.Lock()
	defer t.lock.Unlock()
	t.removing.Delete(key)
	t.removals++
	t.removalDone.Broadcast()
}

// forgetAbsent drops release times of images that are no longer present and
// not in use, so the tracker does not grow with every image ever used.
func (t *UseTracker) forgetAbsent(present sets.Set[string]) {
	t.lock.Lock()
	defer t.lock.Unlock()
	for key := range t.lastReleased {
		if !present.Has(key) && t.refs[key] == 0 {
			delete(t.lastReleased, key)
		}
	}
}
