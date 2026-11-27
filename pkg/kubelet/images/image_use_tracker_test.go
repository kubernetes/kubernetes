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
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestUseTrackerStartRemoval(t *testing.T) {
	testCases := []struct {
		name string
		// use runs before GC lists the containers, the func it returns runs after.
		use        func(tracker *UseTracker) (after func())
		wantRemove bool
	}{
		{
			name:       "never used",
			use:        func(*UseTracker) func() { return func() {} },
			wantRemove: true,
		},
		{
			name: "in use",
			use: func(tracker *UseTracker) func() {
				tracker.Acquire("image", "")
				return func() {}
			},
		},
		{
			name: "in use twice, released once",
			use: func(tracker *UseTracker) func() {
				tracker.Acquire("image", "")
				release := tracker.Acquire("image", "")
				return release
			},
		},
		{
			name: "released before the snapshot",
			use: func(tracker *UseTracker) func() {
				tracker.Acquire("image", "")()
				return func() {}
			},
			wantRemove: true,
		},
		{
			name: "released after the snapshot",
			use: func(tracker *UseTracker) func() {
				return tracker.Acquire("image", "")
			},
		},
		{
			name: "release called twice counts once",
			use: func(tracker *UseTracker) func() {
				tracker.Acquire("image", "")
				release := tracker.Acquire("image", "")
				release()
				release()
				return func() {}
			},
		},
		{
			name: "other image in use",
			use: func(tracker *UseTracker) func() {
				tracker.Acquire("other", "")
				return func() {}
			},
			wantRemove: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				tracker := NewUseTracker()
				after := tc.use(tracker)
				time.Sleep(time.Second)
				snapshot := time.Now()
				time.Sleep(time.Second)
				after()
				assert.Equal(t, tc.wantRemove, tracker.startRemoval("image", snapshot))
			})
		})
	}
}

func TestUseTrackerAcquireWaitsForRemoval(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tracker := NewUseTracker()
		before := tracker.Removals()
		if !tracker.startRemoval("image", time.Now()) {
			t.Fatal("startRemoval refused an unused image")
		}

		acquired := make(chan struct{})
		go func() {
			tracker.Acquire("image", "")
			close(acquired)
		}()
		otherAcquired := make(chan struct{})
		go func() {
			tracker.Acquire("other", "")
			close(otherAcquired)
		}()

		synctest.Wait()
		select {
		case <-acquired:
			t.Fatal("Acquire returned while the image was being removed")
		default:
		}
		<-otherAcquired

		tracker.finishRemoval("image")
		<-acquired
		assert.Equal(t, before+2, tracker.Removals(), "starting and finishing a removal should each change Removals")
		assert.False(t, tracker.startRemoval("image", time.Now()), "image acquired after its removal should be protected")
	})
}

func TestUseTrackerRuntimeHandler(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(map[bool]string{false: "gate disabled", true: "gate enabled"}[enabled], func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RuntimeClassInImageCriAPI, enabled)
			tracker := NewUseTracker()
			tracker.Acquire("image", "handler")
			// GC removes images by the same key as its image records.
			assert.False(t, tracker.startRemoval(useTrackerKey("image", "handler"), time.Now()))
			assert.Equal(t, enabled, tracker.startRemoval(useTrackerKey("image", ""), time.Now()),
				"an image pulled for another runtime handler is only a separate image with the gate enabled")
		})
	}
}

func TestUseTrackerForgetAbsent(t *testing.T) {
	tracker := NewUseTracker()
	tracker.Acquire("present", "")()
	tracker.Acquire("absent", "")()
	tracker.Acquire("absent-in-use", "")()
	tracker.Acquire("absent-in-use", "")
	tracker.forgetAbsent(sets.New("present"))

	assert.Contains(t, tracker.lastReleased, "present")
	assert.NotContains(t, tracker.lastReleased, "absent")
	assert.Contains(t, tracker.lastReleased, "absent-in-use")
}
