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

package imperativeevictionresponder

import (
	"fmt"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/ktesting"
	testingclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

func TestEvict(t *testing.T) { testEvict(ktesting.Init(t)) }
func testEvict(tCtx ktesting.TContext) {
	fakeClock := testingclock.NewFakePassiveClock(time.Now())

	tests := []struct {
		name                 string
		pod                  *v1.Pod
		lastAttemptTime      *time.Time
		lastEvictionMessage  string
		evictionResponseFunc core.ReactionFunc

		expectEvictionMessage string
		expectLastAttemptTime *time.Time
		expectAddAfter        *time.Duration
		expectEvictionCall    bool
	}{
		{
			name: "eviction",
			pod:  mkValidPod(),

			expectEvictionMessage: "has been marked for deletion via the /eviction subresource",
			expectEvictionCall:    true,
		},
		{
			name:                "eviction_after_one_failure",
			pod:                 mkValidPod(),
			lastAttemptTime:     new(fakeClock.Now().Add(-5 * time.Second)),
			lastEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=1): server error",

			expectEvictionMessage: "has been marked for deletion via the /eviction subresource",
			expectEvictionCall:    true,
		},
		{
			name:                "eviction_after_two_failures",
			pod:                 mkValidPod(),
			lastAttemptTime:     new(fakeClock.Now().Add(-10 * time.Second)),
			lastEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=2): server error",

			expectEvictionMessage: "has been marked for deletion via the /eviction subresource",
			expectEvictionCall:    true,
		},
		{
			name:                "eviction_after_hundred_failures",
			pod:                 mkValidPod(),
			lastAttemptTime:     new(fakeClock.Now().Add(-10 * time.Minute)),
			lastEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=100): server error",

			expectEvictionMessage: "has been marked for deletion via the /eviction subresource",
			expectEvictionCall:    true,
		},
		{
			name:                "backoff_after_one_failure",
			pod:                 mkValidPod(),
			lastAttemptTime:     new(fakeClock.Now().Add(-4 * time.Second)),
			lastEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=1): server error",

			expectEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=1): server error",
			expectLastAttemptTime: new(fakeClock.Now().Add(-4 * time.Second)),
			expectAddAfter:        new(time.Second),
			expectEvictionCall:    false,
		},
		{
			name:                "invalid_message_falls_back_to_one_failure",
			pod:                 mkValidPod(),
			lastAttemptTime:     new(fakeClock.Now().Add(-4 * time.Second)),
			lastEvictionMessage: "pod deletion via the /eviction subresource failed (attemptz=5): server error",

			expectEvictionMessage: "pod deletion via the /eviction subresource failed (attemptz=5): server error",
			expectLastAttemptTime: new(fakeClock.Now().Add(-4 * time.Second)),
			expectAddAfter:        new(time.Second),
			expectEvictionCall:    false,
		},
		{
			name:                "backoff_after_two_failure",
			pod:                 mkValidPod(),
			lastAttemptTime:     new(fakeClock.Now().Add(-1 * time.Second)),
			lastEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=2): server error",

			expectEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=2): server error",
			expectLastAttemptTime: new(fakeClock.Now().Add(-1 * time.Second)),
			expectAddAfter:        new(9 * time.Second),
			expectEvictionCall:    false,
		},
		{
			name:                 "backoff_after_hundred_failures",
			pod:                  mkValidPod(),
			lastAttemptTime:      new(fakeClock.Now().Add(-9*time.Minute - 50*time.Second)),
			lastEvictionMessage:  "pod deletion via the /eviction subresource failed (attempts=100): server error",
			evictionResponseFunc: nil,

			expectEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=100): server error",
			expectLastAttemptTime: new(fakeClock.Now().Add(-9*time.Minute - 50*time.Second)),
			expectAddAfter:        new(10 * time.Second),
			expectEvictionCall:    false,
		},
		{
			name:                "resync_after_pod_not_found",
			pod:                 mkValidPod(),
			lastAttemptTime:     new(fakeClock.Now().Add(-5 * time.Second)),
			lastEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=1): server error",
			evictionResponseFunc: func(action core.Action) (bool, runtime.Object, error) {
				return true, nil, errors.NewNotFound(schema.GroupResource{Resource: "pods"}, "test-pod")
			},

			// We do not update the message as we expect to complete the Eviction status after the resync.
			expectEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=1): server error",
			expectLastAttemptTime: new(fakeClock.Now().Add(-5 * time.Second)),
			expectAddAfter:        new(time.Second),
			expectEvictionCall:    true,
		},
		{
			name: "eviction_failed",
			pod:  mkValidPod(),
			evictionResponseFunc: func(action core.Action) (bool, runtime.Object, error) {
				return true, nil, errors.NewInternalError(fmt.Errorf("server error"))
			},

			expectEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=1): Internal error occurred: server error",
			expectLastAttemptTime: new(fakeClock.Now()),
			expectAddAfter:        new(5 * time.Second),
			expectEvictionCall:    true,
		},
		{
			name:                "eviction_failed_twice",
			pod:                 mkValidPod(),
			lastAttemptTime:     new(fakeClock.Now().Add(-5 * time.Second)),
			lastEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=1): Internal error occurred: server error",
			evictionResponseFunc: func(action core.Action) (bool, runtime.Object, error) {
				return true, nil, errors.NewInternalError(fmt.Errorf("server error"))
			},

			expectEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=2): Internal error occurred: server error",
			expectLastAttemptTime: new(fakeClock.Now()),
			expectAddAfter:        new(10 * time.Second),
			expectEvictionCall:    true,
		},
		{
			name:                "eviction_failed_repeatedly",
			pod:                 mkValidPod(),
			lastAttemptTime:     new(fakeClock.Now().Add(-10 * time.Second)),
			lastEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=2): Internal error occurred: server error",
			evictionResponseFunc: func(action core.Action) (bool, runtime.Object, error) {
				return true, nil, errors.NewInternalError(fmt.Errorf("new error"))
			},

			expectEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=3): Internal error occurred: new error",
			expectLastAttemptTime: new(fakeClock.Now()),
			expectAddAfter:        new(20 * time.Second),
			expectEvictionCall:    true,
		},
		{
			name:                "eviction_failed_repeatedly_with_server_backoff_override",
			pod:                 mkValidPod(),
			lastAttemptTime:     new(fakeClock.Now().Add(-5 * time.Second)),
			lastEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=1): Internal error occurred: server error",
			evictionResponseFunc: func(action core.Action) (bool, runtime.Object, error) {
				return true, nil, errors.NewTooManyRequests("too many requests", 13)
			},

			expectEvictionMessage: "pod deletion via the /eviction subresource failed (attempts=2): too many requests",
			expectLastAttemptTime: new(fakeClock.Now()),
			expectAddAfter:        new(13 * time.Second),
			expectEvictionCall:    true,
		},
	}

	for _, tc := range tests {
		tCtx.Run(tc.name, func(tCtx ktesting.TContext) {
			fakeClient := fake.NewSimpleClientset()
			evictionCalled := false
			fakeClient.PrependReactor("create", "pods", func(action core.Action) (bool, runtime.Object, error) {
				if action.GetSubresource() == "eviction" {
					evictionCalled = true
					if tc.evictionResponseFunc != nil {
						return tc.evictionResponseFunc(action)
					}
					return true, nil, nil
				}
				return false, nil, nil
			})

			c := &ImperativeEvictionResponderController{
				controllerName:                     "imperative-eviction-responder-controller",
				kubeClient:                         fakeClient,
				heartbeatMinDurationBetweenUpdates: 5 * time.Second,
				heartbeatMaxDurationBetweenUpdates: 3 * time.Minute,
				maxImperativeEvictionBackoff:       10 * time.Minute,
				lastEvictionAttempts:               NewLastEvictionAttempts(),
				clock:                              fakeClock,
			}

			if tc.lastAttemptTime != nil {
				c.lastEvictionAttempts.set(evictionIndexValueForPod(tc.pod), *tc.lastAttemptTime)
			}
			evictionMessage, lastAttemptTime, addAfter := c.evict(tCtx, tc.pod, tc.lastEvictionMessage)

			if !strings.Contains(evictionMessage, tc.expectEvictionMessage) {
				tCtx.Errorf("expected %v, got %v", evictionMessage, tc.expectEvictionMessage)
			}
			if want, got := tc.expectLastAttemptTime, lastAttemptTime; !ptr.Equal(want, got) {
				tCtx.Errorf("expected %v, got %v", want, got)
			}
			if want, got := tc.expectAddAfter, addAfter; !ptr.Equal(want, got) {
				tCtx.Errorf("expected %v, got %v", want, got)
			}
			if want, got := tc.expectEvictionCall, evictionCalled; want != got {
				tCtx.Errorf("expected %v, got %v", want, got)
			}
		})
	}
}

func mkValidPod(tweaks ...func(obj *v1.Pod)) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
			UID:       "6f1d9e45-cd7e-4706-bf8c-0d77dc8dd9a8",
		},
	}
	for _, tweak := range tweaks {
		tweak(pod)
	}
	return pod
}
