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
	"math"
	"testing"
	"time"

	"k8s.io/api/lifecycle/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apimachinerytypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
)

func TestExponentialBackoff(t *testing.T) {
	tests := []struct {
		name     string
		exponent uint64
		want     time.Duration
	}{
		{
			name:     "calculates backoff",
			exponent: 1,
			want:     10 * time.Second,
		},
		{
			name:     "calculates backoff with a maximum",
			exponent: 10,
			want:     time.Minute,
		},
		{
			name:     "calculates backoff with a maximum after an overflow",
			exponent: 1000,
			want:     time.Minute,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			backoff := exponentialBackoff(5*time.Second, time.Minute, tc.exponent)
			if backoff != tc.want {
				t.Errorf("expected %v, got %v", tc.want, backoff)
			}
		})
	}
}

func TestShouldHandleEviction(t *testing.T) {
	tests := []struct {
		name     string
		eviction *v1alpha1.Eviction
		want     bool
	}{
		{
			name: "correct target",
			eviction: &v1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						string(v1alpha1.EvictionResponderImperativeEviction): string(v1alpha1.EvictionParticipantRoleResponder),
					},
				},
				Spec: v1alpha1.EvictionSpec{
					Target: v1alpha1.EvictionTarget{
						Pod: &v1alpha1.EvictionPodReference{
							Name: "pod",
							UID:  "04c63289-90de-470b-8ce1-fa50d962ea10",
						},
					},
				},
			},
			want: true,
		},
		{
			name: "no target",
			eviction: &v1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						string(v1alpha1.EvictionResponderImperativeEviction): string(v1alpha1.EvictionParticipantRoleResponder),
					},
				},
			},
			want: false,
		},
		{
			name: "wrong target",
			eviction: &v1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						string(v1alpha1.EvictionResponderImperativeEviction): string(v1alpha1.EvictionParticipantRoleResponder),
					},
				},
				Spec: v1alpha1.EvictionSpec{
					Target: v1alpha1.EvictionTarget{},
				},
			},
			want: false,
		},
		{
			name: "missing participant label",
			eviction: &v1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						string(v1alpha1.EvictionResponderImperativeEviction): "",
					},
				},
				Spec: v1alpha1.EvictionSpec{
					Target: v1alpha1.EvictionTarget{
						Pod: &v1alpha1.EvictionPodReference{
							Name: "pod",
							UID:  "04c63289-90de-470b-8ce1-fa50d962ea10",
						},
					},
				},
			},
			want: false,
		},
		{
			name: "wrong participant label",
			eviction: &v1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						string(v1alpha1.EvictionResponderImperativeEviction): string(v1alpha1.EvictionParticipantRoleRequester),
					},
				},
				Spec: v1alpha1.EvictionSpec{
					Target: v1alpha1.EvictionTarget{
						Pod: &v1alpha1.EvictionPodReference{
							Name: "pod",
							UID:  "04c63289-90de-470b-8ce1-fa50d962ea10",
						},
					},
				},
			},
			want: false,
		},
		{
			name: "invalid participant label",
			eviction: &v1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						string(v1alpha1.EvictionResponderImperativeEviction): "invalid",
					},
				},
				Spec: v1alpha1.EvictionSpec{
					Target: v1alpha1.EvictionTarget{
						Pod: &v1alpha1.EvictionPodReference{
							Name: "pod",
							UID:  "04c63289-90de-470b-8ce1-fa50d962ea10",
						},
					},
				},
			},
			want: false,
		},

		{
			name: "no participant label",
			eviction: &v1alpha1.Eviction{
				Spec: v1alpha1.EvictionSpec{
					Target: v1alpha1.EvictionTarget{
						Pod: &v1alpha1.EvictionPodReference{
							Name: "pod",
							UID:  "04c63289-90de-470b-8ce1-fa50d962ea10",
						},
					},
				},
			},
			want: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			shouldHandle := shouldHandleEviction(tc.eviction)
			if shouldHandle != tc.want {
				t.Errorf("expected %v, got %v", tc.want, shouldHandle)
			}
		})
	}
}

func TestHasLabelChanged(t *testing.T) {
	tests := []struct {
		name        string
		oldEviction *v1alpha1.Eviction
		eviction    *v1alpha1.Eviction
		want        bool
	}{
		{
			name:        "no label",
			oldEviction: &v1alpha1.Eviction{},
			eviction:    &v1alpha1.Eviction{},
			want:        false,
		},
		{
			name: "label stays the same",
			oldEviction: &v1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						string(v1alpha1.EvictionResponderImperativeEviction): string(v1alpha1.EvictionParticipantRoleResponder),
						"foo": "bar",
					},
				},
			},
			eviction: &v1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						string(v1alpha1.EvictionResponderImperativeEviction): string(v1alpha1.EvictionParticipantRoleResponder),
						"baz": "bar",
					},
				},
			},
			want: false,
		},
		{
			name:        "label introduced",
			oldEviction: &v1alpha1.Eviction{},
			eviction: &v1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						string(v1alpha1.EvictionResponderImperativeEviction): string(v1alpha1.EvictionParticipantRoleResponder),
					},
				},
			},
			want: true,
		},
		{
			name: "label changed",
			oldEviction: &v1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						string(v1alpha1.EvictionResponderImperativeEviction): string(v1alpha1.EvictionParticipantRoleResponder),
						"foo": "bar",
					},
				},
			},
			eviction: &v1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						string(v1alpha1.EvictionResponderImperativeEviction): string(v1alpha1.EvictionParticipantRoleRequester),
					},
				},
			},
			want: true,
		},
		{
			name: "label value removed",
			oldEviction: &v1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						string(v1alpha1.EvictionResponderImperativeEviction): string(v1alpha1.EvictionParticipantRoleResponder),
						"foo": "bar",
					},
				},
			},
			eviction: &v1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						string(v1alpha1.EvictionResponderImperativeEviction): "",
					},
				},
			},
			want: true,
		},
		{
			name: "label removed",
			oldEviction: &v1alpha1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						string(v1alpha1.EvictionResponderImperativeEviction): string(v1alpha1.EvictionParticipantRoleResponder),
						"foo": "bar",
					},
				},
			},
			eviction: &v1alpha1.Eviction{},
			want:     true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			shouldHandle := hasLabelChanged(tc.oldEviction, tc.eviction, string(v1alpha1.EvictionResponderImperativeEviction))
			if shouldHandle != tc.want {
				t.Errorf("expected %v, got %v", tc.want, shouldHandle)
			}
		})
	}
}

func TestGetRecordedAttempts(t *testing.T) {
	tests := []struct {
		name    string
		message string
		want    uint64
	}{
		{
			name:    "missing message",
			message: "",
			want:    0,
		},
		{
			name:    "missing attempts",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion",
			want:    0,
		},
		{
			name:    "invalid attempts",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion via the /eviction subresource failed (attempts=bar): pods \"foo\" is forbidden:",
			want:    0,
		},
		{
			name:    "negative number attempts",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion via the /eviction subresource failed (attempts=-7): pods \"foo\" is forbidden:",
			want:    0,
		},
		{
			name:    "found 0",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion via the /eviction subresource failed (attempts=0): pods \"foo\" is forbidden:",
			want:    0,
		},
		{
			name:    "found 1",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion via the /eviction subresource failed (attempts=1): pods \"foo\" is forbidden:",
			want:    1,
		},
		{
			name:    "found 7 with preceding zeros",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion via the /eviction subresource failed (attempts=007): pods \"foo\" is forbidden:",
			want:    7,
		},
		{
			name:    "found MaxUint64",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion via the /eviction subresource failed (attempts=18446744073709551615): pods \"foo\" is forbidden:",
			want:    math.MaxUint64,
		},
		{
			name:    "MaxUint64 overflow",
			message: "foo pod (uid=\"492fe0ed-e8ee-4ad4-900f-591e35d6d352\") deletion via the /eviction subresource failed (attempts=18446744073709551616): pods \"foo\" is forbidden:",
			want:    0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			attempts := getRecordedAttempts(tc.message)
			if attempts != tc.want {
				t.Errorf("expected %d, got %d", tc.want, attempts)
			}
		})
	}
}

func TestToResponderStatusApplyConfiguration(t *testing.T) {
	now := metav1.Now()
	responderStatus := v1alpha1.ResponderStatus{
		Name:           "foo",
		StartTime:      &now,
		HeartbeatTime:  &now,
		CompletionTime: &now,
	}
	applyConfiguration := toResponderStatusApplyConfiguration(responderStatus)
	if want, got := "foo", ptr.Deref(applyConfiguration.Name, ""); want != got {
		t.Errorf("expected %v, got %v", want, got)
	}
	if want, got := now, ptr.Deref(applyConfiguration.StartTime, metav1.Time{}); want != got {
		t.Errorf("expected %v, got %v", want, got)
	}
	if want, got := now, ptr.Deref(applyConfiguration.HeartbeatTime, metav1.Time{}); want != got {
		t.Errorf("expected %v, got %v", want, got)
	}
	if applyConfiguration.ExpectedCompletionTime != nil {
		t.Errorf("expected nil, got %v", *applyConfiguration.ExpectedCompletionTime)
	}
	if want, got := now, ptr.Deref(applyConfiguration.CompletionTime, metav1.Time{}); want != got {
		t.Errorf("expected %v, got %v", want, got)
	}
	if *applyConfiguration.Message != "" {
		t.Errorf("expected \"\", got %v", *applyConfiguration.Message)
	}

	responderStatus.ExpectedCompletionTime = &now
	responderStatus.Message = new("message")
	applyConfiguration = toResponderStatusApplyConfiguration(responderStatus)
	if want, got := now, ptr.Deref(applyConfiguration.ExpectedCompletionTime, metav1.Time{}); want != got {
		t.Errorf("expected %v, got %v", want, got)
	}
	if want, got := "message", ptr.Deref(applyConfiguration.Message, ""); want != got {
		t.Errorf("expected %v, got %v", want, got)
	}
}

func TestLastEvictionAttempts(t *testing.T) {
	now := time.Now()
	attemptsTracker := NewLastEvictionAttempts()
	testUID := apimachinerytypes.UID("c10bd213-64bf-4cfd-9b5a-2541fbbd5b02")

	attemptsTracker.set(testUID, now)
	attemptsTracker.set("7d5c4d97-68bd-4a1c-8695-6ea49962caa7", now)
	attemptsTracker.set("7c4d5933-f269-4846-9e78-f56640d0ba6e", time.Now())

	got, ok := attemptsTracker.get(testUID)
	if !ok {
		t.Errorf("expected ok, got %v", ok)
	}
	if got != now {
		t.Errorf("expected %v, got %v", now, got)
	}

	attemptsTracker.remove(testUID)
	got, ok = attemptsTracker.get(testUID)
	if ok {
		t.Errorf("expected nok, got %v", ok)
	}
	if !got.IsZero() {
		t.Errorf("expected zero time, got %v", got)
	}
}
