/*
Copyright 2019 The Kubernetes Authors.

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

package drain

import (
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/utils/pointer"
)

func TestSkipDeletedFilter(t *testing.T) {
	tCases := []struct {
		timeStampAgeSeconds             int
		skipWaitForDeleteTimeoutSeconds int
		expectedDelete                  bool
	}{
		{
			timeStampAgeSeconds:             0,
			skipWaitForDeleteTimeoutSeconds: 20,
			expectedDelete:                  true,
		},
		{
			timeStampAgeSeconds:             1,
			skipWaitForDeleteTimeoutSeconds: 20,
			expectedDelete:                  true,
		},
		{
			timeStampAgeSeconds:             100,
			skipWaitForDeleteTimeoutSeconds: 20,
			expectedDelete:                  false,
		},
	}
	for i, tc := range tCases {
		h := &Helper{
			SkipWaitForDeleteTimeoutSeconds: tc.skipWaitForDeleteTimeoutSeconds,
		}
		pod := corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod",
				Namespace: "default",
			},
		}

		if tc.timeStampAgeSeconds > 0 {
			dTime := &metav1.Time{Time: time.Now().Add(time.Duration(tc.timeStampAgeSeconds) * time.Second * -1)}
			pod.ObjectMeta.SetDeletionTimestamp(dTime)
		}

		podDeleteStatus := h.skipDeletedFilter(pod)
		if podDeleteStatus.Delete != tc.expectedDelete {
			t.Errorf("test %v: unexpected podDeleteStatus.delete; actual %v; expected %v", i, podDeleteStatus.Delete, tc.expectedDelete)
		}
	}
}

func TestDaemonSetFilter(t *testing.T) {
	tCases := []struct {
		ownerKind            string
		force                bool
		ignoreAllDaemonSets  bool
		podPhase             corev1.PodPhase
		expectedDelete       bool
		expectedReason       string
		expectedErrorMessage string
	}{
		{
			ownerKind:            "DaemonSet",
			ignoreAllDaemonSets:  true,
			force:                true,
			podPhase:             corev1.PodRunning,
			expectedDelete:       false,
			expectedReason:       PodDeleteStatusTypeWarning,
			expectedErrorMessage: daemonSetWarning,
		},
		{
			ownerKind:            "DaemonSet",
			ignoreAllDaemonSets:  false,
			force:                true,
			podPhase:             corev1.PodRunning,
			expectedDelete:       false,
			expectedReason:       PodDeleteStatusTypeError,
			expectedErrorMessage: daemonSetFatal,
		},
		{
			ownerKind:            "DaemonSet",
			ignoreAllDaemonSets:  false,
			force:                false,
			podPhase:             corev1.PodRunning,
			expectedDelete:       false,
			expectedReason:       PodDeleteStatusTypeError,
			expectedErrorMessage: daemonSetFatal,
		},
		{
			ownerKind:            "DaemonSet",
			ignoreAllDaemonSets:  true,
			force:                false,
			podPhase:             corev1.PodRunning,
			expectedDelete:       false,
			expectedReason:       PodDeleteStatusTypeWarning,
			expectedErrorMessage: daemonSetWarning,
		},
		{
			ownerKind:           "DaemonSet",
			ignoreAllDaemonSets: true,
			force:               false,
			podPhase:            corev1.PodFailed,
			expectedDelete:      true,
			expectedReason:      PodDeleteStatusTypeOkay,
		},
		{
			ownerKind:           "ReplicaSet",
			ignoreAllDaemonSets: false,
			force:               false,
			podPhase:            corev1.PodRunning,
			expectedDelete:      true,
			expectedReason:      PodDeleteStatusTypeOkay,
		},
		{
			ownerKind:           "ReplicaSet",
			ignoreAllDaemonSets: true,
			force:               false,
			podPhase:            corev1.PodRunning,
			expectedDelete:      true,
			expectedReason:      PodDeleteStatusTypeOkay,
		},
	}
	for i, tc := range tCases {
		pod := corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod",
				Namespace: "default",
				OwnerReferences: []metav1.OwnerReference{
					{
						APIVersion: "apps/v1",
						Kind:       tc.ownerKind,
						Controller: pointer.BoolPtr(true),
						Name:       "bar",
					},
				},
			},
			Status: corev1.PodStatus{
				Phase: tc.podPhase,
			},
		}

		h := &Helper{
			Force:               tc.force,
			IgnoreAllDaemonSets: tc.ignoreAllDaemonSets,
			Client: fake.NewSimpleClientset(&appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "default",
				},
			}),
		}

		podDeleteStatus := h.daemonSetFilter(pod)
		if podDeleteStatus.Delete != tc.expectedDelete {
			t.Errorf("test %v: unexpected podDeleteStatus.delete; actual %v; expected %v", i, podDeleteStatus.Delete, tc.expectedDelete)
		}
		if podDeleteStatus.Reason != tc.expectedReason {
			t.Errorf("test %v: unexpected podDeleteStatus.reason; actual %v; expected %v", i, podDeleteStatus.Reason, tc.expectedReason)
		}
		if podDeleteStatus.Message != tc.expectedErrorMessage {
			t.Errorf("test %v: unexpected podDeleteStatus.message; actual %v; expected %v", i, podDeleteStatus.Message, tc.expectedErrorMessage)
		}
	}
}
