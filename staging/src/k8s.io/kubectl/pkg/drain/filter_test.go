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

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
