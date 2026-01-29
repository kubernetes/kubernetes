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

package images

import (
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	legacyregistry "k8s.io/component-base/metrics/legacyregistry"
	metricstestutil "k8s.io/component-base/metrics/testutil"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestEnsureImageRequestsMetrics(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
		}}

	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			Uid:       string(pod.UID),
		},
	}

	cases := pullerTestCases()

	useSerializedEnv := true
	for _, tt := range cases {
		t.Run(tt.testName, func(t *testing.T) {
			ctx := ktesting.Init(t)
			puller, _, _, container, _, _, _ := pullerTestEnv(t, tt, useSerializedEnv, nil)

			ensureImageRequestsCounter.Reset()
			_, _, _ = puller.EnsureImageExists(ctx, &v1.ObjectReference{}, pod, container.Image, tt.pullSecrets, podSandboxConfig, "", container.ImagePullPolicy)
			if err := metricstestutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.expectedEnsureImageMetrics), "kubelet_image_manager_ensure_image_requests_total"); err != nil {
				t.Fatal(err)
			}
		})
	}
}
