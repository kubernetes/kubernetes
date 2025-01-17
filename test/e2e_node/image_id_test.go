/*
Copyright 2016 The Kubernetes Authors.

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

package e2enode

import (
	"context"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/dump"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/nodefeature"

	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("ImageID", nodefeature.ImageID, feature.ImageID, func() {

	busyBoxImage := "registry.k8s.io/e2e-test-images/busybox@sha256:a9155b13325b2abef48e71de77bb8ac015412a566829f621d06bfae5c699b1b9"

	f := framework.NewDefaultFramework("image-id-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should be set to the manifest digest (from RepoDigests) when available", func(ctx context.Context) {
		podDesc := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-with-repodigest",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{{
					Name:    "test",
					Image:   busyBoxImage,
					Command: []string{"sh"},
				}},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		pod := e2epod.NewPodClient(f).Create(ctx, podDesc)

		framework.ExpectNoError(e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(ctx,
			f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout))
		runningPod, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		status := runningPod.Status
		gomega.Expect(status.ContainerStatuses).To(gomega.HaveLen(1), dump.Pretty(status))
		gomega.Expect(status.ContainerStatuses[0].ImageID).To(
			gomega.SatisfyAny(
				gomega.Equal(busyBoxImage),
				gomega.MatchRegexp(`[[:xdigit:]]{64}`),
			),
		)
	})
})
