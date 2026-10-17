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

package windows

import (
	"context"
	"errors"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = sigDescribe(feature.Windows, feature.ContainerStopSignals, "Container Stop Signals", framework.WithFeatureGate(features.ContainerStopSignals), skipUnlessWindows(func() {
	f := framework.NewDefaultFramework("windows-container-stop-signals")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.DescribeTable("should validate stop signals for Windows pods",
		func(ctx context.Context, stopSignal v1.Signal, accepted bool) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "windows-container-stop-signal",
				},
				Spec: v1.PodSpec{
					OS: &v1.PodOS{Name: v1.Windows},
					Containers: []v1.Container{{
						Name:  "test-container",
						Image: imageutils.GetPauseImageName(),
						Lifecycle: &v1.Lifecycle{
							StopSignal: &stopSignal,
						},
					}},
					NodeSelector: map[string]string{
						v1.LabelOSStable: string(v1.Windows),
					},
				},
			}

			createdPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(
				ctx,
				pod,
				metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}},
			)
			if accepted {
				framework.ExpectNoError(err, "expected Windows pod with stop signal %q to pass API validation", stopSignal)
				gomega.Expect(createdPod.Spec.Containers[0].Lifecycle.StopSignal).To(
					gomega.HaveValue(gomega.Equal(stopSignal)),
					"API response should preserve the stop signal",
				)
				return
			}

			gomega.Expect(err).To(
				gomega.MatchError(apierrors.IsInvalid, "API validation should return an Invalid error"),
				"expected Windows pod with stop signal %q to fail API validation",
				stopSignal,
			)
			var statusError *apierrors.StatusError
			gomega.Expect(errors.As(err, &statusError)).To(
				gomega.BeTrueBecause("expected API validation to return a StatusError, got %T: %v", err, err),
			)
			gomega.Expect(statusError.ErrStatus.Details.Causes).To(
				gomega.ContainElement(gomega.HaveField("Field", "spec.containers[0].lifecycle.stopSignal")),
				"expected validation error for the container stop signal",
			)
		},
		ginkgo.Entry("accepts SIGTERM", v1.SIGTERM, true),
		ginkgo.Entry("accepts SIGKILL", v1.SIGKILL, true),
		ginkgo.Entry("rejects SIGQUIT", v1.SIGQUIT, false),
		ginkgo.Entry("rejects SIGUSR1", v1.SIGUSR1, false),
	)
}))
