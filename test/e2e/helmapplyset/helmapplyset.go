/*
Copyright 2024 The Kubernetes Authors.

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

package helmapplyset

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
)

var _ = ginkgo.Describe("HelmApplySet Status Aggregation", func() {
	f := framework.NewDefaultFramework("helmapplyset-status")

	ginkgo.It("should aggregate status for managed resources", func(ctx context.Context) {
		// This test assumes the HelmApplySet controller is running in the cluster
		// checking the feature gate or availability would be good practice

		ns := f.Namespace.Name
		releaseName := "e2e-status-test"
		applySetID := "applyset-e2e-test-v1"

		// 1. Create parent Secret (simulating what Feature 1 would do)
		// In a full E2E, we might install a real Helm chart, but here we simulate the artifacts
		parentSecret := &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "applyset-" + releaseName,
				Namespace: ns,
				Labels: map[string]string{
					"applyset.kubernetes.io/id": applySetID,
				},
				Annotations: map[string]string{
					"applyset.kubernetes.io/contains-group-kinds": "Deployment.apps,Service",
				},
			},
			Type: v1.SecretTypeOpaque,
		}
		_, err := f.ClientSet.CoreV1().Secrets(ns).Create(ctx, parentSecret, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create parent secret")

		// 2. Create managed resources with ApplySet label
		deployment := e2edeployment.NewDeployment("test-dep", 1, map[string]string{"app": "test"}, "nginx", "nginx:latest", appsv1.RollingUpdateDeploymentStrategyType)
		deployment.ObjectMeta.Labels["applyset.kubernetes.io/part-of"] = applySetID
		
		_, err = f.ClientSet.AppsV1().Deployments(ns).Create(ctx, deployment, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create deployment")

		// Wait for deployment to be ready
		err = e2edeployment.WaitForDeploymentComplete(f.ClientSet, deployment)
		framework.ExpectNoError(err, "deployment failed to become ready")

		// 3. Verify Conditions on Parent Secret
		// The controller should update the parent secret with conditions
		gomega.Eventually(func() string {
			s, err := f.ClientSet.CoreV1().Secrets(ns).Get(ctx, parentSecret.Name, metav1.GetOptions{})
			if err != nil {
				return ""
			}
			// Check for Ready condition
			cond, ok := s.Annotations["status.conditions.ready"]
			if !ok {
				return "Missing Ready condition"
			}
			return cond
		}, 2*time.Minute, 5*time.Second).Should(gomega.ContainSubstring(`"status":"True"`), "Parent Secret should have Ready=True condition")

		// 4. Degrade the Deployment (scale up but don't wait, or use image that fails)
		// Let's use an invalid image to cause a failure
		ginkgo.By("Updating deployment with invalid image to cause degradation")
		deployment, err = f.ClientSet.AppsV1().Deployments(ns).Get(ctx, "test-dep", metav1.GetOptions{})
		framework.ExpectNoError(err)
		
		deployment.Spec.Template.Spec.Containers[0].Image = "nginx:invalid-tag-should-fail"
		_, err = f.ClientSet.AppsV1().Deployments(ns).Update(ctx, deployment, metav1.UpdateOptions{})
		framework.ExpectNoError(err)

		// 5. Verify Status Changes to Degraded/Progressing
		gomega.Eventually(func() string {
			s, err := f.ClientSet.CoreV1().Secrets(ns).Get(ctx, parentSecret.Name, metav1.GetOptions{})
			if err != nil {
				return ""
			}
			// Should eventually be degraded or progressing (depending on exact timing of pod failure)
			// We check for Ready=False
			cond, ok := s.Annotations["status.conditions.ready"]
			if !ok {
				return "Missing Ready condition"
			}
			return cond
		}, 5*time.Minute, 5*time.Second).Should(gomega.ContainSubstring(`"status":"False"`), "Parent Secret should have Ready=False condition")
	})
})

