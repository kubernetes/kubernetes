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

// OWNER = sig/cli

package kubectl

import (
	"context"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	admissionapi "k8s.io/pod-security-admission/api"

	commonutils "k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
)

var _ = SIGDescribe("Kubectl delete", func() {
	defer ginkgo.GinkgoRecover()
	var deploymentYaml string
	f := framework.NewDefaultFramework("kubectl-delete")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	deploymentName := "agnhost-deployment"

	var ns string
	var c clientset.Interface
	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		deploymentYaml = commonutils.SubstituteImageName(string(readTestFileOrDie(agnhostDeployment1Filename)))
	})

	ginkgo.Describe("interactive", func() {
		ginkgo.It("based on user confirmation input", func(ctx context.Context) {
			ginkgo.By("apply deployment with replicas 2")
			e2ekubectl.RunKubectlOrDieInput(ns, deploymentYaml, "apply", "-f", "-")

			ginkgo.By("verifying the deployment is created and running")
			d, err := c.AppsV1().Deployments(ns).Get(ctx, deploymentName, metav1.GetOptions{})
			if err != nil {
				framework.Failf("Failed getting deployment %v", err)
			}
			err = e2edeployment.WaitForDeploymentComplete(c, d)
			framework.ExpectNoError(err, "waiting for the deployment to complete")

			ginkgo.By("check that resource is not deleted when user types no")
			output := e2ekubectl.RunKubectlOrDieInput(ns, "n", "delete", "--interactive", "deployment", deploymentName)
			expectedOutput := "You are about to delete the following 1 resource(s):"
			if !strings.Contains(output, expectedOutput) ||
				!strings.Contains(output, "deployment.apps/agnhost-deployment") ||
				!strings.Contains(output, "deletion is cancelled") {
				framework.Failf("unexpected output %s", output)
			}

			ginkgo.By("verify that deployment is not deleted")
			d, err = c.AppsV1().Deployments(ns).Get(ctx, deploymentName, metav1.GetOptions{})
			if err != nil || d.DeletionTimestamp != nil {
				framework.Failf("Failed getting deployment that shouldn't be deleted %v", err)
			}

			if d == nil || d.Status.AvailableReplicas != 2 {
				framework.Failf("unexpected available replicas")
			}

			ginkgo.By("check that resource is deleted when user types yes")
			e2ekubectl.RunKubectlOrDieInput(ns, "y", "delete", "--interactive", "deployment", deploymentName)

			ginkgo.By("ensure that the deployment is deleted successfully")
			err = wait.PollUntilContextTimeout(ctx, 2*time.Second, 30*time.Second, true, func(ctx context.Context) (bool, error) {
				_, err = c.AppsV1().Deployments(ns).Get(ctx, deploymentName, metav1.GetOptions{})
				if err == nil {
					return false, nil
				}

				if apierrors.IsNotFound(err) {
					return true, nil
				}

				return false, err
			})
			framework.ExpectNoError(err, "waiting for the deployment that is deleted after getting confirmation by user")
		})
	})
})
