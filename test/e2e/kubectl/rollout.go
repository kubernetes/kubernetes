/*
Copyright 2023 The Kubernetes Authors.

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

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/onsi/ginkgo/v2"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	commonutils "k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Kubectl rollout", func() {
	defer ginkgo.GinkgoRecover()
	var deploymentYaml string
	f := framework.NewDefaultFramework("kubectl-rollout")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var c clientset.Interface
	var ns string
	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		deploymentYaml = commonutils.SubstituteImageName(string(readTestFileOrDie(httpdDeployment1Filename)))
	})

	ginkgo.Describe("undo", func() {
		ginkgo.AfterEach(func() {
			cleanupKubectlInputs(deploymentYaml, ns, "app=httpd")
		})
		ginkgo.It("undo should rollback and update deployment env", func(ctx context.Context) {
			var err error
			// create deployment
			e2ekubectl.RunKubectlOrDieInput(ns, deploymentYaml, "apply", "-f", "-")

			if err = e2edeployment.WaitForDeploymentRevisionAndImage(c, ns, "httpd-deployment", "1", imageutils.GetE2EImage(imageutils.HttpdNew)); err != nil {
				framework.Failf("created deployment not ready")
			}

			var d *appsv1.Deployment
			if d, err = c.AppsV1().Deployments(ns).Get(ctx, "httpd-deployment", metav1.GetOptions{}); err != nil {
				framework.Failf("get deployment failed")
			}

			origEnv := d.Spec.Template.Spec.Containers[0].Env
			for _, env := range origEnv {
				if env.Name == "foo" && env.Value == "bar" {
					framework.Failf("labeled deployment should not have an env named foo and valued bar at the beginning")
				}
			}

			origLabels := d.Spec.Template.Labels
			if len(origLabels) == 0 {
				framework.Failf("original labels should not be empty in kubectl rollout test")
			}

			origAnnotations := d.Spec.Template.Annotations
			if len(origAnnotations) == 0 {
				framework.Failf("original annotations should not be empty in kubectl rollout test")
			}

			// do a small update
			if _, err = e2ekubectl.RunKubectl(ns, "set", "env", "deployment/httpd-deployment", "foo=bar"); err != nil {
				framework.Failf("kubectl failed set env for deployment")
			}
			// wait for env to be set
			if err = e2edeployment.WaitForDeploymentComplete(c, d); err != nil {
				framework.Failf("update deployment failed")
			}
			if d, err = c.AppsV1().Deployments(ns).Get(ctx, "httpd-deployment", metav1.GetOptions{}); err != nil {
				framework.Failf("get deployment failed")
			}
			envs := d.Spec.Template.Spec.Containers[0].Env

			envUpdated := false
			for _, env := range envs {
				if env.Name == "foo" && env.Value == "bar" {
					envUpdated = true
					break
				}
			}
			if !envUpdated {
				framework.Failf("update deployment's env failed")
			}

			// rollback
			if _, err = e2ekubectl.RunKubectl(ns, "rollout", "undo", "deployment/httpd-deployment"); err != nil {
				framework.Failf("kubectl failed to rollback deployment")
			}
			// wait for rollback finished
			if err = e2edeployment.WaitForDeploymentComplete(c, d); err != nil {
				framework.Failf("rollback deployment failed")
			}
			if d, err = c.AppsV1().Deployments(ns).Get(ctx, "httpd-deployment", metav1.GetOptions{}); err != nil {
				framework.Failf("get deployment failed")
			}

			rollbackedEnv := d.Spec.Template.Spec.Containers[0].Env
			rollbackedLabels := d.Spec.Template.Labels
			rollbackedAnnotations := d.Spec.Template.Annotations

			if diff := cmp.Diff(origEnv, rollbackedEnv, cmpopts.SortSlices(func(a, b v1.EnvVar) bool {
				return a.Name < b.Name
			})); diff != "" {
				framework.Failf("inconsistent env after rolled back: %s", diff)
			}
			if diff := cmp.Diff(origLabels, rollbackedLabels); diff != "" {
				framework.Failf("inconsistent labels after rolled back: %s", diff)
			}
			if diff := cmp.Diff(origAnnotations, rollbackedAnnotations); diff != "" {
				framework.Failf("inconsistent annotations after rolled back: %s", diff)
			}
		})
	})
})
