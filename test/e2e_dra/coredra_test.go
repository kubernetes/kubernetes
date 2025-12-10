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

package e2edra

import (
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func coreDRA(tCtx ktesting.TContext, f *framework.Framework, b *drautils.Builder) step2Func {
	namespace := f.Namespace.Name
	claim := b.ExternalClaim()
	pod := b.PodExternal()
	b.Create(tCtx, claim, pod)
	b.TestPod(tCtx, pod)

	return func(tCtx ktesting.TContext) step3Func {
		// Remove pod prepared in step 1.
		framework.ExpectNoError(f.ClientSet.ResourceV1beta1().ResourceClaims(namespace).Delete(tCtx, claim.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(f.ClientSet.CoreV1().Pods(namespace).Delete(tCtx, pod.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(tCtx, f.ClientSet, pod.Name, namespace, f.Timeouts.PodDelete))

		// Create another claim and pod, this time using the latest Kubernetes.
		claim = b.ExternalClaim()
		pod = b.PodExternal()
		pod.Spec.ResourceClaims[0].ResourceClaimName = &claim.Name
		b.Create(tCtx, claim, pod)
		b.TestPod(tCtx, pod)

		return func(tCtx ktesting.TContext) {
			// We need to clean up explicitly because the normal
			// cleanup doesn't work (driver shuts down first).
			//
			// The retry loops are necessary because of a stale connection
			// to the restarted apiserver. Sometimes, attempts fail with "EOF" as error
			// or (even weirder) with
			//     getting *v1.Pod: pods "tester-2" is forbidden: User "kubernetes-admin" cannot get resource "pods" in API group "" in the namespace "dra-9021"
			ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) error {
				return f.ClientSet.ResourceV1beta1().ResourceClaims(namespace).Delete(tCtx, claim.Name, metav1.DeleteOptions{})
			}).Should(gomega.Succeed(), "delete claim after downgrade")
			ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) error {
				return f.ClientSet.CoreV1().Pods(namespace).Delete(tCtx, pod.Name, metav1.DeleteOptions{})
			}).Should(gomega.Succeed(), "delete pod after downgrade")
			ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *v1.Pod {
				pod, err := f.ClientSet.CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
				if apierrors.IsNotFound(err) {
					return nil
				}
				tCtx.ExpectNoError(err, "get pod")
				return pod
			}).Should(gomega.BeNil(), "no pod after deletion after downgrade")
		}
	}
}
