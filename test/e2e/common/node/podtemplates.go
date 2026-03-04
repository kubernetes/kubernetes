/*
Copyright 2020 The Kubernetes Authors.

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

package node

import (
	"context"
	"encoding/json"

	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/retry"
	apimachineryutils "k8s.io/kubernetes/test/e2e/common/apimachinery"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	podTemplateRetryPeriod  = 1 * time.Second
	podTemplateRetryTimeout = 1 * time.Minute
)

var _ = SIGDescribe("PodTemplates", func() {
	f := framework.NewDefaultFramework("podtemplate")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	/*
	   Release: v1.19
	   Testname: PodTemplate lifecycle
	   Description: Attempt to create a PodTemplate. Patch the created PodTemplate. Fetching the PodTemplate MUST reflect changes.
	          By fetching all the PodTemplates via a Label selector it MUST find the PodTemplate by it's static label and updated value. The PodTemplate must be deleted.
	*/
	framework.ConformanceIt("should run the lifecycle of PodTemplates", func(ctx context.Context) {
		testNamespaceName := f.Namespace.Name
		podTemplateName := "nginx-pod-template-" + string(uuid.NewUUID())

		// get a list of PodTemplates (in all namespaces to hit endpoint)
		podTemplateList, err := f.ClientSet.CoreV1().PodTemplates("").List(ctx, metav1.ListOptions{
			LabelSelector: "podtemplate-static=true",
		})
		framework.ExpectNoError(err, "failed to list all PodTemplates")
		gomega.Expect(podTemplateList.Items).To(gomega.BeEmpty(), "unable to find templates")

		// create a PodTemplate
		_, err = f.ClientSet.CoreV1().PodTemplates(testNamespaceName).Create(ctx, &v1.PodTemplate{
			ObjectMeta: metav1.ObjectMeta{
				Name: podTemplateName,
				Labels: map[string]string{
					"podtemplate-static": "true",
				},
			},
			Template: v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "nginx", Image: imageutils.GetE2EImage(imageutils.Nginx)},
					},
				},
			},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create PodTemplate")

		// get template
		podTemplateRead, err := f.ClientSet.CoreV1().PodTemplates(testNamespaceName).Get(ctx, podTemplateName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get created PodTemplate")
		gomega.Expect(podTemplateRead.ObjectMeta.Name).To(gomega.Equal(podTemplateName))
		gomega.Expect(podTemplateRead).To(apimachineryutils.HaveValidResourceVersion())

		// patch template
		podTemplatePatch, err := json.Marshal(map[string]interface{}{
			"metadata": map[string]interface{}{
				"labels": map[string]string{
					"podtemplate": "patched",
				},
			},
		})
		framework.ExpectNoError(err, "failed to marshal patch data")
		patchedPodTemplate, err := f.ClientSet.CoreV1().PodTemplates(testNamespaceName).Patch(ctx, podTemplateName, types.StrategicMergePatchType, []byte(podTemplatePatch), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch PodTemplate")
		gomega.Expect(resourceversion.CompareResourceVersion(podTemplateRead.ResourceVersion, patchedPodTemplate.ResourceVersion)).To(gomega.BeNumerically("==", -1), "patched object should have a larger resource version")

		// get template (ensure label is there)
		podTemplateRead, err = f.ClientSet.CoreV1().PodTemplates(testNamespaceName).Get(ctx, podTemplateName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get PodTemplate")
		gomega.Expect(podTemplateRead.ObjectMeta.Labels).To(gomega.HaveKeyWithValue("podtemplate", "patched"), "failed to patch template, new label not found")

		// delete the PodTemplate
		err = f.ClientSet.CoreV1().PodTemplates(testNamespaceName).Delete(ctx, podTemplateName, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete PodTemplate")

		// list the PodTemplates
		podTemplateList, err = f.ClientSet.CoreV1().PodTemplates("").List(ctx, metav1.ListOptions{
			LabelSelector: "podtemplate-static=true",
		})
		framework.ExpectNoError(err, "failed to list PodTemplate")
		gomega.Expect(podTemplateList.Items).To(gomega.BeEmpty(), "PodTemplate list returned items, failed to delete PodTemplate")
	})

	/*
		Release: v1.19
		Testname: PodTemplate, delete a collection
		Description: A set of Pod Templates is created with a label selector which MUST be found when listed.
		The set of Pod Templates is deleted and MUST NOT show up when listed by its label selector.
	*/
	framework.ConformanceIt("should delete a collection of pod templates", func(ctx context.Context) {
		podTemplateNames := []string{"test-podtemplate-1", "test-podtemplate-2", "test-podtemplate-3"}

		ginkgo.By("Create set of pod templates")
		// create a set of pod templates in test namespace
		for _, podTemplateName := range podTemplateNames {
			_, err := f.ClientSet.CoreV1().PodTemplates(f.Namespace.Name).Create(ctx, &v1.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{
					Name:   podTemplateName,
					Labels: map[string]string{"podtemplate-set": "true"},
				},
				Template: v1.PodTemplateSpec{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{Name: "token-test", Image: imageutils.GetE2EImage(imageutils.Agnhost)},
						},
					},
				},
			}, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create pod template")
			framework.Logf("created %v", podTemplateName)
		}

		ginkgo.By("get a list of pod templates with a label in the current namespace")
		// get a list of pod templates
		podTemplateList, err := f.ClientSet.CoreV1().PodTemplates(f.Namespace.Name).List(ctx, metav1.ListOptions{
			LabelSelector: "podtemplate-set=true",
		})
		framework.ExpectNoError(err, "failed to get a list of pod templates")

		gomega.Expect(podTemplateList.Items).To(gomega.HaveLen(len(podTemplateNames)), "looking for expected number of pod templates")

		ginkgo.By("delete collection of pod templates")
		// delete collection

		framework.Logf("requesting DeleteCollection of pod templates")
		err = f.ClientSet.CoreV1().PodTemplates(f.Namespace.Name).DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{
			LabelSelector: "podtemplate-set=true"})
		framework.ExpectNoError(err, "failed to delete all pod templates")

		ginkgo.By("check that the list of pod templates matches the requested quantity")

		err = wait.PollImmediate(podTemplateRetryPeriod, podTemplateRetryTimeout, checkPodTemplateListQuantity(ctx, f, "podtemplate-set=true", 0))
		framework.ExpectNoError(err, "failed to count required pod templates")

	})

	/*
	   Release: v1.24
	   Testname: PodTemplate, replace
	   Description: Attempt to create a PodTemplate which MUST succeed.
	   Attempt to replace the PodTemplate to include a new annotation
	   which MUST succeed. The annotation MUST be found in the new PodTemplate.
	*/
	framework.ConformanceIt("should replace a pod template", func(ctx context.Context) {
		ptClient := f.ClientSet.CoreV1().PodTemplates(f.Namespace.Name)
		ptName := "podtemplate-" + utilrand.String(5)

		ginkgo.By("Create a pod template")
		ptResource, err := ptClient.Create(ctx, &v1.PodTemplate{
			ObjectMeta: metav1.ObjectMeta{
				Name: ptName,
			},
			Template: v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "e2e-test", Image: imageutils.GetE2EImage(imageutils.Agnhost)},
					},
				},
			},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod template")

		ginkgo.By("Replace a pod template")
		var updatedPT *v1.PodTemplate

		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			ptResource, err = ptClient.Get(ctx, ptName, metav1.GetOptions{})
			framework.ExpectNoError(err, "Unable to get pod template %s", ptName)
			ptResource.Annotations = map[string]string{
				"updated": "true",
			}
			updatedPT, err = ptClient.Update(ctx, ptResource, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err)
		gomega.Expect(updatedPT.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")
		framework.Logf("Found updated podtemplate annotation: %#v\n", updatedPT.Annotations["updated"])
	})

})

func checkPodTemplateListQuantity(ctx context.Context, f *framework.Framework, label string, quantity int) func() (bool, error) {
	return func() (bool, error) {
		var err error

		framework.Logf("requesting list of pod templates to confirm quantity")

		list, err := f.ClientSet.CoreV1().PodTemplates(f.Namespace.Name).List(ctx, metav1.ListOptions{
			LabelSelector: label})

		if err != nil {
			return false, err
		}

		if len(list.Items) != quantity {
			return false, err
		}
		return true, nil
	}
}
