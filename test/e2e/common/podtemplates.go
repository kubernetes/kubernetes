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

package common

import (
	"context"
	"encoding/json"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	"strconv"

	"github.com/onsi/ginkgo"
)

var _ = ginkgo.Describe("[sig-node] PodTemplates", func() {
	f := framework.NewDefaultFramework("podtemplate")
	/*
	   Release : v1.19
	   Testname: PodTemplate lifecycle
	   Description: Attempt to create a PodTemplate. Patch the created PodTemplate. Fetching the PodTemplate MUST reflect changes.
	          By fetching all the PodTemplates via a Label selector it MUST find the PodTemplate by it's static label and updated value. The PodTemplate must be deleted.
	*/
	framework.ConformanceIt("should run the lifecycle of PodTemplates", func() {
		testNamespaceName := f.Namespace.Name
		podTemplateName := "nginx-pod-template-" + string(uuid.NewUUID())

		// get a list of PodTemplates (in all namespaces to hit endpoint)
		podTemplateList, err := f.ClientSet.CoreV1().PodTemplates("").List(context.TODO(), metav1.ListOptions{
			LabelSelector: "podtemplate-static=true",
		})
		framework.ExpectNoError(err, "failed to list all PodTemplates")
		framework.ExpectEqual(len(podTemplateList.Items), 0, "unable to find templates")

		// create a PodTemplate
		_, err = f.ClientSet.CoreV1().PodTemplates(testNamespaceName).Create(context.TODO(), &v1.PodTemplate{
			ObjectMeta: metav1.ObjectMeta{
				Name: podTemplateName,
				Labels: map[string]string{
					"podtemplate-static": "true",
				},
			},
			Template: v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "nginx", Image: "nginx"},
					},
				},
			},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create PodTemplate")

		// get template
		podTemplateRead, err := f.ClientSet.CoreV1().PodTemplates(testNamespaceName).Get(context.TODO(), podTemplateName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get created PodTemplate")
		framework.ExpectEqual(podTemplateRead.ObjectMeta.Name, podTemplateName)

		// patch template
		podTemplatePatch, err := json.Marshal(map[string]interface{}{
			"metadata": map[string]interface{}{
				"labels": map[string]string{
					"podtemplate": "patched",
				},
			},
		})
		framework.ExpectNoError(err, "failed to marshal patch data")
		_, err = f.ClientSet.CoreV1().PodTemplates(testNamespaceName).Patch(context.TODO(), podTemplateName, types.StrategicMergePatchType, []byte(podTemplatePatch), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch PodTemplate")

		// get template (ensure label is there)
		podTemplateRead, err = f.ClientSet.CoreV1().PodTemplates(testNamespaceName).Get(context.TODO(), podTemplateName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get PodTemplate")
		framework.ExpectEqual(podTemplateRead.ObjectMeta.Labels["podtemplate"], "patched", "failed to patch template, new label not found")

		// delete the PodTemplate
		err = f.ClientSet.CoreV1().PodTemplates(testNamespaceName).Delete(context.TODO(), podTemplateName, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete PodTemplate")

		// list the PodTemplates
		podTemplateList, err = f.ClientSet.CoreV1().PodTemplates("").List(context.TODO(), metav1.ListOptions{
			LabelSelector: "podtemplate-static=true",
		})
		framework.ExpectNoError(err, "failed to list PodTemplate")
		framework.ExpectEqual(len(podTemplateList.Items), 0, "PodTemplate list returned items, failed to delete PodTemplate")
	})

	ginkgo.It("should delete a collection of pod templates", func() {
		podTemplateNames := []string{"test-podtemplate-1", "test-podtemplate-2", "test-podtemplate-3"}
		podTemplateNamesCount := len(podTemplateNames)

		ginkgo.By("Create set of pod templates")
		// create a set of pod templates in test namespace
		for _, podTemplateName := range podTemplateNames {
			_, err := f.ClientSet.CoreV1().PodTemplates(f.Namespace.Name).Create(context.TODO(), &v1.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{
					Name:   podTemplateName,
					Labels: map[string]string{"podtemplate-set": "true"},
				},
				Template: v1.PodTemplateSpec{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{Name: "nginx", Image: "nginx"},
						},
					},
				},
			}, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create pod template")
			framework.Logf("created %v", podTemplateName)
		}

		ginkgo.By("get a list of pod templates with a label in the current namespace")
		// get a list of pod templates
		podTemplateList, err := f.ClientSet.CoreV1().PodTemplates(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{
			LabelSelector: "podtemplate-set=true",
		})
		framework.ExpectNoError(err, "failed to get a list of pod templates")

		// check that we find all the pod templates created
		podTemplateListItemsCount := len(podTemplateList.Items)
		errMsg := "found " + strconv.Itoa(podTemplateListItemsCount) + " pod templates when " + strconv.Itoa(podTemplateNamesCount) + " pod templates where expected"
		logMsg := "found " + strconv.Itoa(podTemplateListItemsCount) + " pod templates"

		foundCreatedSet := true
		if podTemplateListItemsCount != podTemplateNamesCount {
			foundCreatedSet = false
		} else {
			framework.Logf(logMsg)
		}
		framework.ExpectEqual(foundCreatedSet, true, errMsg)

		ginkgo.By("delete collection of pod templates")
		// delete collection

		framework.Logf("requesting DeleteCollection of pod templates")
		err = f.ClientSet.CoreV1().PodTemplates(f.Namespace.Name).DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{
			LabelSelector: "podtemplate-set=true"})
		framework.ExpectNoError(err, "failed to delete all pod templates")

		ginkgo.By("get a list of pod templates with a label in the current namespace")
		// get list of pod templates
		podTemplateList, err = f.ClientSet.CoreV1().PodTemplates(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{
			LabelSelector: "podtemplate-set=true",
		})
		framework.ExpectNoError(err, "failed to get a list of pod templates")

		// check that we don't find any created pod templates
		podTemplateListItemsCount = len(podTemplateList.Items)
		errMsg = "found " + strconv.Itoa(podTemplateListItemsCount) + " pod templates when zero pod templates where expected"
		logMsg = "found " + strconv.Itoa(podTemplateListItemsCount) + " pod templates"

		foundCreatedSet = false
		if podTemplateListItemsCount != 0 {
			foundCreatedSet = true
		} else {
			framework.Logf(logMsg)
		}
		framework.ExpectEqual(foundCreatedSet, false, errMsg)

	})

})
