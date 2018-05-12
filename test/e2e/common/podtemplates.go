/*
Copyright 2017 The Kubernetes Authors.

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
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("PodTemplates", func() {

	f := framework.NewDefaultFramework("podtemplates")

	BeforeEach(func() {

	})

	/*
		Testname: podtemplates
		Description: Make sure PodTemplates can be created,
		retrieved, and watched
	*/
	It("should create a PodTemplate", func() {
		zero := int64(0)
		name := "test-pod-podtemplate-" + string(uuid.NewUUID())

		By("Preparing a PodTemplate structure")
		podTemplate := &v1.PodTemplate{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": name},
				},
				Spec: v1.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers: []v1.Container{
						{
							Name:  name,
							Image: framework.ServeHostnameImage,
						},
					},
				},
			},
		}

		By("Setting up watch")
		w, err := f.ClientSet.CoreV1().PodTemplates(f.Namespace.Name).Watch(metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred(), "failed to set up watch")

		By("Creating a PodTemplate")
		f.ClientSet.CoreV1().PodTemplates(f.Namespace.Name).Create(podTemplate)
		Expect(err).NotTo(HaveOccurred())

		By("Trying to get a newly created PodTemplate")
		fetchedPodTemplate, err := f.ClientSet.CoreV1().PodTemplates(f.Namespace.Name).Get(name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "failed to query for podTemplate")
		Expect(fetchedPodTemplate.Name).To(Equal(podTemplate.Name))

		By("Verifying PodTemplate creation was observed")
		select {
		case event, _ := <-w.ResultChan():
			if event.Type != watch.Added {
				framework.Failf("Failed to observe podTemplate creation: %v", event)
			}
		case <-time.After(framework.ServiceRespondingTimeout):
			framework.Failf("Timeout while waiting for podTemplate creation")
		}
	})
})
