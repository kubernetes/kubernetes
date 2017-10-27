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

package apimachinery

import (
	"fmt"
	"math/rand"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/test/e2e/framework"
)

const numberOfTotalResources = 400

var _ = SIGDescribe("Servers with support for API chunking", func() {
	f := framework.NewDefaultFramework("chunking")

	It("should return chunks of results for list calls", func() {
		ns := f.Namespace.Name
		c := f.ClientSet
		client := c.CoreV1().PodTemplates(ns)

		By("creating a large number of resources")
		workqueue.Parallelize(20, numberOfTotalResources, func(i int) {
			for tries := 3; tries >= 0; tries-- {
				_, err := client.Create(&v1.PodTemplate{
					ObjectMeta: metav1.ObjectMeta{
						Name: fmt.Sprintf("template-%04d", i),
					},
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{Name: "test", Image: "test2"},
							},
						},
					},
				})
				if err == nil {
					return
				}
				framework.Logf("Got an error creating template %d: %v", i, err)
			}
			Fail("Unable to create template %d, exiting", i)
		})

		By("retrieving those results in paged fashion several times")
		for i := 0; i < 3; i++ {
			opts := metav1.ListOptions{}
			found := 0
			var lastRV string
			for {
				opts.Limit = int64(rand.Int31n(numberOfTotalResources/10) + 1)
				list, err := client.List(opts)
				Expect(err).ToNot(HaveOccurred())
				framework.Logf("Retrieved %d/%d results with rv %s and continue %s", len(list.Items), opts.Limit, list.ResourceVersion, list.Continue)

				if len(lastRV) == 0 {
					lastRV = list.ResourceVersion
				}
				if lastRV != list.ResourceVersion {
					Expect(list.ResourceVersion).To(Equal(lastRV))
				}
				for _, item := range list.Items {
					Expect(item.Name).To(Equal(fmt.Sprintf("template-%04d", found)))
					found++
				}
				if len(list.Continue) == 0 {
					break
				}
				opts.Continue = list.Continue
			}
			Expect(found).To(BeNumerically("==", numberOfTotalResources))
		}

		By("retrieving those results all at once")
		list, err := client.List(metav1.ListOptions{Limit: numberOfTotalResources + 1})
		Expect(err).ToNot(HaveOccurred())
		Expect(list.Items).To(HaveLen(numberOfTotalResources))
	})
})
