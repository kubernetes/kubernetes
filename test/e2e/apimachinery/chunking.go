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
	"context"
	"fmt"
	"math/rand"
	"reflect"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/test/e2e/framework"
)

const numberOfTotalResources = 400

var _ = SIGDescribe("Servers with support for API chunking", func() {
	f := framework.NewDefaultFramework("chunking")

	BeforeEach(func() {
		ns := f.Namespace.Name
		c := f.ClientSet
		client := c.CoreV1().PodTemplates(ns)
		By("creating a large number of resources")
		workqueue.ParallelizeUntil(context.TODO(), 20, numberOfTotalResources, func(i int) {
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
	})

	It("should return chunks of results for list calls", func() {
		ns := f.Namespace.Name
		c := f.ClientSet
		client := c.CoreV1().PodTemplates(ns)
		By("retrieving those results in paged fashion several times")
		for i := 0; i < 3; i++ {
			opts := metav1.ListOptions{}
			found := 0
			var lastRV string
			for {
				opts.Limit = int64(rand.Int31n(numberOfTotalResources/10) + 1)
				list, err := client.List(opts)
				Expect(err).ToNot(HaveOccurred(), "failed to list pod templates in namespace: %s, given limit: %d", ns, opts.Limit)
				framework.Logf("Retrieved %d/%d results with rv %s and continue %s", len(list.Items), opts.Limit, list.ResourceVersion, list.Continue)
				Expect(len(list.Items)).To(BeNumerically("<=", opts.Limit))

				if len(lastRV) == 0 {
					lastRV = list.ResourceVersion
				}
				Expect(list.ResourceVersion).To(Equal(lastRV))
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
		opts := metav1.ListOptions{Limit: numberOfTotalResources + 1}
		list, err := client.List(opts)
		Expect(err).ToNot(HaveOccurred(), "failed to list pod templates in namespace: %s, given limit: %d", ns, opts.Limit)
		Expect(list.Items).To(HaveLen(numberOfTotalResources))
	})

	It("should support continue listing from the last key if the original version has been compacted away, though the list is inconsistent", func() {
		ns := f.Namespace.Name
		c := f.ClientSet
		client := c.CoreV1().PodTemplates(ns)

		By("retrieving the first page")
		oneTenth := int64(numberOfTotalResources / 10)
		opts := metav1.ListOptions{}
		opts.Limit = oneTenth
		list, err := client.List(opts)
		Expect(err).ToNot(HaveOccurred(), "failed to list pod templates in namespace: %s, given limit: %d", ns, opts.Limit)
		firstToken := list.Continue
		firstRV := list.ResourceVersion
		framework.Logf("Retrieved %d/%d results with rv %s and continue %s", len(list.Items), opts.Limit, list.ResourceVersion, firstToken)

		By("retrieving the second page until the token expires")
		opts.Continue = firstToken
		var inconsistentToken string
		wait.Poll(20*time.Second, 2*storagebackend.DefaultCompactInterval, func() (bool, error) {
			_, err := client.List(opts)
			if err == nil {
				framework.Logf("Token %s has not expired yet", firstToken)
				return false, nil
			}
			if err != nil && !errors.IsResourceExpired(err) {
				return false, err
			}
			framework.Logf("got error %s", err)
			status, ok := err.(errors.APIStatus)
			if !ok {
				return false, fmt.Errorf("expect error to implement the APIStatus interface, got %v", reflect.TypeOf(err))
			}
			inconsistentToken = status.Status().ListMeta.Continue
			if len(inconsistentToken) == 0 {
				return false, fmt.Errorf("expect non empty continue token")
			}
			framework.Logf("Retrieved inconsistent continue %s", inconsistentToken)
			return true, nil
		})

		By("retrieving the second page again with the token received with the error message")
		opts.Continue = inconsistentToken
		list, err = client.List(opts)
		Expect(err).ToNot(HaveOccurred(), "failed to list pod templates in namespace: %s, given inconsistent continue token %s and limit: %d", ns, opts.Continue, opts.Limit)
		Expect(list.ResourceVersion).ToNot(Equal(firstRV))
		Expect(len(list.Items)).To(BeNumerically("==", opts.Limit))
		found := oneTenth
		for _, item := range list.Items {
			Expect(item.Name).To(Equal(fmt.Sprintf("template-%04d", found)))
			found++
		}

		By("retrieving all remaining pages")
		opts.Continue = list.Continue
		lastRV := list.ResourceVersion
		for {
			list, err := client.List(opts)
			Expect(err).ToNot(HaveOccurred(), "failed to list pod templates in namespace: %s, given limit: %d", ns, opts.Limit)
			framework.Logf("Retrieved %d/%d results with rv %s and continue %s", len(list.Items), opts.Limit, list.ResourceVersion, list.Continue)
			Expect(len(list.Items)).To(BeNumerically("<=", opts.Limit))
			Expect(list.ResourceVersion).To(Equal(lastRV))
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
	})
})
