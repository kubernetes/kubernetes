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
	"reflect"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

const numberOfTotalResources = 400

var _ = SIGDescribe("Servers with support for API chunking", func() {
	f := framework.NewDefaultFramework("chunking")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func(ctx context.Context) {
		ns := f.Namespace.Name
		c := f.ClientSet
		client := c.CoreV1().PodTemplates(ns)
		ginkgo.By("creating a large number of resources")
		workqueue.ParallelizeUntil(ctx, 20, numberOfTotalResources, func(i int) {
			for tries := 3; tries >= 0; tries-- {
				_, err := client.Create(ctx, &v1.PodTemplate{
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
				}, metav1.CreateOptions{})
				if err == nil {
					return
				}
				framework.Logf("Got an error creating template %d: %v", i, err)
			}
			framework.Failf("Unable to create template %d, exiting", i)
		})
	})

	/*
		Release: v1.29
		Testname: API Chunking, server should return chunks of results for list calls
		Description: Create a large number of PodTemplates. Attempt to retrieve the first chunk with limit set;
		the server MUST return the chunk of the size not exceeding the limit with RemainingItems set in the response.
		Attempt to retrieve the remaining items by providing the received continuation token and limit;
		the server MUST return the remaining items in chunks of the size not exceeding the limit, with appropriately
		set RemainingItems field in the response and with the ResourceVersion returned in the first response.
		Attempt to list all objects at once without setting the limit; the server MUST return all items in a single
		response.
	*/
	framework.ConformanceIt("should return chunks of results for list calls", func(ctx context.Context) {
		ns := f.Namespace.Name
		c := f.ClientSet
		client := c.CoreV1().PodTemplates(ns)
		ginkgo.By("retrieving those results in paged fashion several times")
		for i := 0; i < 3; i++ {
			opts := metav1.ListOptions{}
			found := 0
			var lastRV string
			for {
				// With numberOfTotalResources=400, we want to ensure that both
				// number of items per page and number of pages are non-trivial.
				opts.Limit = 17
				list, err := client.List(ctx, opts)
				framework.ExpectNoError(err, "failed to list pod templates in namespace: %s, given limit: %d", ns, opts.Limit)
				framework.Logf("Retrieved %d/%d results with rv %s and continue %s", len(list.Items), opts.Limit, list.ResourceVersion, list.Continue)
				gomega.Expect(len(list.Items)).To(gomega.BeNumerically("<=", opts.Limit))

				if len(lastRV) == 0 {
					lastRV = list.ResourceVersion
				}
				gomega.Expect(list.ResourceVersion).To(gomega.Equal(lastRV))
				if list.GetContinue() == "" {
					gomega.Expect(list.GetRemainingItemCount()).To(gomega.BeNil())
				} else {
					gomega.Expect(list.GetRemainingItemCount()).ToNot(gomega.BeNil())
					gomega.Expect(int(*list.GetRemainingItemCount()) + len(list.Items) + found).To(gomega.BeNumerically("==", numberOfTotalResources))
				}
				for _, item := range list.Items {
					gomega.Expect(item.Name).To(gomega.Equal(fmt.Sprintf("template-%04d", found)))
					found++
				}
				if len(list.Continue) == 0 {
					break
				}
				opts.Continue = list.Continue
			}
			gomega.Expect(found).To(gomega.BeNumerically("==", numberOfTotalResources))
		}

		ginkgo.By("retrieving those results all at once")
		opts := metav1.ListOptions{Limit: numberOfTotalResources + 1}
		list, err := client.List(ctx, opts)
		framework.ExpectNoError(err, "failed to list pod templates in namespace: %s, given limit: %d", ns, opts.Limit)
		gomega.Expect(list.Items).To(gomega.HaveLen(numberOfTotalResources))
	})

	/*
		Release: v1.29
		Testname: API Chunking, server should support continue listing from the last key even if the original version has been compacted away
		Description: Create a large number of PodTemplates. Attempt to retrieve the first chunk with limit set;
		the server MUST return the chunk of the size not exceeding the limit with RemainingItems set in the response.
		Attempt to retrieve the second page until the continuation token expires; the server MUST return a
		continuation token for inconsistent list continuation.
		Attempt to retrieve the second page with the received inconsistent list continuation token; the server
		MUST return the number of items not exceeding the limit, a new continuation token and appropriately set
		RemainingItems field in the response.
		Attempt to retrieve the remaining pages by passing the received continuation token; the server
		MUST return the remaining items in chunks of the size not exceeding the limit, with appropriately
		set RemainingItems field in the response and with the ResourceVersion returned as part of the inconsistent list.
	*/
	framework.ConformanceIt("should support continue listing from the last key if the original version has been compacted away, though the list is inconsistent", f.WithSlow(), func(ctx context.Context) {
		ns := f.Namespace.Name
		c := f.ClientSet
		client := c.CoreV1().PodTemplates(ns)

		ginkgo.By("retrieving the first page")
		oneTenth := int64(numberOfTotalResources / 10)
		opts := metav1.ListOptions{}
		opts.Limit = oneTenth
		list, err := client.List(ctx, opts)
		framework.ExpectNoError(err, "failed to list pod templates in namespace: %s, given limit: %d", ns, opts.Limit)
		firstToken := list.Continue
		firstRV := list.ResourceVersion
		if list.GetContinue() == "" {
			gomega.Expect(list.GetRemainingItemCount()).To(gomega.BeNil())
		} else {
			gomega.Expect(list.GetRemainingItemCount()).ToNot(gomega.BeNil())
			gomega.Expect(int(*list.GetRemainingItemCount()) + len(list.Items)).To(gomega.BeNumerically("==", numberOfTotalResources))
		}
		framework.Logf("Retrieved %d/%d results with rv %s and continue %s", len(list.Items), opts.Limit, list.ResourceVersion, firstToken)

		ginkgo.By("retrieving the second page until the token expires")
		opts.Continue = firstToken
		var inconsistentToken string
		wait.Poll(20*time.Second, 2*storagebackend.DefaultCompactInterval, func() (bool, error) {
			_, err := client.List(ctx, opts)
			if err == nil {
				framework.Logf("Token %s has not expired yet", firstToken)
				return false, nil
			}
			if err != nil && !apierrors.IsResourceExpired(err) {
				return false, err
			}
			framework.Logf("got error %s", err)
			status, ok := err.(apierrors.APIStatus)
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

		ginkgo.By("retrieving the second page again with the token received with the error message")
		opts.Continue = inconsistentToken
		list, err = client.List(ctx, opts)
		framework.ExpectNoError(err, "failed to list pod templates in namespace: %s, given inconsistent continue token %s and limit: %d", ns, opts.Continue, opts.Limit)
		gomega.Expect(list.ResourceVersion).ToNot(gomega.Equal(firstRV))
		gomega.Expect(len(list.Items)).To(gomega.BeNumerically("==", opts.Limit))
		found := int(oneTenth)

		if list.GetContinue() == "" {
			gomega.Expect(list.GetRemainingItemCount()).To(gomega.BeNil())
		} else {
			gomega.Expect(list.GetRemainingItemCount()).ToNot(gomega.BeNil())
			gomega.Expect(int(*list.GetRemainingItemCount()) + len(list.Items) + found).To(gomega.BeNumerically("==", numberOfTotalResources))
		}
		for _, item := range list.Items {
			gomega.Expect(item.Name).To(gomega.Equal(fmt.Sprintf("template-%04d", found)))
			found++
		}

		ginkgo.By("retrieving all remaining pages")
		opts.Continue = list.Continue
		lastRV := list.ResourceVersion
		for {
			list, err := client.List(ctx, opts)
			framework.ExpectNoError(err, "failed to list pod templates in namespace: %s, given limit: %d", ns, opts.Limit)
			if list.GetContinue() == "" {
				gomega.Expect(list.GetRemainingItemCount()).To(gomega.BeNil())
			} else {
				gomega.Expect(list.GetRemainingItemCount()).ToNot(gomega.BeNil())
				gomega.Expect(int(*list.GetRemainingItemCount()) + len(list.Items) + found).To(gomega.BeNumerically("==", numberOfTotalResources))
			}
			framework.Logf("Retrieved %d/%d results with rv %s and continue %s", len(list.Items), opts.Limit, list.ResourceVersion, list.Continue)
			gomega.Expect(len(list.Items)).To(gomega.BeNumerically("<=", opts.Limit))
			gomega.Expect(list.ResourceVersion).To(gomega.Equal(lastRV))
			for _, item := range list.Items {
				gomega.Expect(item.Name).To(gomega.Equal(fmt.Sprintf("template-%04d", found)))
				found++
			}
			if len(list.Continue) == 0 {
				break
			}
			opts.Continue = list.Continue
		}
		gomega.Expect(found).To(gomega.BeNumerically("==", numberOfTotalResources))
	})
})
