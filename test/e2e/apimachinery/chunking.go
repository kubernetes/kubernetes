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

	"encoding/base64"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/util/uuid"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

func shouldCheckRemainingItem() bool {
	return utilfeature.DefaultFeatureGate.Enabled(features.RemainingItemCount)
}

const numberOfTotalResources = 400

var _ = SIGDescribe("Servers with support for API chunking", func() {
	specUUID := uuid.NewUUID()
	configMapLabelSelector := fmt.Sprintf("configMapUUIDGroup=%v", specUUID)
	f := framework.NewDefaultFramework("chunking")

	ginkgo.BeforeEach(func() {
		ns := f.Namespace.Name
		c := f.ClientSet
		client := c.CoreV1().ConfigMaps(ns)
		ginkgo.By("creating a large number of resources")
		workqueue.ParallelizeUntil(context.TODO(), 20, numberOfTotalResources, func(i int) {
			for tries := 3; tries >= 0; tries-- {
				_, err := client.Create(&v1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name: fmt.Sprintf("configmap-%04d", i),
						Labels: map[string]string{
							"configMapUUIDGroup": string(specUUID),
						},
					},
					Data: map[string]string{
						"testDataField": "testDataValue",
					},
				})
				if err == nil {
					return
				}
				framework.Logf("Got an error creating ConfigMap %d: %v", i, err)
			}
			framework.Failf("Unable to create ConfigMap %d, exiting", i)
		})
	})

	ginkgo.It("should return chunks of results for list calls", func() {
		ns := f.Namespace.Name
		c := f.ClientSet
		client := c.CoreV1().ConfigMaps(ns)
		ginkgo.By("retrieving those results in paged fashion several times")
		for i := 0; i < 3; i++ {
			opts := metav1.ListOptions{}
			found := 0
			var lastRV string
			var prevFound int
			for {
				opts.Limit = int64(rand.Int31n(numberOfTotalResources/10) + 1)
				list, err := client.List(opts)
				framework.ExpectNoError(err, "failed to list ConfigMaps in namespace: %s, given limit: %d", ns, opts.Limit)
				framework.Logf("Retrieved %d/%d results with rv %s and continue %s", len(list.Items), opts.Limit, list.ResourceVersion, list.Continue)
				gomega.Expect(len(list.Items)).To(gomega.BeNumerically("<=", opts.Limit))

				prevFound = found
				for _, item := range list.Items {
					framework.ExpectEqual(item.Name, fmt.Sprintf("configmap-%04d", found))
					found++
				}
				if len(lastRV) == 0 {
					lastRV = list.ResourceVersion
				}
				framework.ExpectEqual(list.ResourceVersion, lastRV)
				if shouldCheckRemainingItem() {
					if list.GetContinue() == "" {
						gomega.Expect(numberOfTotalResources - found).To(gomega.Equal(int(0)))
					} else {
						gomega.Expect(numberOfTotalResources - found).ToNot(gomega.Equal(int(0)))
						gomega.Expect((numberOfTotalResources - found) + len(list.Items) + prevFound).To(gomega.BeNumerically("==", numberOfTotalResources))
					}
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
		list, err := client.List(opts)
		framework.ExpectNoError(err, "failed to list ConfigMaps in namespace: %s, given limit: %d", ns, opts.Limit)
		gomega.Expect(list.Items).To(gomega.HaveLen(numberOfTotalResources))
	})

	ginkgo.It("should chunk lists of ConfigMaps", func() {
		ns := f.Namespace.Name
		c := f.ClientSet
		client := c.CoreV1().ConfigMaps(ns)
		loopCount := int64(0)
		var continueChunkHistory []string
		ginkgo.By("retrieving invidual chunk list of ConfigMap")
		e2elog.Logf("configMapLabelSelector: %v", configMapLabelSelector)
		for {
			opts := metav1.ListOptions{
				LabelSelector: string(configMapLabelSelector),
			}
			if loopCount > 0 && len(continueChunkHistory) > 0 {
				opts.Continue = continueChunkHistory[loopCount-1]
			}
			opts.Limit = 1
			list, err := client.List(opts)
			framework.ExpectNoError(err, "failed fetching ConfigMap list")

			if len(list.Continue) == 0 {
				e2elog.Logf("reached end of chunk list")
				break
			} else {
				e2elog.Logf("list.Continue: %v", list.Continue)
			}

			if loopCount > 0 && len(continueChunkHistory) > 0 {
				searchChunkHistory, _ := contains(continueChunkHistory, list.Continue)
				gomega.Expect(searchChunkHistory).NotTo(gomega.Equal(true), "chunks already exists")
			}
			continueChunkHistory = append(continueChunkHistory, list.Continue)
			loopCount++

			continueDecode, _ := base64.StdEncoding.DecodeString(list.Continue)
			e2elog.Logf("Fetched %d chunks; Continue: %s; Items: %s; Continue unwrapped value: %v", loopCount, list.Continue, fmt.Sprintf("%T", list.Items), string(continueDecode))

		}

		ginkgo.By("making sure that total number of resources created matches total number of resources discovered by fetching indiviual chunks")
		gomega.Expect(len(continueChunkHistory)+1).To(gomega.Equal(numberOfTotalResources), "number of resources created should match the number iterated through")

		ginkgo.By("retrieving those results all at once")
		opts := metav1.ListOptions{
			Limit:         numberOfTotalResources + 1,
			LabelSelector: string(configMapLabelSelector),
		}
		list, err := client.List(opts)
		framework.ExpectNoError(err, "failed to list all ConfigMaps in namespace: %s, given limit: %d", ns, opts.Limit)
		gomega.Expect(list.Items).To(gomega.HaveLen(numberOfTotalResources))
	})
})

func contains(arr []string, search string) (bool, int) {
	var posCont int
	for pos, val := range arr {
		if val == search {
			posCont = pos
			return true, pos
		}
	}
	return false, posCont
}
