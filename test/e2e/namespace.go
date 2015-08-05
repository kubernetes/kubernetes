/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"
	//"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/wait"
	"strings"
	"sync"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func countRemaining(c *client.Client, withName string) (int, error) {
	var cnt = 0
	nsList, err := c.Namespaces().List(labels.Everything(), fields.Everything())
	for _, item := range nsList.Items {
		if strings.Contains(item.Name, "nslifetest") {
			cnt++
		}
	}
	return cnt, err
}

func extinguish(c *client.Client, totalNS int, maxAllowedAfterDel int, maxSeconds int) {
	var err error

	By("Creating testing namespaces")
	wg := &sync.WaitGroup{}
	for n := 0; n < totalNS; n += 1 {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			defer GinkgoRecover()
			_, err = createTestingNS(fmt.Sprintf("nslifetest-%v", n), c)
			Expect(err).NotTo(HaveOccurred())
		}(n)
	}
	wg.Wait()

	By("Waiting 10 seconds")
	//Wait 10 seconds, then SEND delete requests for all the namespaces.
	time.Sleep(time.Duration(10 * time.Second))
	By("Deleting namespaces")
	nsList, err := c.Namespaces().List(labels.Everything(), fields.Everything())
	Expect(err).NotTo(HaveOccurred())
	var nsCount = 0
	for _, item := range nsList.Items {
		if strings.Contains(item.Name, "nslifetest") {
			wg.Add(1)
			nsCount++
			go func(nsName string) {
				defer wg.Done()
				defer GinkgoRecover()
				Expect(c.Namespaces().Delete(nsName)).To(Succeed())
				Logf("namespace : %v api call to delete is complete ", nsName)
			}(item.Name)
		}
	}
	Expect(nsCount).To(Equal(totalNS))
	wg.Wait()

	By("Waiting for namespaces to vanish")
	//Now POLL until all namespaces have been eradicated.
	expectNoError(wait.Poll(2*time.Second, time.Duration(maxSeconds)*time.Second,
		func() (bool, error) {
			if rem, err := countRemaining(c, "nslifetest"); err != nil || rem > maxAllowedAfterDel {
				Logf("Remaining namespaces : %v", rem)
				return false, err
			} else {
				return true, nil
			}
		}))

}

var _ = Describe("Namespaces", func() {

	//This namespace is modified throughout the course of the test.
	var c *client.Client
	var err error = nil
	BeforeEach(func() {
		By("Creating a kubernetes client")
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
	})

	//Confirms that namespace draining is functioning reasonably
	//at minute intervals.
	It("Delete 90 percent of 100 namespace in 150 seconds",
		func() { extinguish(c, 100, 10, 150) })

	//comprehensive draining ; uncomment after #7372
	PIt("Delete ALL of 100 namespace in 150 seconds",
		func() { extinguish(c, 100, 0, 150) })
})
