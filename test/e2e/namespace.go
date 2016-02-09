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
	"strings"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func extinguish(c *client.Client, totalNS int, maxAllowedAfterDel int, maxSeconds int) {
	var err error

	By("Creating testing namespaces")
	wg := &sync.WaitGroup{}
	for n := 0; n < totalNS; n += 1 {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			defer GinkgoRecover()
			_, err = createTestingNS(fmt.Sprintf("nslifetest-%v", n), c, nil)
			Expect(err).NotTo(HaveOccurred())
		}(n)
	}
	wg.Wait()

	//Wait 10 seconds, then SEND delete requests for all the namespaces.
	By("Waiting 10 seconds")
	time.Sleep(time.Duration(10 * time.Second))
	deleted, err := deleteNamespaces(c, []string{"nslifetest"}, nil /* skipFilter */)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(deleted)).To(Equal(totalNS))

	By("Waiting for namespaces to vanish")
	//Now POLL until all namespaces have been eradicated.
	expectNoError(wait.Poll(2*time.Second, time.Duration(maxSeconds)*time.Second,
		func() (bool, error) {
			var cnt = 0
			nsList, err := c.Namespaces().List(api.ListOptions{})
			if err != nil {
				return false, err
			}
			for _, item := range nsList.Items {
				if strings.Contains(item.Name, "nslifetest") {
					cnt++
				}
			}
			if cnt > maxAllowedAfterDel {
				Logf("Remaining namespaces : %v", cnt)
				return false, nil
			}
			return true, nil
		}))
}

// This test must run [Serial] due to the impact of running other parallel
// tests can have on its performance.  Each test that follows the common
// test framework follows this pattern:
//   1. Create a Namespace
//   2. Do work that generates content in that namespace
//   3. Delete a Namespace
// Creation of a Namespace is non-trivial since it requires waiting for a
// ServiceAccount to be generated.
// Deletion of a Namespace is non-trivial and performance intensive since
// its an orchestrated process.  The controller that handles deletion must
// query the namespace for all existing content, and then delete each piece
// of content in turn.  As the API surface grows to add more KIND objects
// that could exist in a Namespace, the number of calls that the namespace
// controller must orchestrate grows since it must LIST, DELETE (1x1) each
// KIND.
// There is work underway to improve this, but it's
// most likely not going to get significantly better until etcd v3.
// Going back to this test, this test generates 100 Namespace objects, and then
// rapidly deletes all of them.  This causes the NamespaceController to observe
// and attempt to process a large number of deletes concurrently.  In effect,
// it's like running 100 traditional e2e tests in parallel.  If the namespace
// controller orchestrating deletes is slowed down deleting another test's
// content then this test may fail.  Since the goal of this test is to soak
// Namespace creation, and soak Namespace deletion, its not appropriate to
// further soak the cluster with other parallel Namespace deletion activities
// that each have a variable amount of content in the associated Namespace.
// When run in [Serial] this test appears to delete Namespace objects at a
// rate of approximately 1 per second.
var _ = Describe("Namespaces [Serial]", func() {

	f := NewFramework("namespaces")

	It("should delete fast enough (90 percent of 100 namespaces in 150 seconds)",
		func() { extinguish(f.Client, 100, 10, 150) })

	// On hold until etcd3; see #7372
	It("should always delete fast (ALL of 100 namespaces in 150 seconds) [Feature:ComprehensiveNamespaceDraining]",
		func() { extinguish(f.Client, 100, 0, 150) })
})
