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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	//"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("ns-soak", func() {

	//This namespace is modified throughout the course of the test.
	var namespace *api.Namespace
	var c *client.Client
	var err error = nil
	BeforeEach(func() {
		By("Creating a kubernetes client")
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
	})

	// First test because it has no dependencies on variables created later on.
	totalNS := 100
	It("Namespaces should be deleted within 100 seconds", func() {
		for n := 0; n < totalNS; n += 1 {
			namespace, err = createTestingNS(fmt.Sprintf("nslifetest-%v", n), c)
		}

		//100 seconds after the ns are created, they should all be gone !
		deletionCompleted := time.After(time.Duration(150) * time.Second)
		deletionStart := time.After(time.Duration(10 * time.Second))
		tick := time.Tick(2 * time.Second)
	T:
		for {
			select {
			case <-deletionCompleted:
				Logf("Timeout %v reached. Breaking!", deletionCompleted)
				break T
			case <-deletionStart:
				print("STARTING DELETION")
				nsList, err := c.Namespaces().List(labels.Everything(), fields.Everything())
				Expect(err).NotTo(HaveOccurred())
				for _, item := range nsList.Items {
					if strings.Contains(item.Name, "nslifetest") {
						print(item.Name)
						if err := c.Namespaces().Delete(item.Name); err != nil {
							Failf("Failed deleting error ::: --- %v ", err)
						}
					}
					Logf("namespace : %v api call to delete is complete ", item)
				}

			case <-tick:
				nsList, err := c.Namespaces().List(labels.Everything(), fields.Everything())
				Expect(err).NotTo(HaveOccurred())
				var cnt = 0
				for _, item := range nsList.Items {
					if strings.Contains(item.Name, "nslifetest") {
						cnt++
					}
				}
				Logf("currently remaining NS == %v", cnt)
				//if all the ns's are gone, we break early
				if cnt == 0 {
					break T
				}
			}
		}

		nsList, err := c.Namespaces().List(labels.Everything(), fields.Everything())
		Expect(err).NotTo(HaveOccurred())
		for _, item := range nsList.Items {
			if strings.Contains(item.Name, "nslifetest") {
				Failf("FAILED.  There is still a remaining ns %v", item.Name)
			}
			Logf("namespace : %v", item)
		}

	})

})
