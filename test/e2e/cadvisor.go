/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"time"

	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"

	. "github.com/onsi/ginkgo"
)

const (
	timeout       = 1 * time.Minute
	maxRetries    = 6
	sleepDuration = 10 * time.Second
)

var _ = Describe("Cadvisor", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
	})

	It("should be healthy on every node.", func() {
		CheckCadvisorHealthOnAllNodes(c, 5*time.Minute)
	})
})

func CheckCadvisorHealthOnAllNodes(c *client.Client, timeout time.Duration) {
	By("getting list of nodes")
	nodeList, err := c.Nodes().List(labels.Everything(), fields.Everything())
	expectNoError(err)
	var errors []error
	retries := maxRetries
	for {
		errors = []error{}
		for _, node := range nodeList.Items {
			// cadvisor is not accessible directly unless its port (4194 by default) is exposed.
			// Here, we access '/stats/' REST endpoint on the kubelet which polls cadvisor internally.
			statsResource := fmt.Sprintf("api/v1/proxy/nodes/%s/stats/", node.Name)
			By(fmt.Sprintf("Querying stats from node %s using url %s", node.Name, statsResource))
			_, err = c.Get().AbsPath(statsResource).Timeout(timeout).Do().Raw()
			if err != nil {
				errors = append(errors, err)
			}
		}
		if len(errors) == 0 {
			return
		}
		if retries--; retries <= 0 {
			break
		}
		Logf("failed to retrieve kubelet stats -\n %v", errors)
		time.Sleep(sleepDuration)
	}
	Failf("Failed after retrying %d times for cadvisor to be healthy on all nodes. Errors:\n%v", maxRetries, errors)
}
